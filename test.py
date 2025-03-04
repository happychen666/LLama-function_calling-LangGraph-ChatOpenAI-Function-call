from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_core.messages import HumanMessage, ToolMessage

tools = [OpenWeatherMapQueryRun()]

model = ChatOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            temperature=0)

functions = [convert_to_openai_function(t) for t in tools]

model = model.bind_tools(functions)


def function_1(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


from langgraph.prebuilt import ToolInvocation, ToolNode  # 引入 ToolNode
import json
from langchain_core.messages import FunctionMessage

# 使用 ToolNode 替代 ToolExecutor
tool_node = ToolNode(tools)

def function_2(state):
    messages = state['messages']
    last_message = messages[-1]  # 取最后一条消息，获取要发送给工具的查询
    print('last_message===\n',last_message)

    # 确保 tool_calls 存在且为非空列表
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    if tool_calls:
        function_data = tool_calls[0].get("function", {})  # 获取 function 字典
        arguments_str = function_data.get("arguments", "{}")  # 获取 arguments JSON 字符串
        parsed_tool_input = json.loads(arguments_str)  # 解析 JSON
        print('parsed_tool_input===\n', parsed_tool_input)
        tool_call_id=tool_calls[0]["id"]
    else:
        print("Warning: tool_calls is empty.")
    
    print('function_data===\n',function_data,function_data["name"])
    print('parsed_tool_input===\n',parsed_tool_input,parsed_tool_input['location'])
    # 构造 ToolInvocation
    action = ToolInvocation(
        tool=function_data["name"],
        tool_input=parsed_tool_input['location'],
    )

    # 使用 tool_node 处理请求
    response = tool_node.invoke(action)
    print('response===\n',response,'\n',action.tool)
    # 构造 FunctionMessage
    function_message = ToolMessage(response, tool_call_id=tool_call_id)

    # 返回消息列表
    return {"messages": [function_message]}


def where_to_go(state):
    messages = state['messages']
    last_message = messages[-1]
    
    if "tool_calls" in last_message.additional_kwargs:
        return "continue"
    else:
        return "end"
    

# from langgraph.graph import Graph, END

# workflow = Graph()

# Or you could import StateGraph and pass AgentState to it
from langgraph.graph import StateGraph, END
workflow = StateGraph(AgentState)

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)

# The conditional edge requires the following info below.
# First, we define the start node. We use `agent`.
# This means these are the edges taken after the `agent` node is called.
# Next, we pass in the function that will determine which node is called next, in our case where_to_go().

workflow.add_conditional_edges("agent", where_to_go,{   # Based on the return from where_to_go
                                                        # If return is "continue" then we call the tool node.
                                                        "continue": "tool",
                                                        # Otherwise we finish. END is a special node marking that the graph should finish.
                                                        "end": END
                                                    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that if `tool` is called, then it has to call the 'agent' next. 
workflow.add_edge('tool', 'agent')

# Basically, agent node has the option to call a tool node based on a condition, 
# whereas tool node must call the agent in all cases based on this setup.

workflow.set_entry_point("agent")


app = workflow.compile()


from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="how are you today?")]}#what is the temperature in las vegas
result = app.invoke(inputs)
print('type result=====\n\n\n',type(result))
print('result=====\n\n\n',result)
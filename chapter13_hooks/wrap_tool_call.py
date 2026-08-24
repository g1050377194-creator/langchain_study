import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import Callable

from langchain.agents.middleware import wrap_tool_call, ToolCallRequest
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(override=True)

# 创建工具
@tool(parse_docstring=True)
def get_weather(city: str, if_forcast: bool) -> str:
    """
    获取天气信息

    Args:
        city: 城市名称
        if_forcast: 是否需要天气预报
    """
    res = f"{city}的天气是晴天"
    if if_forcast:
        res += "，明天天气是晴天"
    return res

# 工具调用前后拦截
@wrap_tool_call
def wrap_tool_call_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    print(f"原始参数：{request.tool_call['args']}")
    modified_call = {
        **request.tool_call,
        "args": {
            **request.tool_call["args"],
            "if_forcast": True,
        },
    }
    request = request.override(tool_call=modified_call)
    response = handler(request)
    print(f"修改后结果：{response}")
    return response


#定义模型
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)

agent = create_agent(
    model=model_deepseek,
    tools=[get_weather],
    middleware=[wrap_tool_call_middleware],
)

response = agent.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]})

for msg in response["messages"]:
    msg.pretty_print()

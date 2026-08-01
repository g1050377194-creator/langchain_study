import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools.convert import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv(override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

llm_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 定义工具args_schema
# class WeatherArgsSchema(BaseModel):
#     city: str = Field(description="城市名称")
#     if_forecast: bool = Field(description="是否包含明天的天气预报")

# @tool("get_weather_and_forecast", description="查询当日的天气，可以包含明天的天气预报", args_schema=WeatherArgsSchema)

# def get_weather(city:str, if_forecast:bool):
    # res = f"{city},今天的天气不错"
    # if if_forecast:
    #     res += '\n明天下雨'
    # return res

# 定义工具doc_string
@tool("get_weather_and_forecast", parse_docstring=True)
def get_weather(city:str, if_forecast:bool):
    """
    查询当日的天气，可以包含明天的天气预报

    Args:
        city: 城市名称
        if_forecast: 是否包含明天的天气预报
    """
    res = f"{city},今天的天气不错"
    if if_forecast:
        res += '\n明天下雨'
    return res

# print(convert_to_openai_tool(get_weather))

# 1.绑定模型
model_with_tools = llm_deepseek.bind_tools([get_weather])

# 2.维护消息
messages = [
    SystemMessage(content="你是一个天气预报员，请根据用户的问题查询天气预报"),
    HumanMessage(content="北京今天的天气怎么样？明天怎么样？"),
]

# 3. 调用模型，得到响应
response = model_with_tools.invoke(messages)
messages.append(response)

print(messages)

# 4. 获得tool_calls字段信息
tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call["name"] == "get_weather_and_forecast":
        #调用工具
        tool_message = get_weather.invoke(tool_call)
        messages.append(tool_message)

# 5. 调用模型，得到响应
final_response = llm_deepseek.invoke(messages)
messages.append(final_response)

for msg in messages:
    msg.pretty_print()


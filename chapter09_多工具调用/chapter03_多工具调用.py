import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich import print as rprint
from langchain_core.tools import tool

load_dotenv(override=True)

#定义模型
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)
# 1.定义工具
# 定义股票工具
@tool(parse_docstring=True)
def get_stock_price(company:str, timeframe:str = "today") -> str:
    """
    获取股票价格

    Args:
        company: 公司名称
        timeframe: 时间范围(today, week, month)
    """

    mock_data = {
        "苹果公司": {
            "today": 100,
            "week": 110,
            "month": 120,
        },
        "谷歌公司": {
            "today": 200,
            "week": 210,
            "month": 220,
        },
        "微软公司": {
            "today": 300,
            "week": 310,
            "month": 320,
        },
    }
    if company in mock_data:
        price = mock_data[company][timeframe]
        return f"{company}的{timeframe}股票价格为{price}"
    else:
        return f"没有找到{company}的{timeframe}股票价格"

@tool(parse_docstring=True)
def search_news(company: str)-> str:
    """
    搜索公司的新闻

    Args:
        company: 公司名称
    
    Returns:
        公司的新闻
    """
    mock_data = {
        "苹果公司": [
            "苹果公司今天发布了新款手机",
            "苹果公司本周发布了新款手机",
            "苹果公司本月发布了新款手机",
        ],
        "谷歌公司": [
            "谷歌公司今天发布了新款手机",
            "谷歌公司本周发布了新款手机",
            "谷歌公司本月发布了新款手机",
        ],
        "微软公司": [
            "微软公司今天发布了新款手机",
            "微软公司本周发布了新款手机",
            "微软公司本月发布了新款手机",
        ],
    }
    if company in mock_data:
        return mock_data[company]
    else:
        return f"没有找到{company}的新闻"

tools = [get_stock_price, search_news]
#绑定工具
model_with_tools = model_deepseek.bind_tools(tools)

message_list = []
human_message = HumanMessage(content="我想查询苹果公司的股票价格?最近有什么新闻?")
message_list.append(human_message)

# 3.工具调用
while True:
    response = model_with_tools.invoke(message_list)

    rprint(response)
    # 先把 AI 的 tool_calls 消息加入历史，再追加 ToolMessage
    message_list.append(response)

    if not response.tool_calls:
        print('没有工具可以调用了')
        break

    # 工具的具体调用
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_stock_price":
            stock_result = get_stock_price.invoke(tool_call)
            print(stock_result, 'stock_result')
            message_list.append(stock_result)
        if tool_call["name"] == "search_news":
            news_result = search_news.invoke(tool_call)
            print(news_result, 'news_result')
            message_list.append(news_result)

# 输出
for message in message_list:
    message.pretty_print()


import sys
sys.stdout.reconfigure(encoding="utf-8")
# 创建模型


import os
# from chapter09_多工具调用.chapter03_多工具调用 import response
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

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

# 定义工具
@tool(prase_docstring=True)
def search_custmer_database(query: str) -> str:
    """
    查询客户数据库

    Args:
        query(str): 查询条件
    
    Returns:
        str: 查询结果
    """
    # 模拟数据库查询结果
    if "张三" in query.lower():
        return "客户记录：张三，VIP客户，最近购买日期：2026-01-01，累计消费：$15,000"
    elif "李四" in query.lower():
        return "客户记录：李四，普通客户，最近购买日期：2026-02-15，累计消费：$5,000"
    
    else:
        return "未找到相关客户记录"
    
@tool(prase_docstring=True)
def send_email(customer: str) -> str:
    """

    Args:
        customer(str): 客户名称
    
    Returns:
        str: 发送结果
    """
    # 模拟发送邮件结果
    return f"邮件发送成功：{customer}"

# 定义模式
class CustomerServiceAgent(BaseModel):
    customer: str = Field(description="客户名称")

# 创建智能体
agent = create_agent(
    model=model,
    system_prompt=SystemMessage(content="清分析指定客户的情况："
    "1.先搜索客户数据库了解最新情况"
    "2.如果是VIP客户，则发送邮件解答"
    "3.基于搜索结果生成结构化分析报告"
    "4.如果用户提问与客户记录无关火找不到客户信息，则返回空对象"
    )
)
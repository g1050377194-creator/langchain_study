import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_deepseek import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv(override=True)

# 从 .env 读取，或直接写连接串
DB_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://postgres:你的密码@localhost:5432/langchain_memory"
)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 使用 PostgreSQL 持久化记忆
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 首次运行必须调用，自动建表

    agent = create_agent(
        model=model_deepseek,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1234567890"}}

    print("\n第一轮对话：")
    r1 = agent.invoke(
        {"messages": [HumanMessage(content="你好，我是小明，很高兴认识你。")]},
        config=config,
    )
    print(f"Agent: {r1['messages'][-1].content}")

    print("\n第二轮对话：")
    r2 = agent.invoke(
        {"messages": [HumanMessage(content="我叫什么？")]} ,
        config=config,
    )
    print(f"Agent: {r2['messages'][-1].content}")
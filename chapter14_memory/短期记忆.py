import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_deepseek import init_chat_model

load_dotenv(override=True)

checkpointer = InMemorySaver()


# 定义模型
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 调用模型
agent = create_agent(
    model=model_deepseek,
    checkpointer=checkpointer,
)

# 同一个thread_id，记忆是共享的
config = {
    "configurable": {
        "thread_id": "1234567890",
    }
}

print("\n第一轮对话：")
response = agent.invoke({"messages":[HumanMessage(content="你好，我是小明，很高兴认识你。")]}, config=config)
print(f"Agent:{response['messages'][-1].content}")

print("\n第二轮对话：")
response2 = agent.invoke({"messages":[HumanMessage(content="你叫什么？")]}, config=config)
print(f"Agent:{response2['messages'][-1].content}")

print("\n第三轮对话：")
response3 = agent.invoke({"messages":[HumanMessage(content="我前面问了什么问题？")]}, config=config)
print(f"Agent:{response3['messages'][-1].content}")

from langchain_core.messages import HumanMessage
from langchain.agents.middleware import before_model, after_model, before_agent, after_agent
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from typing import Any
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent
from rich import print as rprint
from langchain.agents.middleware import AgentMiddleware

# 装饰器
# @before_model
# def before_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     state["messages"][-1].content += "--------->before model<-----------"
#     return None

# @after_model
# def after_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     state["messages"][-1].content += "--------->after model<-----------"
#     return None

# @before_agent
# def before_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     state["messages"][-1].content += "--------->before agent<-----------"
#     return None

# @after_agent
# def after_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     state["messages"][-1].content += "--------->after agent<-----------"
#     return None

# 类的实现
class MyMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += "--------->before model<-----------"
        return None
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += "--------->after model<-----------"
        return None
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += "--------->before agent<-----------"
        return None
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += "--------->after agent<-----------"
        return None

load_dotenv(override=True)

#定义模型
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

middleware = MyMiddleware()

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)

agent = create_agent(
    model=model_deepseek,
    tools=[],
    middleware=[middleware],
)

response = agent.invoke({"messages":[HumanMessage(content="请分析客户情况")]})

for msg in response["messages"]:
    rprint(msg.content)
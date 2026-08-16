import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents.middleware import wrap_model_call
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent
from rich import print as rprint
from langchain_core.messages import HumanMessage

# 模型的前后以及调用的过程
@wrap_model_call
def wrap_model_call_middleware(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse | None:
    request.messages[-1].content += "--------->wrap model call<-----------"
    # 模型的调用
    response = handler(request)
    response.result[0].content += "--------->wrap model call<-----------"
    return response

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

agent = create_agent(
    model=model_deepseek,
    middleware=[wrap_model_call_middleware],
)

response = agent.invoke({"messages":[HumanMessage(content="您好")]})
# print(response)

for msg in response["messages"]:
    msg.pretty_print()
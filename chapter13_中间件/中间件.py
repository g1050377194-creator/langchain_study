# 绑定模型
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Notebook 在子目录里时，默认找不到项目根目录的 .env，需显式指定
ENV_PATH = Path("..") / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

#定义模型
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
print("TAVILY_API_KEY loaded:", bool(TAVILY_API_KEY))

model_deepseek = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=API_KEY,
    base_url=BASE_URL,
)


create_agent(
    model=model_deepseek,
    tools=[],
    middleware=[
        PIIMiddleware()
    ]
)
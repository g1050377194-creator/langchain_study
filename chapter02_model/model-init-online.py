from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

# 加载配置文件
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

print(model.invoke('你是谁'))
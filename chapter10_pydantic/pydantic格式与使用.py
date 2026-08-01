import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

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

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="人物名称")
    # age: int = Field(description="人物年龄")
    #可选字段
    age: Optional[int] = Field(description="人物年龄")

    occupation: str = Field(description="人物职业")

structured_llm = model_deepseek.with_structured_output(Person)
response = structured_llm.invoke("请介绍一下李华，他是一个的工程师")
print(response)
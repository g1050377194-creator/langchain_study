import sys
sys.stdout.reconfigure(encoding="utf-8")


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

class MovieData(TypedDict):
    title: str
    year: int
    director: str
    rating: float

movie: MovieData = {
    "title": "The Dark Knight",
    "year": 2008,
    "director": "Christopher Nolan",
    "rating": 9.0,
}
# 定义嵌套
class Actor(TypedDict):
    name: str
    role: str

class MovieTypeDict(TypedDict):
    title: Annotated[str, "电影标题"]
    year: Annotated[int, "上映年份"]
    director: Annotated[str, "导演"]
    rating: Annotated[float, ...,"评分"]
    actors: Annotated[list[Actor], "演员列表"]

# 定义必填


structured_model = model_deepseek.with_structured_output(MovieTypeDict)

response = structured_model.invoke("请介绍一下《The Dark Knight》这部电影,这部电影的导演是Christopher Nolan,演员有Christian Bale, Heath Ledger, Aaron Eckhart")



print(response)
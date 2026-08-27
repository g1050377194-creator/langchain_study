import sys
import os
from dataclasses import dataclass
from typing_extensions import TypedDict

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv(override=True)

# ============================================================
# 长期记忆 vs 短期记忆（先搞清概念）
# ------------------------------------------------------------
# 短期记忆（checkpointer + thread_id）：
#   - 同一会话线程内的消息历史
#   - 换一个 thread_id，就“不记得”上一轮对话了
#
# 长期记忆（store + user_id）：
#   - 跨会话、跨 thread 保存用户画像/偏好等事实
#   - 只要 user_id 相同，新开一个对话也能读到
# ============================================================

API_KEY = os.getenv("QWEN_API_KEY")
BASE_URL = os.getenv("QWEN_BASE_URL")

# Qwen 走 DashScope 的 OpenAI 兼容接口，provider 要用 openai（没有 qwen）
model_qwen = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
)


# 通过 context 传入当前用户身份（不是 thread_id）
@dataclass
class Context:
    user_id: str


# 给模型看的结构化字段，方便它调用保存工具
class UserInfo(TypedDict):
    name: str
    hobby: str


# 开发学习用内存 Store；生产可换成 PostgresStore
store = InMemoryStore()
# 短期记忆仍然用 checkpointer（可选，但建议一起学）
checkpointer = InMemorySaver()


@tool(parse_docstring=True)
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """
    把用户信息写入长期记忆。

    Args:
        user_info: 要保存的用户信息，包含 name 和 hobby
    """
    assert runtime.store is not None
    user_id = runtime.context.user_id
    # namespace 类似文件夹，key 类似文件名
    runtime.store.put(("users",), user_id, dict(user_info))
    return f"已保存用户信息: {user_info}"


@tool(parse_docstring=True)
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """
    从长期记忆中读取当前用户信息。
    """
    assert runtime.store is not None
    user_id = runtime.context.user_id
    item = runtime.store.get(("users",), user_id)
    if item is None:
        return "暂无该用户的长期记忆"
    return str(item.value)


agent = create_agent(
    model=model_qwen,
    tools=[save_user_info, get_user_info],
    system_prompt=(
        "你是一个会使用长期记忆的助手。"
        "当用户告诉你姓名或爱好时，调用 save_user_info 保存；"
        "当用户询问自己的信息时，调用 get_user_info 查询。"
    ),
    # 长期记忆
    store=store,
    # 短期记忆（同一 thread 内的聊天记录）
    checkpointer=checkpointer,
    context_schema=Context,
)

USER_ID = "user_xiaoming"
context = Context(user_id=USER_ID)

# ---------- 会话 A：新开一个 thread，写入长期记忆 ----------
config_a = {"configurable": {"thread_id": "session-A"}}

print("\n【会话A】写入长期记忆")
r1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫小明，爱好是打篮球，请记住。")]},
    config=config_a,
    context=context,
)
print(f"Agent: {r1['messages'][-1].content}")

# ---------- 会话 B：换一个 thread_id，短期记忆应丢失，长期记忆仍在 ----------
config_b = {"configurable": {"thread_id": "session-B"}}

print("\n【会话B】新开对话（不同 thread_id），靠长期记忆回忆")
r2 = agent.invoke(
    {"messages": [HumanMessage(content="你还记得我是谁吗？我有什么爱好？")]},
    config=config_b,
    context=context,
)
print(f"Agent: {r2['messages'][-1].content}")

# 也可以不经过 Agent，直接读 Store，方便调试
print("\n【直接读 Store】")
saved = store.get(("users",), USER_ID)
print(saved.value if saved else "空")

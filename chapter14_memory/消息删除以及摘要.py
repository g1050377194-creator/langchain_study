


from ast import In
from email import message
from re import A
from langchain_core.runnables import Runnable, RunnableConfig
from chapter13_hooks.自定义中间件 import middleware
from chapter14_memory.短期记忆 import checkpointer


@after_model
def delere_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    # 保持最近的5条消息
    if len(messages) > 5:
        # 通常使用RemoveMessage来标记删除，并返回更新状态
        to_delete = len(messages) - 5
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:to_delete]]}
    return None

# agent = create_agent(
#     model = model,
#     middleware =  [delete_old_messages],
#     checkpointer = InMemorySaver()
# )

# 摘要
agent = create_agent(
    model=model_out,
    tools = [],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model_in,
            trigger=[
                ("tokens", 100), # 超过 100 tokens 就摘要
            ],
            keep=("messages", 2),
            summary_prompt="对历史摘要，消息列表如下\n{message}"
        )
    ]
)


config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({'messages': "你好，我是miki"}, config=config)
agent.invoke({"messages": "从现在起，你叫miki"}, config=config)
agent.invoke({"messages": "今天天气怎么样"}, config=config)
final_response = agent.invoke({"messages": "告诉我，你是谁？我是谁？"}, config=config)

for msg in final_response["messages"]:
    msg.pretty_print()
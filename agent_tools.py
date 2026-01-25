import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. СОЗДАЕМ ИНСТРУМЕНТ (Tool)
@tool
def multiply(a: int, b: int) -> int:
    """Умножает два числа. Используй это для вычислений."""
    return a * b

tools = [multiply]

# 2. НАСТРАИВАЕМ LLM
# model="gpt-4o" или "gpt-3.5-turbo"
llm = ChatOpenAI(model="gpt-4o") 
# "Привязываем" инструменты к модели, чтобы она знала о них
llm_with_tools = llm.bind_tools(tools)

# 3. СОСТОЯНИЕ
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. УЗЛЫ
def agent(state: State):
    # Агент решает: ответить текстом или вызвать инструмент
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. СБОРКА ГРАФА
builder = StateGraph(State)

# Добавляем узлы
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools)) # Готовый узел для запуска функций

# Старт -> Агент
builder.add_edge(START, "agent")

# Условный переход:
# Если Агент решил вызвать функцию -> идем в "tools"
# Если Агент просто ответил -> идем в END
builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# Если сработал инструмент, возвращаем результат агенту, чтобы он сформулировал ответ
builder.add_edge("tools", "agent")

# Компилируем с памятью
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 6. ЗАПУСК
config = {"configurable": {"thread_id": "math_expert"}}

print("🤖 Агент-математик готов! (Напишите 'exit' для выхода)")

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, 
        config, 
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            # Показываем только ответы бота
            if last_msg.type == "ai": 
                print(f"🤖 Бот: {last_msg.content}")

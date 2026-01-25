import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o")

# 1. СОСТОЯНИЕ
class State(TypedDict):
    query: str
    response: str

# 2. УЗЛЫ-СПЕЦИАЛИСТЫ
def math_expert(state: State):
    print("🧮 Работает математик...")
    msg = [SystemMessage(content="Ты математик. Отвечай только цифрами и формулами."),
           HumanMessage(content=state['query'])]
    return {"response": llm.invoke(msg).content}

def writer_expert(state: State):
    print("✍️ Работает писатель...")
    msg = [SystemMessage(content="Ты поэт. Отвечай стихами."),
           HumanMessage(content=state['query'])]
    return {"response": llm.invoke(msg).content}

# 3. ЛОГИКА МАРШРУТИЗАЦИИ (ROUTER)
# Эта функция классифицирует запрос
def route_query(state: State) -> Literal["math", "writer"]:
    query = state['query']
    print(f"🚦 Менеджер анализирует: '{query}'")
    
    # Спрашиваем LLM, к какой категории относится запрос
    classifier = llm.invoke([
        SystemMessage(content="Твоя задача: классифицировать запрос. Ответь только одним словом: 'MATH' (если это вычисления) или 'WRITER' (если это просьба написать текст/стих/эссе)."),
        HumanMessage(content=query)
    ])
    
    category = classifier.content.strip().upper()
    
    if "MATH" in category:
        return "math"
    else:
        return "writer"

# 4. СБОРКА ГРАФА
builder = StateGraph(State)

builder.add_node("math_node", math_expert)
builder.add_node("writer_node", writer_expert)

# Условный старт: сразу решаем, куда идти
builder.add_conditional_edges(
    START,
    route_query,
    {
        "math": "math_node",
        "writer": "writer_node"
    }
)

builder.add_edge("math_node", END)
builder.add_edge("writer_node", END)

app = builder.compile()

# 5. ТЕСТЫ
print("\n--- ТЕСТ 1 ---")
res1 = app.invoke({"query": "Сколько будет 25 + 25?"})
print(f"🤖 Ответ: {res1['response']}")

print("\n--- ТЕСТ 2 ---")
res2 = app.invoke({"query": "Напиши четверостишие про Python"})
print(f"🤖 Ответ: {res2['response']}")


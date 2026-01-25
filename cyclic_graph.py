import os
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Ключ уже в памяти терминала, но можно раскомментировать
# os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-4o")

# 1. СОСТОЯНИЕ
class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    iteration: int # Чтобы не зациклиться навечно

# 2. УЗЛЫ
def generator(state: State):
    print(f"\n💡 Генератор (Попытка {state.get('iteration', 1)}): Пишу шутку...")
    
    msg = f"Придумай шутку про {state['topic']}."
    if state.get("feedback"):
        msg += f"\nКритик сказал исправить: {state['feedback']}"
    
    response = llm.invoke([HumanMessage(content=msg)])
    return {"joke": response.content, "iteration": state.get("iteration", 0) + 1}

def critic(state: State):
    print("⚖️  Критик: Оцениваю...")
    joke = state['joke']
    
    # Инструкция для LLM-критика
    prompt = f"""Ты строгий критик юмора. Оцени шутку: "{joke}"
    Если шутка смешная и короткая, ответь только словом: OK
    Если нет, напиши, что исправить (одним предложением)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    review = response.content
    
    print(f"   Вердикт: {review}")
    return {"feedback": review}

# 3. ЛОГИКА ПЕРЕХОДА (Router)
def route_step(state: State):
    feedback = state['feedback']
    iteration = state['iteration']
    
    # Если ОК или уже 3 попытки — заканчиваем
    if feedback == "OK" or iteration >= 3:
        return "end"
    else:
        return "retry"

# 4. СБОРКА ГРАФА
builder = StateGraph(State)

builder.add_node("generator", generator)
builder.add_node("critic", critic)

builder.set_entry_point("generator")

# Связи
builder.add_edge("generator", "critic")

# УСЛОВНЫЙ ПЕРЕХОД
builder.add_conditional_edges(
    "critic",          # От кого
    route_step,        # Функция-решала
    {                  # Карта путей
        "end": END,
        "retry": "generator"
    }
)

graph = builder.compile()

# 5. ЗАПУСК
print("🚀 Запускаем цикл...")
graph.invoke({"topic": "Java разработчиков", "iteration": 0})

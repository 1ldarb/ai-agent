import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Вставьте ключ, если он не в переменных окружения
# os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-4o")

# 1. СОСТОЯНИЕ
class State(TypedDict):
    topic: str
    joke: str

# 2. УЗЛЫ (С ИСКУССТВЕННЫМ ИНТЕЛЛЕКТОМ)
def generator_node(state: State):
    print("💡 Генератор: Придумываю шутку...")
    msg = f"Придумай короткую шутку про {state['topic']}."
    response = llm.invoke([HumanMessage(content=msg)])
    return {"joke": response.content}

def editor_node(state: State):
    print("✍️ Редактор: Улучшаю стиль...")
    # Берем шутку из состояния и просим улучшить
    original_joke = state['joke']
    msg = f"Сделай эту шутку более дерзкой и короткой: {original_joke}"
    response = llm.invoke([HumanMessage(content=msg)])
    return {"joke": response.content} # Обновляем шутку в состоянии

# 3. СБОРКА ГРАФА
builder = StateGraph(State)

builder.add_node("generator", generator_node)
builder.add_node("editor", editor_node)

# Логика: Старт -> Генератор -> Редактор -> Конец
builder.set_entry_point("generator")
builder.add_edge("generator", "editor")
builder.add_edge("editor", END)

graph = builder.compile()

# 4. ЗАПУСК
print("🚀 Запускаем AI-граф...")
result = graph.invoke({"topic": "программистов на Python"})

print("\n🏁 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
print(result['joke'])

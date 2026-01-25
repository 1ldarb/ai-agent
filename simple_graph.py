from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. СОСТОЯНИЕ (State) — это "память" графа
# Она передается от узла к узлу.
class GraphState(TypedDict):
    message: str

# 2. УЗЛЫ (Nodes) — наши функции-работники
def worker_1(state: GraphState):
    print("🤖 Работник 1: Получил сообщение, добавляю подпись.")
    new_msg = state['message'] + " -> [Одобрено №1]"
    return {"message": new_msg}

def worker_2(state: GraphState):
    print("🤖 Работник 2: Финализирую.")
    new_msg = state['message'] + " -> [Готово]"
    return {"message": new_msg}

# 3. СБОРКА ГРАФА
workflow = StateGraph(GraphState)

# Добавляем узлы
workflow.add_node("step_1", worker_1)
workflow.add_node("step_2", worker_2)

# Строим поток: Вход -> step_1 -> step_2 -> Конец
workflow.set_entry_point("step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", END)

# Компиляция
app = workflow.compile()

# 4. ЗАПУСК
print("🚀 Запускаем граф...")
result = app.invoke({"message": "Старт"})
print(f"🏁 Итог: {result['message']}")


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
# Импортируем готовый граф из вашего прошлого файла
# (Убедитесь, что в router_graph.py код создания графа не под "if __name__ == '__main__':")
# Для простоты я скопирую нужную часть сюда, чтобы всё работало из коробки:

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- НАСТРОЙКА ГРАФА (Как было) ---
llm = ChatOpenAI(model="gpt-4o")

class State(TypedDict):
    query: str
    response: str

def math_expert(state: State):
    msg = [SystemMessage(content="Ты математик. Отвечай цифрами."), HumanMessage(content=state['query'])]
    return {"response": llm.invoke(msg).content}

def writer_expert(state: State):
    msg = [SystemMessage(content="Ты поэт."), HumanMessage(content=state['query'])]
    return {"response": llm.invoke(msg).content}

def route_query(state: State) -> Literal["math", "writer"]:
    # Упрощенная логика для скорости (или используйте LLM-классификатор из router_graph.py)
    query = state['query'].lower()
    if any(x in query for x in ["сколько", "умножь", "+", "-", "/"]):
        return "math"
    return "writer"

builder = StateGraph(State)
builder.add_node("math", math_expert)
builder.add_node("writer", writer_expert)
builder.add_conditional_edges(START, route_query, {"math": "math", "writer": "writer"})
builder.add_edge("math", END)
builder.add_edge("writer", END)
agent_app = builder.compile()

# --- ВЕБ-СЕРВЕР (FastAPI) ---
app = FastAPI(title="AI Agent API")

# Формат входных данных
class Request(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: Request):
    print(f"📨 Получен запрос: {request.query}")
    # Запускаем граф
    result = agent_app.invoke({"query": request.query})
    return {"reply": result["response"]}

# Запуск: python server.py
if __name__ == "__main__":
    print("🚀 Сервер запускается на http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages # Правильный импорт
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

# Ключ уже в памяти терминала
llm = ChatOpenAI(model="gpt-4o")

class State(TypedDict):
    # add_messages сохраняет историю, а не перезаписывает её
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Подключаем память
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

print("💬 Чат с памятью (напишите 'exit' для выхода):")

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    input_message = HumanMessage(content=user_input)
    
    # Запускаем, передавая config (thread_id)
    for event in app.stream({"messages": [input_message]}, config=config):
        for value in event.values():
            print(f"🤖 Бот: {value['messages'][-1].content}")

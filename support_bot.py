import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Загружаем базу знаний ---
try:
    with open("faq.txt", "r", encoding="utf-8") as f:
        faq_data = f.read()
except FileNotFoundError:
    print("Ошибка: Файл faq.txt не найден!")
    exit()

# --- 2. Настраиваем Модель и Промпт ---
llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = """Ты — вежливый сотрудник поддержки магазина техники 'TechStore'.
Твоя задача — отвечать на вопросы клиентов, используя ТОЛЬКО предоставленную ниже базу знаний.

База знаний:
{context}

ВАЖНО:
1. Если ответа нет в базе, отвечай: "К сожалению, у меня нет этой информации. Пожалуйста, свяжитесь с менеджером по телефону."
2. Не придумывай условия, которых нет в тексте.
3. Будь краток и дружелюбен.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])

# Создаем цепочку: Промпт -> Модель -> Текст
chain = prompt | llm | StrOutputParser()

# --- 3. Запуск чата ---
if __name__ == "__main__":
    print("🤖 Бот поддержки TechStore готов! (Напишите 'выход' для завершения)\n")
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("Бот: До свидания!")
            break
            
        # Запускаем цепочку, передавая базу знаний и вопрос
        response = chain.invoke({"context": faq_data, "question": user_input})
        print(f"Бот: {response}\n")

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Вставьте свой ключ
api_key = "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"

# 1. МОДЕЛЬ (Model)
model = ChatOpenAI(api_key=api_key, model="gpt-4o")

# 2. ПРОМПТ (Prompt)
# Мы создаем шаблон, куда будем подставлять переменные
prompt = ChatPromptTemplate.from_template("Расскажи короткий и смешной факт про {topic}")

# 3. ЦЕПОЧКА (Chain)
# Самое важное: Промпт -> Модель -> Преобразование в строку
chain = prompt | model | StrOutputParser()

# 4. ЗАПУСК
print("⏳ Думаю...")
response = chain.invoke({"topic": "программистов на Python"})
print(f"💬 Ответ:\n{response}")

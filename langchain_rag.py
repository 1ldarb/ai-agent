import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Вставьте ключ
api_key = "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"

# 1. ПОДКЛЮЧЕНИЕ К БАЗЕ (Ту же, что создали на 4-й неделе)
# Важно: используем ту же модель эмбеддингов!
embedding_function = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Загружаем базу с диска
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
    collection_name="faq_kb"
)

# Делаем из базы "Ретривер" (Инструмент поиска)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 2. МОДЕЛЬ
llm = ChatOpenAI(api_key=api_key, model="gpt-4o")

# 3. ПРОМПТ
template = """Ответь на вопрос, основываясь ТОЛЬКО на следующем контексте:
{context}

Вопрос: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. ЦЕПОЧКА (RAG Chain)
# Самая магия тут:
# - context заполняется автоматически через retriever
# - question берется из ввода пользователя
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. ЗАПУСК
print("🤖 Бот готов! (Нажмите Ctrl+C для выхода)")
while True:
    q = input("\n👤 Ваш вопрос: ")
    print("⏳ Ищу ответ...")
    response = rag_chain.invoke(q)
    print(f"💬 Ответ: {response}")

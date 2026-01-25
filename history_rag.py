import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# --- НАСТРОЙКИ ---
# Твой ключ (не забудь потом перенести в .env для безопасности!)
API_KEY = "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"

# 1. ПОДКЛЮЧЕНИЕ К БАЗЕ CHROMA
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings, 
    collection_name="faq_kb"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o")

# 3. ЛОГИКА LCEL (Цепочки через '|')
# Промпт для переформулирования вопроса с учетом истории
condense_q_system = "На основе истории чата и вопроса, сделай самостоятельный вопрос на русском языке. НЕ ОТВЕЧАЙ на него."
condense_q_prompt = ChatPromptTemplate.from_messages([
    ("system", condense_q_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Промпт для финального ответа
qa_system = "Ты профессиональный ассистент. Используй только контекст для ответа. Если ответа нет — скажи 'не знаю'.\n\n{context}"
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Вспомогательная функция для форматирования найденных документов
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. ФУНКЦИЯ ПОЛУЧЕНИЯ ОТВЕТА
def get_answer(user_input, history):
    # А. Генерируем "чистый" вопрос без ссылок на историю (типа "А как это?")
    standalone_chain = condense_q_prompt | llm | StrOutputParser()
    standalone_query = standalone_chain.invoke({"input": user_input, "chat_history": history})
    
    # Б. Ищем в базе Chroma
    docs = retriever.invoke(standalone_query)
    context = format_docs(docs)
    
    # В. Генерируем финальный ответ
    response_chain = qa_prompt | llm | StrOutputParser()
    return response_chain.invoke({
        "context": context,
        "chat_history": history,
        "input": user_input
    })

# 5. ЦИКЛ ЧАТА
chat_history = []
print("\n🤖 Бот с памятью на Python 3.12 готов! (Напиши 'выход' для завершения)")

while True:
    query = input("\n👤 Вы: ")
    if query.lower() in ["exit", "выход"]:
        print("До встречи!")
        break

    try:
        answer = get_answer(query, chat_history)
        print(f"💬 Бот: {answer}")

        # Обновляем историю сообщений
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        
        # Ограничиваем историю 6-ю сообщениями для экономии токенов
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
            
    except Exception as e:
        print(f"❌ Ошибка в работе: {e}")

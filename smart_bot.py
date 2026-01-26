import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Подготовка Базы Знаний (RAG) ---
print("⚙️  Индексирую базу знаний...")

# Загружаем текст
loader = TextLoader("faq.txt", encoding="utf-8")
documents = loader.load()

# Разбиваем на кусочки (чтобы искать точечно)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# Создаем Векторную Базу (Chroma)
# Она превращает текст в цифры (эмбеддинги) для поиска по смыслу
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="techstore_faq"
)

# Создаем "Искателя" (Retriever)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # Искать 1 самый похожий кусок

# --- 2. Настройка Мозга ---
llm = ChatOpenAI(model="gpt-4o-mini")

template = """Ответь на вопрос, основываясь ТОЛЬКО на следующем контексте:
{context}

Вопрос: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Цепочка: (Поиск + Вопрос) -> Промпт -> Модель -> Ответ
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 3. Запуск ---
if __name__ == "__main__":
    print("\n🧠 Умный RAG-бот готов! (Пишите 'выход')\n")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["выход", "exit"]:
            break
            
        # Бот сам найдет нужный кусок текста и ответит
        response = rag_chain.invoke(user_input)
        print(f"Бот: {response}\n")

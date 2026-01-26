import uvicorn
import os # ДОБАВЛЕНО: для работы с путями
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Импортируем "мозги" (LangChain компоненты)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Подготовка RAG (Копируем логику из smart_bot.py) ---
print("⚙️  Загружаю сервер и базу знаний...")

# ДОБАВЛЕНО: Умный поиск пути к файлу для Docker
base_dir = os.path.dirname(os.path.abspath(__file__)) # Определяем папку, где лежит этот скрипт
faq_path = os.path.join(base_dir, "faq.txt") # Соединяем путь с именем файла

loader = TextLoader(faq_path, encoding="utf-8") # Используем полный путь
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="techstore_faq_api"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

llm = ChatOpenAI(model="gpt-4o-mini")
template = """Ответь на вопрос кратко, используя контекст:
{context}

Вопрос: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 2. Настройка FastAPI ---
app = FastAPI(title="TechStore AI Support")

# Модель данных для запроса
class Question(BaseModel):
    text: str

@app.post("/chat")
def chat_endpoint(question: Question):
    """Принимает вопрос, ищет ответ в базе и возвращает текст."""
    response = rag_chain.invoke(question.text)
    return {"answer": response}

# --- 3. Запуск ---
if __name__ == "__main__":
    # Убедись, что host="0.0.0.0", чтобы Render видел сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)

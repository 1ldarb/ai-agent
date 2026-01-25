import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# 1. КОНФИГУРАЦИЯ
os.environ["OPENAI_API_KEY"] = "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"      # Вставьте ключ OpenAI
os.environ["PINECONE_API_KEY"] = "pcsk_5EBXVq_PqyyXF6MVsGrbrr3KqYf2uyQ42HFyEJXw7NqmfoEB4hydYN3ZbiAKY1cHYcBE3U"    # Вставьте ключ Pinecone

index_name = "ai-learning"

# 2. ДАННЫЕ (Факты, которые бот должен запомнить)
documents = [
    Document(page_content="Чтобы сбросить пароль, удерживайте кнопку RESET 5 секунд.", metadata={"source": "faq"}),
    Document(page_content="Доставка по Москве стоит 500 рублей, в регионы — 1200 рублей.", metadata={"source": "faq"}),
    Document(page_content="Оплата принимается картами Visa, MasterCard и МИР.", metadata={"source": "faq"}),
    Document(page_content="Гарантия на все товары составляет 2 года.", metadata={"source": "faq"}),
]

print("📡 Подключаюсь к Pinecone...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Размерность 1536

# 3. ЗАГРУЗКА В ОБЛАКО
# Эта команда создает базу и загружает в нее данные
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)
print("✅ Данные загружены! Ждем индексацию...")
time.sleep(2)

# 4. ПРОВЕРКА ПОИСКА
query = "Сколько стоит доставка в Питер?"
print(f"\n❓ Вопрос: {query}")

# Ищем 1 самый похожий документ
results = vectorstore.similarity_search(query, k=1)
print(f"📄 Найденный факт: {results[0].page_content}")

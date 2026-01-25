import chromadb
from sentence_transformers import SentenceTransformer
import asyncio
from openai import AsyncOpenAI

# --- НАСТРОЙКА ---
# Вставьте свой ключ OpenAI
client_openai = AsyncOpenAI(api_key="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")
# Используем ту же модель для русского языка, что и на 3-й неделе
model_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 1. Создаем базу данных на диске (папка ./chroma_db создастся сама)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Создаем или получаем коллекцию (таблицу)
collection = chroma_client.get_or_create_collection(name="faq_kb")

# --- ПОДГОТОВКА ДАННЫХ ---
docs = [
    "Чтобы сбросить пароль, удерживайте кнопку RESET 5 секунд.",
    "Доставка по Москве занимает 24 часа, по России - 3 дня.",
    "Мы принимаем к оплате карты Visa, MasterCard и Мир.",
    "Если устройство греется, выключите его и дайте остыть."
]
# Метаданные (теги для фильтрации)
metas = [
    {"category": "tech_support"},
    {"category": "delivery"},
    {"category": "sales"},
    {"category": "tech_support"}
]
ids = ["id1", "id2", "id3", "id4"]

# Загружаем данные только если база пустая (чтобы не дублировать)
if collection.count() == 0:
    print("📥 Индексирую базу данных...")
    embeddings = model_emb.encode(docs).tolist() # Chroma требует список, не numpy array
    collection.add(
        embeddings=embeddings,
        documents=docs,
        metadatas=metas,
        ids=ids
    )
else:
    print("✅ База уже существует. Пропускаю индексацию.")

async def ask_bot(question: str, category_filter: str = None):
    print(f"\n👤 Вопрос: '{question}' | Фильтр: {category_filter}")
    
    q_vec = model_emb.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=q_vec,
        n_results=1,
        where={"category": category_filter} if category_filter else None
    )
    
    # Проверяем, что список документов не пуст
    if not results['documents'] or not results['documents'][0]:
        print("❌ Ничего не найдено (возможно, фильтр слишком строгий).")
        return

    # ИСПРАВЛЕНИЕ 1 и 2: Достаем данные из вложенных списков [0][0]
    found_text = results['documents'][0][0]
    found_meta = results['metadatas'][0][0]
    category = found_meta['category']
    
    print(f"🤖 Найден факт [{category}]: {found_text}")

    # 3. Генерация ответа GPT
    prompt = f"Ответь на вопрос: '{question}', используя ТОЛЬКО этот факт: '{found_text}'"
    
    response = await client_openai.chat.completions.create(
        model="gpt-4o-mini", # Рекомендую mini для тестов (дешевле и быстрее)
        messages=[{"role": "user", "content": prompt}]
    )
    
    # ИСПРАВЛЕНИЕ 3: Не забываем про [0] перед .message
    print(f"💬 GPT: {response.choices[0].message.content}")

# --- ЗАПУСК ---
async def main():
    # Тест 1: Вопрос про доставку (без фильтра)
    await ask_bot("Как долго ждать заказ?")
    
    # Тест 2: Технический вопрос, но мы ОШИБОЧНО ищем в категории 'sales' (продажи)
    # Это покажет, как работает фильтр: он должен отсечь правильный ответ, так как категория не та.
    await ask_bot("Что делать, если перегрелся?", category_filter="sales")
    
    # Тест 3: Тот же вопрос, но с правильным фильтром 'tech_support'
    await ask_bot("Что делать, если перегрелся?", category_filter="tech_support")

if __name__ == "__main__":
    asyncio.run(main())

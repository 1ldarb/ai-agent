import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI

# 1. Настройка (Вставьте свой API ключ!)
client = AsyncOpenAI(api_key="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")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Наша "База знаний" (факты о вымышленном магазине)
knowledge_base = [
    "Доставка по городу стоит 500 рублей, за город - 1000 рублей.",
    "Магазин работает ежедневно с 9:00 до 21:00 без выходных.",
    "Мы продаем ноутбуки, смартфоны и аксессуары к ним.",
    "Возврат товара возможен в течение 14 дней при наличии чека.",
    "Техническая поддержка доступна только в Telegram."
]

# Предварительно считаем векторы для базы (чтобы не делать это каждый раз)
print("⏳ Индексирую базу знаний...")
kb_embeddings = embedder.encode(knowledge_base)

async def answer_user(question: str):
    print(f"\n👤 Вопрос: {question}")
    
    question_vec = embedder.encode([question])
    scores = cosine_similarity(question_vec, kb_embeddings)
    
    best_idx = scores.argmax()
    best_text = knowledge_base[best_idx]
    
    # ИСПРАВЛЕНИЕ 1: Добавляем [0], чтобы достать число из первой строки
    score_value = scores[0][best_idx]
    print(f"🤖 Найденный факт: {best_text} (Сходство: {score_value:.4f})")
    
    prompt = f"""
    Ты помощник поддержки. Ответь на вопрос клиента, используя ТОЛЬКО эту информацию:
    "{best_text}"
    
    Вопрос: {question}
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # ИСПРАВЛЕНИЕ 2: Добавляем [0] перед .message
    print(f"💬 Ответ GPT: {response.choices[0].message.content}")

async def main():
    # Зададим пару вопросов
    await answer_user("Сколько стоит привезти заказ домой?")
    await answer_user("Куда писать, если сломался телефон?")

if __name__ == "__main__":
    asyncio.run(main())

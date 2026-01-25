import asyncio
from openai import AsyncOpenAI

# Вставь свой ключ
client = AsyncOpenAI(api_key="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")

async def ask_gpt(question):
    print(f"⏳ Отправляю вопрос: {question}...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}]
        )
        # Обязательно [0], помнишь?
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка: {e}"

async def main():
    tasks = [
        ask_gpt("Придумай имя для кота-хакера"),
        ask_gpt("Придумай кличку для собаки-космонавта"),
        ask_gpt("Придумай прозвище для попугая-пирата")
    ]

    print(f"🚀 Запускаю {len(tasks)} задач одновременно...")
    results = await asyncio.gather(*tasks)

    print("\n🏁 Все готово:")
    for result in results:
        print(f"- {result}")

if __name__ == "__main__":
    asyncio.run(main())

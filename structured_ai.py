import asyncio
import os
from openai import AsyncOpenAI
from pydantic import BaseModel

# Вставьте ваш ключ
client = AsyncOpenAI(api_key="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")

# 1. Создаем "форму" для ответа
class PetName(BaseModel):
    name: str
    reason: str

async def generate_name(animal: str):
    print(f"⏳ Думаю над именем для: {animal}...")
    
    # Используем новую модель, которая поддерживает Structured Outputs
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06", 
        messages=[
            {"role": "system", "content": "Ты креативный помощник."},
            {"role": "user", "content": f"Придумай имя для {animal}."}
        ],
        response_format=PetName, # <--- Требуем строгий формат
    )
    
    return completion.choices[0].message.parsed

async def main():
    # Генерируем параллельно
    tasks = [
        generate_name("кота-хакера"),
        generate_name("собаки-космонавта"),
        generate_name("попугая-пирата")
    ]
    
    print("🚀 Запускаю генерацию...")
    results = await asyncio.gather(*tasks)
    
    print("\n🏁 Итог (чистые данные):")
    for item in results:
        # Обращаемся к данным как к объектам, а не словарю!
        print(f"Имя: {item.name} | Почему: {item.reason}")

if __name__ == "__main__":
    asyncio.run(main())

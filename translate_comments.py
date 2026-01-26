import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def translate_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    print(f"🔄 Обрабатываю: {filepath}...")

    prompt = (
        "Ты — опытный Python-разработчик. "
        "Твоя задача: перевести ВСЕ комментарии и docstrings в предоставленном коде на АНГЛИЙСКИЙ язык. "
        "Сам код (логику, имена переменных) НЕ МЕНЯЙ. "
        "Верни только готовый код Python."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": code}
        ]
    )

    new_code = response.choices[0].message.content
    
    # Очистка от маркдауна, если модель вернула ```python ... ```
    if new_code.startswith("```python"):
        new_code = new_code.replace("```python", "").replace("```", "").strip()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_code)
    
    print(f"✅ Готово: {filepath}")

# Список файлов, которые нужно перевести
files_to_translate = ["server.py", "support_bot.py", "researcher.py"]

for file in files_to_translate:
    if os.path.exists(file):
        translate_file(file)
    else:
        print(f"⚠️ Файл {file} не найден.")


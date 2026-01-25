from openai import OpenAI

# Вставьте ваш ключ сюда
client = OpenAI(api_key="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")

# 1. Спрашиваем у пользователя вопрос через терминал
user_question = input("Что ты хочешь узнать, юнга? Введи вопрос: ")

print("Отправляю запрос пирату...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Ты — старый ворчливый пират. Отвечай на сленге."},
        # 2. Вставляем переменную user_question внутрь сообщения
        {"role": "user", "content": user_question}
    ]
)

print("Ответ AI:")
print(response.choices[0].message.content)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Вставьте ваш API ключ
api_key = "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"

# 1. Модель
model = ChatOpenAI(api_key=api_key, model="gpt-4o")

# 2. Промпт для суммаризации
# Мы просим модель действовать как аналитик
prompt = ChatPromptTemplate.from_template(
    """Ты профессиональный аналитик. 
    Прочитай следующий текст и выпиши 3 ключевые мысли в виде списка пультов (bullet points).
    
    Текст:
    {text}
    """
)

# 3. Собираем цепочку
chain = prompt | model | StrOutputParser()

# 4. Входные данные (длинный текст про Python)
article_text = """
Python — это высокоуровневый язык программирования общего назначения с динамической строгой типизацией 
и автоматическим управлением памятью. Структура языка и подход к объектно-ориентированному программированию 
позволяют писать четкий и логичный код для проектов любого масштаба. 
Одной из главных особенностей Python является его интерпретируемость: код запускается построчно, 
что упрощает отладку, но может снижать скорость выполнения по сравнению с компилируемыми языками (C++, Java).
Огромное сообщество разработчиков создало тысячи библиотек для всего: от веб-разработки (Django) 
до искусственного интеллекта (PyTorch, TensorFlow).
"""

print("⏳ Анализирую текст...")
result = chain.invoke({"text": article_text})
print(f"\n📝 Краткая выжимка:\n{result}")

import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# ─── НАСТРОЙКИ ───
API_KEY = "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"

# 1. Модель
llm = ChatOpenAI(api_key=API_KEY, temperature=0, model="gpt-4o")

# 2. Инструменты (Математика + Поиск)
search = DuckDuckGoSearchRun()
math_tool = load_tools(["llm-math"], llm=llm)[0]
tools = [search, math_tool]

# 3. Создаём агента
app = create_react_agent(llm, tools)

# 4. ЗАПУСК
print("\n🧠 Агент-исследователь запущен!")

# Пример сложного запроса:
question = "Найди текущую цену Биткоина в долларах и посчитай, сколько будет стоить 0.5 BTC с учетом комиссии 1.5%."

input_data = {
    "messages": [
        SystemMessage(content="Ты профессиональный аналитик. Сначала найди данные в поиске, а затем используй калькулятор для расчетов."),
        HumanMessage(content=question)
    ]
}

try:
    result = app.invoke(input_data)
    final_answer = result["messages"][-1].content
    print(f"\n✅ Ответ: {final_answer}")
except Exception as e:
    print(f"\n❌ Ошибка: {e}")

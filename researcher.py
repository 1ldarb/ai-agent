import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from serpapi import GoogleSearch

load_dotenv()

# --- Инструмент 1: Поиск ---
@tool
def search_google(query: str):
    """Ищет список ссылок и новостей в Google. Возвращает сниппеты и ссылки."""
    print(f"🕵️  Ищу в Google: {query}")
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    organic_results = results.get("organic_results", [])
    output = []
    # Берем топ-3 результата с ссылками
    for r in organic_results[:3]:
        title = r.get("title", "Без заголовка")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        output.append(f"Title: {title}\nLink: {link}\nSummary: {snippet}\n")
    
    return "\n---\n".join(output) if output else "Ничего не найдено."

# --- Инструмент 2: Чтение сайта (НОВОЕ) ---
@tool
def scrape_webpage(url: str):
    """Читает полный текст веб-страницы по ссылке. Используй это, чтобы узнать детали."""
    print(f"📖 Читаю статью: {url}")
    try:
        # Притворяемся обычным браузером
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Удаляем скрипты и стили, оставляем только текст
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        # Ограничиваем длину (чтобы не перегрузить модель), берем первые 8000 символов
        return "Текст статьи:\n" + " ".join(text.split())[:8000]
    except Exception as e:
        return f"Ошибка при чтении: {e}"

# --- Граф Агента ---
llm = ChatOpenAI(model="gpt-4o-mini")

# ТЕПЕРЬ У АГЕНТА ДВА ИНСТРУМЕНТА:
tools = [search_google, scrape_webpage] 

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- Запуск ---
if __name__ == "__main__":
    # Задаем тему исследования
    query = "Собери информацию про GPT-5: предполагаемая дата выхода, новые функции и слухи. Сделай структурированный отчет."
    print(f"🚀 Начинаю исследование: {query}\n")
    
    # Запускаем агента
    final_state = app.invoke({"messages": [HumanMessage(content=query)]})
    report = final_state["messages"][-1].content
    
    print("\n🤖 Результат:")
    print(report)
    
    # Сохраняем в файл
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# Отчет исследования\n\n**Тема:** {query}\n\n---\n\n{report}")
        
    print("\n📄 Отчет успешно сохранен в файл 'report.md'!")

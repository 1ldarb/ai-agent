import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

def google_search(query):
    print(f"🔎 Ищу: {query}...")
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Берем первый результат
    if "organic_results" in results:
        return results["organic_results"][0]["snippet"]
    return "Ничего не найдено."

# Проверка
print(google_search("Текущий курс биткоина"))

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

# --- Tool 1: Search ---
@tool
def search_google(query: str):
    """Searches for a list of links and news on Google. Returns snippets and links."""
    print(f"🕵️  Searching in Google: {query}")
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    organic_results = results.get("organic_results", [])
    output = []
    # Take the top-3 results with links
    for r in organic_results[:3]:
        title = r.get("title", "No title")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        output.append(f"Title: {title}\nLink: {link}\nSummary: {snippet}\n")
    
    return "\n---\n".join(output) if output else "Nothing found."

# --- Tool 2: Webpage Reading (NEW) ---
@tool
def scrape_webpage(url: str):
    """Reads the full text of a webpage from the link. Use this to get details."""
    print(f"📖 Reading article: {url}")
    try:
        # Pretending to be a regular browser
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Removing scripts and styles, keeping only text
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        # Limiting the length (to not overload the model), taking the first 8000 characters
        return "Article text:\n" + " ".join(text.split())[:8000]
    except Exception as e:
        return f"Error while reading: {e}"

# --- Agent Graph ---
llm = ChatOpenAI(model="gpt-4o-mini")

# NOW THE AGENT HAS TWO TOOLS:
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

# --- Execution ---
if __name__ == "__main__":
    # Setting the research topic
    query = "Gather information about GPT-5: expected release date, new features and rumors. Make a structured report."
    print(f"🚀 Starting research: {query}\n")
    
    # Running the agent
    final_state = app.invoke({"messages": [HumanMessage(content=query)]})
    report = final_state["messages"][-1].content
    
    print("\n🤖 Result:")
    print(report)
    
    # Saving to file
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n**Topic:** {query}\n\n---\n\n{report}")
        
    print("\n📄 Report successfully saved to 'report.md'!")
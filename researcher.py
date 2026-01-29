import os
import time
import requests
from bs4 import BeautifulSoup
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from serpapi import GoogleSearch

# Load environment variables (OPENAI_API_KEY, SERPAPI_API_KEY)
load_dotenv()

# --- Tools Definition ---

@tool
def search_google(query: str):
    """Searches Google for real-time information and news using SerpApi."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    output = []
    # Limit to top 3 results to save tokens
    for r in results[:3]:
        output.append(f"Title: {r.get('title')}\nLink: {r.get('link')}\nSnippet: {r.get('snippet')}\n")
    return "\n---\n".join(output) if output else "No results found."

@tool
def scrape_webpage(url: str):
    """Extracts text content from a URL for deep analysis. 
    Reduced character limit to prevent RateLimitErrors (429)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove non-text elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # FIX: Reduced limit from 8000 to 4000 to stay within Token Per Minute (TPM) limits
        content = " ".join(soup.get_text().split())
        return "Content snippet:\n" + content[:4000] 
    except Exception as e:
        return f"Scraping error: {e}"

# --- Agent Nodes ---

# Initialize LLM (gpt-4o-mini is cost-effective but has strict rate limits)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_google, scrape_webpage]

def researcher_node(state: MessagesState):
    """Handles information gathering and report drafting."""
    # FIX: Add a small delay to avoid hitting OpenAI Rate Limits
    time.sleep(1) 
    
    system_prompt = SystemMessage(content=(
        "You are an Expert Researcher. Gather information using tools and write a structured report. "
        "If the Reviewer provides feedback, update your report to meet the requirements."
    ))
    messages = [system_prompt] + state["messages"]
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def reviewer_node(state: MessagesState):
    """Reviews the report and ensures quality and accuracy."""
    # FIX: Add a delay to let the Token Per Minute (TPM) quota reset
    time.sleep(1) 
    
    # Calculate conversation depth
    turns = len(state["messages"])
    
    # Adaptive feedback based on turn count to prevent infinite loops
    feedback_instruction = (
        "Focus only on major issues. If the report is generally good, approve it." 
        if turns > 6 else 
        "Be very strict. Check for depth, citations, and clarity."
    )

    system_prompt = SystemMessage(content=(
        "You are a Senior Content Editor. "
        f"{feedback_instruction} "
        "To end the process, you MUST start your message with the word 'FINAL_APPROVED'. "
        "Otherwise, provide specific feedback for the Researcher."
    ))
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- Graph Routing Logic ---

def routing_logic(state: MessagesState) -> Literal["tools", "reviewer", END]:
    """Determines the next step in the multi-agent workflow."""
    last_msg = state["messages"][-1]
    
    # If the LLM called a tool, go to the tools node
    if last_msg.tool_calls:
        return "tools"
    
    # If the Reviewer gave final approval, terminate the process
    if "FINAL_APPROVED" in last_msg.content:
        return END
        
    # Otherwise, send it back for review/revision
    return "reviewer"

# --- Build the LangGraph Workflow ---

workflow = StateGraph(MessagesState)

# Define a retry policy to automatically handle 429 errors from OpenAI
retry_policy = {"max_attempts": 3}

# Add nodes with the defined retry policy
workflow.add_node("researcher", researcher_node, retry_policy=retry_policy)
workflow.add_node("reviewer", reviewer_node, retry_policy=retry_policy)
workflow.add_node("tools", ToolNode(tools), retry_policy=retry_policy)

# Set up edges and conditional routing
workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", routing_logic)
workflow.add_edge("tools", "researcher")
workflow.add_edge("reviewer", "researcher")

# Compile the graph with memory support for session persistence
app = workflow.compile(checkpointer=MemorySaver())

# --- Terminal Execution Block ---
# This block only runs if you call 'python researcher.py' directly,
# but it won't interfere when imported into app.py.

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "terminal_test_1"}}
    print("🔍 AI Researcher (Terminal Mode) is ready.")
    
    while True:
        user_query = input("\nEnter research topic (or 'exit'): ")
        if user_query.lower() in ["exit", "quit", "q"]:
            break
            
        inputs = {"messages": [HumanMessage(content=user_query)]}
        print("\n🚀 Starting research workflow...")
        
        # Stream events to see real-time progress in the terminal
        for event in app.stream(inputs, config=config):
            for node, values in event.items():
                print(f"\n--- [ NODE: {node.upper()} ] ---")
                print(values["messages"][-1].content[:500] + "...") # Show snippet of the step

import streamlit as st
import requests
import os
from langchain_core.messages import HumanMessage
# Importing the graph from your researcher.py file
from researcher import app as researcher_app

# 1. Page Configuration 
st.set_page_config(page_title="Ildar AI Hub", page_icon="🕵️‍♂️", layout="wide")

# --- SIDEBAR: System Architecture & Navigation ---
with st.sidebar:
    st.title("🛠️ System Architecture")
    
    # Selection for switching between different AI tools
    mode = st.radio(
        "Select Active Agent:",
        ["Support Assistant", "Global Researcher"],
        help="Choose the tool you want to use for this session."
    )
    
    st.info("This system uses RAG and Multi-Agent workflows to provide verified data.")
    
    # Tech Stack with green highlights as requested
    st.subheader("Tech Stack:")
    st.markdown("""
    * :green[LangGraph] (Orchestration)
    * :green[GPT-4o-mini] (Reasoning Engine)
    * :green[Pinecone] (Vector Database)
    * :green[SerpApi] (Real-time Search)
    * :green[Streamlit] (Interactive Frontend)
    """)
    
    st.divider()
    
    # Reset button to clear all chat histories
    if st.button("Reset Research Session", use_container_width=True):
        if "support_messages" in st.session_state:
            st.session_state.support_messages = []
        if "research_messages" in st.session_state:
            st.session_state.research_messages = []
        st.rerun()

# --- MAIN INTERFACE LOGIC ---

if mode == "Support Assistant":
    # --- Mode: TechStore Support Assistant ---
    st.title("🤖 TechStore Support")
    st.caption("Autonomous assistant powered by Pinecone RAG and local knowledge base (faq.txt).")
    
    # Initialize session state for Support chat
    if "support_messages" not in st.session_state:
        st.session_state.support_messages = []

    # Display Support message history
    for msg in st.session_state.support_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about technical issues, products, or returns..."):
        # Add user message to history
        st.session_state.support_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing knowledge base..."):
            try:
                # API Call to your local FastAPI server (server.py)
                res = requests.post("http://127.0.0.1:8000/chat", json={"text": prompt}, timeout=15)
                answer = res.json().get("answer", "Error: No answer field in API response.")
            except Exception as e:
                # Error handling if the FastAPI server is not running
                answer = "❌ **Connection Error**: Please ensure `server.py` is running in your terminal."
        
        # Add assistant response to history
        st.session_state.support_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    # --- Mode: Global AI Researcher ---
    st.title("🕵️‍♂️ Global AI Researcher")
    st.caption("Multi-agent cyclic graph system for deep web research, verification, and synthesis.")

    # Initialize session state for Research chat
    if "research_messages" not in st.session_state:
        st.session_state.research_messages = []

    # Display Research message history
    for msg in st.session_state.research_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("What would you like me to research today?"):
        # Add user prompt to history
        st.session_state.research_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Use st.status to prevent the UI from "freezing" during long agent loops
            with st.status("🚀 Initializing Multi-Agent Workflow...", expanded=True) as status:
                full_response = ""
                # Создаем контейнер для отчета, который будет обновляться в реальном времени
                report_placeholder = st.empty()
                
                try:
                    # Изменен ID треда для чистого запуска сессии
                    config = {"configurable": {"thread_id": "st_research_v3"}}
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    
                    # .stream() позволяет нам отслеживать каждый шаг графа
                    for event in researcher_app.stream(inputs, config=config):
                        for node, values in event.items():
                            # Обновляем текст статуса текущим узлом
                            status.update(label=f"🛠️ Active Node: **{node.upper()}**", state="running")
                            
                            if "messages" in values:
                                content = values["messages"][-1].content
                                # Если узел RESEARCHER выдал длинный текст (отчет)
                                if node == "researcher" and len(content) > 100:
                                    full_response = content
                                    # Мгновенно выводим текст в браузер
                                    report_placeholder.markdown(full_response)
                    
                    # Финализируем плашку статуса
                    status.update(label="✅ Research Completed Successfully!", state="complete", expanded=False)
                    
                    # Если по какой-то причине текст не отобразился в цикле, выводим его здесь
                    if full_response:
                        report_placeholder.markdown(full_response)
                    else:
                        st.warning("Could not capture the report content.")
                
                except Exception as e:
                    status.update(label="❌ Workflow Error", state="error")
                    st.error(f"An internal error occurred: {str(e)}")
        
        # Сохраняем финальный отчет в историю чата
        if full_response:
            st.session_state.research_messages.append({"role": "assistant", "content": full_response})

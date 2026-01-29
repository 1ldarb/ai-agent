import streamlit as st
from langchain_core.messages import HumanMessage
from researcher import app

# Page configuration for a professional portfolio appearance
st.set_page_config(page_title="AI Research Agent Pro", page_icon="🤖", layout="wide")

# --- Sidebar: System Architecture & Tech Stack ---
with st.sidebar:
    st.title("🛠️ System Architecture")
    st.info("This agent uses a multi-agent cyclic graph to perform high-quality, verified web research.")
    
    st.markdown("""
    **Core Components:**
    - **Researcher Agent**: Gathers data & drafts structured reports.
    - **Reviewer Agent**: Performs QA, verification, and feedback loops.
    - **Tools**: Real-time Google Search & Deep Web Scraping.
    
    **Tech Stack:**
    - `LangGraph` (Stateful Orchestration)
    - `GPT-4o-mini` (Reasoning Engine)
    - `SerpApi` (Real-time Search Integration)
    - `Streamlit` (Interactive Frontend)
    """)
    
    st.divider()
    
    if st.button("Reset Research Session"):
        st.session_state.messages = []
        st.rerun()

# --- Main UI Header ---
st.title("🕵️ Global AI Researcher")
st.caption("Autonomous multi-agent system for deep web research, verification, and synthesis.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages from the session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Field
if prompt := st.chat_input("What would you like me to research today?"):
    # Store and display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Real-time Thought Process Visualization using st.status
        status_log = st.status("🚀 Initializing research agents...", expanded=True)
        
        # Configuration for safety and persistent session memory
        config = {
            "configurable": {"thread_id": "portfolio_research_session"},
            "recursion_limit": 15  # Safety net to prevent infinite critique loops
        }
        
        final_report = ""
        step_count = 0

        try:
            # Streaming the graph execution loop to capture node activity
            for event in app.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                step_count += 1
                for node, data in event.items():
                    # Update status UI based on the active agent node
                    if node == "researcher":
                        status_log.write(f"Step {step_count}: 🕵️ **Researcher** is analyzing data and drafting...")
                    elif node == "tools":
                        status_log.write(f"Step {step_count}: 🔍 **Tools** are fetching live web information...")
                    elif node == "reviewer":
                        status_log.write(f"Step {step_count}: 🧐 **Reviewer** is validating the report quality...")
                    
                    # Store the latest message as the potential final report
                    if "messages" in data:
                        final_report = data["messages"][-1].content
            
            status_log.update(label="✅ Research Completed!", state="complete", expanded=False)

        except Exception as e:
            # Graceful handling of the GraphRecursionError
            if "Recursion limit" in str(e):
                status_log.update(label="⚠️ Complexity limit reached", state="error", expanded=False)
                st.warning("The research was highly complex. Below is the most complete draft generated before the safety limit.")
            else:
                status_log.update(label="❌ System Error", state="error", expanded=False)
                st.error(f"An unexpected error occurred: {e}")

        # Final Report Output Logic
        if final_report:
            # Clean up approval markers for a professional UI look
            clean_report = final_report.replace("FINAL_APPROVED", "").strip()
            st.markdown(clean_report)
            
            # Save the final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": clean_report})
            
            # Professional Download feature for the final report
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=clean_report,
                file_name="ai_research_report.md",
                mime="text/markdown"
            )

🕵️‍♂️ Ildar AI Hub: Advanced Multi-Agent Researcher & RAG System
A professional AI ecosystem designed to demonstrate high-level engineering skills in Agentic Workflows and Retrieval-Augmented Generation (RAG). This project serves as a cornerstone of my transition into AI Development, showcasing a robust backend built with FastAPI and an interactive Streamlit frontend.

🚀 Core Agents
1. Global AI Researcher (LangGraph)

Architecture: A cyclic multi-agent graph that automates deep-web research.

Workflow: The Researcher agent gathers data via SerpApi, which is then cross-checked by a Reviewer agent to ensure factual accuracy and high synthesis quality.

Stability: Implements custom rate-limiting and retry logic to handle high-token loads and API limits (Error 429) gracefully.

2. Support Assistant (RAG + Pinecone)

Technology: A specialized support bot using LangChain and Pinecone vector database.

Context: Strictly follows a provided knowledge base (faq.txt) focused on the US market to provide hallucination-free responses regarding shipping, taxes, and local policies.

Semantic Search: Uses OpenAI Embeddings (text-embedding-3-small) to understand user intent beyond simple keywords.

🛠️ Tech Stack
Category	Technology
Orchestration	LangChain & LangGraph
Vector Database	Pinecone (Cloud-native)
LLM	OpenAI GPT-4o-mini
Backend	FastAPI & Uvicorn
Frontend	Streamlit
External APIs	SerpApi (Real-time Search)
🔌 API & Integration
This project is "API-ready," allowing easy integration with Telegram bots or external web applications.

Interactive API Documentation (Swagger UI) is available at: http://127.0.0.1:8000/docs

Example Request:

JSON
POST /chat
{
  "text": "Do you deliver to New York and accept Venmo?"
}
Example Response:

JSON
{
  "answer": "Yes, we ship to all major hubs including New York City. We also support direct payments via Venmo."
}
⚙️ Installation & Setup
Clone & Environment:

Bash
git clone https://github.com/ildar-dev/ildar-ai-hub.git
cd ildar-ai-hub
Configure Keys: Create a .env file with your OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME.

Ingest Data: Run the ingestion script to sync your faq.txt with the cloud:

Bash
python ingest.py
Launch Hub:

Bash
streamlit run app.py
Developed by Ildar — AI Engineer based in Israel, specializing in Agentic Workflows and Scalable RAG Architectures.

# AI Support Bot (RAG) 🤖

A smart technical support bot built with **FastAPI**, **LangChain**, and **ChromaDB**.

It uses **RAG (Retrieval-Augmented Generation)** to answer customer questions based strictly on a provided knowledge base (`faq.txt`), ensuring accurate and hallucination-free responses using OpenAI's models [4, 5].

## 🚀 Features
- **RAG Architecture:** Retrieves precise context from documents before answering.
- **Vector Search:** Uses **ChromaDB** [1] and OpenAI Embeddings for semantic search (understands meaning, not just keywords).
- **API Ready:** Built on **FastAPI** for easy integration with frontend apps or Telegram bots.
- **Anti-Hallucination:** Strictly follows the provided documentation logic.

## 🛠️ Tech Stack
- **Python 3.10+**
- **LangChain** (Orchestration) [3]
- **ChromaDB** (Vector Database) [2]
- **OpenAI GPT-4o** (LLM) [5]
- **FastAPI & Uvicorn** (Web Server)

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-support-bot.git
   cd ai-support-bot
2. Install dependencies:
3. Configure Environment: Create a .env file in the root directory and add your API key:
4. Add Knowledge Base: Ensure faq.txt is in the root directory with your support content.
5. Run the Server:
6. The server will start at http://0.0.0.0:8000.
🔌 API Usage
Endpoint: POST /chat
Example Request:
{
  "text": "What is your return policy?"
}
Example Response:
{
  "answer": "You can return items within 14 days if the packaging and receipt are preserved."
}
📄 Documentation
Interactive API docs (Swagger UI) are available automatically at:
http://127.0.0.1:8000/docs

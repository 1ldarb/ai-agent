import uvicorn
import os  # ADDED: for working with paths
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Importing the "brains" (LangChain components)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Preparing RAG (Copying logic from smart_bot.py) ---
print("⚙️  Loading server and knowledge base...")

# ADDED: Smart search for file path for Docker
base_dir = os.path.dirname(os.path.abspath(__file__))  # Determine the folder where this script is located
faq_path = os.path.join(base_dir, "faq.txt")  # Combine path with file name

loader = TextLoader(faq_path, encoding="utf-8")  # Use full path
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="techstore_faq_api"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

llm = ChatOpenAI(model="gpt-4o-mini")
template = """Respond to the question briefly, using the context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 2. Setting up FastAPI ---
app = FastAPI(title="TechStore AI Support")

# Data model for the request
class Question(BaseModel):
    text: str

@app.post("/chat")
def chat_endpoint(question: Question):
    """Accepts a question, searches for an answer in the database, and returns text."""
    response = rag_chain.invoke(question.text)
    return {"answer": response}

# --- 3. Running ---
if __name__ == "__main__":
    # Make sure to set host="0.0.0.0" so Render can see the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def run_ingestion():
    print("🛠️  Loading faq.txt...")
    loader = TextLoader("faq.txt")
    document = loader.load()

    # Splitting text for RAG
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    print(f"✅ Created {len(chunks)} text chunks.")

    # Initializing Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Pulling Index Name from your .env
    index_name = os.getenv("PINECONE_INDEX_NAME")

    print(f"📡 Uploading to Pinecone index: {index_name}...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    print("🚀 Done! Your US Knowledge Base is ready.")

if __name__ == "__main__":
    run_ingestion()

import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 1. LOAD CONFIGURATION
# Now we use the .env file instead of hardcoding keys
load_dotenv()

# 2. INITIALIZE EMBEDDINGS
# Using the same model as in your other scripts for consistency
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = os.getenv("PINECONE_INDEX_NAME")

def upload_us_knowledge_base():
    print(f"📡 Connecting to Pinecone index: {index_name}...")
    
    # 3. LOAD DATA FROM FILE
    # Instead of hardcoded text, we load your new US faq.txt
    if not os.path.exists("faq.txt"):
        print("❌ Error: faq.txt not found!")
        return

    loader = TextLoader("faq.txt")
    documents = loader.load()
    
    # Split text into chunks to help the AI find specific answers
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"📦 Prepared {len(docs)} chunks of data for upload.")

    # 4. UPLOAD TO CLOUD
    # This will update your existing Pinecone index with US data
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("✅ Success! Your US Knowledge Base is now active.")
    time.sleep(2)

    # 5. TEST SEARCH
    query = "Do you deliver to New York City?"
    print(f"\n❓ Testing query: {query}")
    results = vectorstore.similarity_search(query, k=1)
    
    if results:
        print(f"📄 Found Fact: {results[0].page_content[:200]}...")
    else:
        print("⚠️ No relevant facts found in Pinecone.")

if __name__ == "__main__":
    upload_us_knowledge_base()

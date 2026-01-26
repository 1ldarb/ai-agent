import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Load the knowledge base ---
try:
    with open("faq.txt", "r", encoding="utf-8") as f:
        faq_data = f.read()
except FileNotFoundError:
    print("Error: The file faq.txt was not found!")
    exit()

# --- 2. Configure the Model and Prompt ---
llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = """You are a polite support staff member of the electronics store 'TechStore'.
Your task is to answer customer questions using ONLY the knowledge base provided below.

Knowledge base:
{context}

IMPORTANT:
1. If the answer is not in the database, respond: "Unfortunately, I don't have that information. Please contact the manager by phone."
2. Do not invent conditions that are not in the text.
3. Be brief and friendly.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])

# Create the chain: Prompt -> Model -> Text
chain = prompt | llm | StrOutputParser()

# --- 3. Start the chat ---
if __name__ == "__main__":
    print("🤖 TechStore support bot is ready! (Type 'exit' to quit)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("Bot: Goodbye!")
            break
            
        # Run the chain, passing the knowledge base and question
        response = chain.invoke({"context": faq_data, "question": user_input})
        print(f"Bot: {response}\n")
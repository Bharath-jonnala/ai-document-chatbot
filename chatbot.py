import os
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Connect to Groq AI
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt Engineering — this controls AI behavior
system_prompt = """
You are a helpful assistant. 
Answer questions clearly and concisely.
If you don't know something, 
say 'I don't know' honestly.
"""

print("=" * 40)
print("   AI Chatbot - Powered by LLaMA3")
print("=" * 40)
print("Type your question. Type 'quit' to exit")
print("=" * 40)

# Chat loop
while True:
    # Get user input
    user_input = input("\nYou: ")
    
    # Exit condition
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    # Send to AI and get response    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]
    )
    
    # Print AI response
    ai_response = response.choices[0].message.content
    print(f"\nAI: {ai_response}")
    print("-" * 40)
import os
from dotenv import load_dotenv

# LangChain imports — updated for latest version
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

print("=" * 50)
print("   RAG Chatbot - Answers from Document")
print("=" * 50)

# ============================================
# STEP 1 - Load Document
# ============================================
print("\n[1/4] Loading document...")
loader = TextLoader("company_data.txt")
documents = loader.load()
print(f"Document loaded!")

# ============================================
# STEP 2 - Split into Chunks
# ============================================
print("\n[2/4] Splitting into chunks...")
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks!")

# ============================================
# STEP 3 - Create Embeddings + Vector Store
# ============================================
print("\n[3/4] Creating embeddings...")
print("(First time takes 1-2 minutes)")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store ready!")

# ============================================
# STEP 4 - Create RAG Chain (Modern LCEL style)
# ============================================
print("\n[4/4] Building RAG chain...")

# Connect LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Prompt Template
prompt_template = """
You are a helpful company assistant.
Answer using ONLY the context below.
If answer not in context say:
"I don't have that information."

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Helper function to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\nRAG Chatbot Ready!")
print("Ask questions about company document")
print("Type 'quit' to exit")
print("=" * 50)

# ============================================
# CHAT LOOP
# ============================================
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    print("\nSearching document...")
    response = rag_chain.invoke(user_input)
    print(f"\nAI: {response}")
    print("-" * 50)
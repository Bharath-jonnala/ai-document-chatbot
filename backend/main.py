import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "*"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global Storage
# ============================================
sessions = {}

# ============================================
# Lazy Loaded Models (IMPORTANT FIX)
# ============================================
embeddings = None
llm = None

def get_models():
    global embeddings, llm

    if embeddings is None:
        print("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    if llm is None:
        print("Loading LLM...")
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in environment variables")

        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )

    return embeddings, llm

# ============================================
# Build RAG Chain
# ============================================
def build_rag_chain(documents):
    embeddings, llm = get_models()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = """
You are a helpful document assistant.
Answer using ONLY the context below.
If answer not in context say exactly:
"I don't have that information in the uploaded document."
Keep answer clear and concise.

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ============================================
# API Models
# ============================================
class QuestionRequest(BaseModel):
    session_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

# ============================================
# Startup Check
# ============================================
@app.on_event("startup")
def startup_event():
    print("✅ FastAPI app started successfully")

# ============================================
# Routes
# ============================================
@app.get("/")
def home():
    return {"message": "Document RAG API running!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(tmp_path)
        elif suffix.lower() == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()

        rag_chain = build_rag_chain(documents)

        sessions[session_id] = rag_chain

        os.unlink(tmp_path)

        return SessionResponse(
            session_id=session_id,
            message=f"File '{file.filename}' uploaded successfully!"
        )

    except Exception as e:
        os.unlink(tmp_path)
        return {"error": str(e)}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if request.session_id not in sessions:
        return AnswerResponse(
            answer="Session not found. Please upload a document first."
        )

    rag_chain = sessions[request.session_id]

    response = rag_chain.invoke(request.question)

    return AnswerResponse(answer=response)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]

    return {"message": "Session deleted"}
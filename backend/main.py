import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ IMPORTANT: lightweight embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

# ============================================
# Lazy Models (LIGHTWEIGHT)
# ============================================
embeddings = None
llm = None

def get_models():
    global embeddings, llm

    if embeddings is None:
        print("Loading lightweight embeddings...")
        embeddings = FastEmbedEmbeddings()

    if llm is None:
        print("Loading LLM...")
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY missing")

        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )

    return embeddings, llm

# ============================================
# RAG Chain
# ============================================
def build_rag_chain(documents):
    embeddings, llm = get_models()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # reduced memory

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from context.

Context: {context}
Question: {question}
Answer:
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# ============================================
# API Models
# ============================================
class QuestionRequest(BaseModel):
    session_id: str
    question: str

# ============================================
# Routes
# ============================================
@app.get("/")
def home():
    return {"message": "RAG API running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        if suffix.lower() == ".docx":
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path)

        docs = loader.load()

        sessions[session_id] = build_rag_chain(docs)

        os.unlink(path)

        return {"session_id": session_id}

    except Exception as e:
        os.unlink(path)
        return {"error": str(e)}

@app.post("/ask")
def ask(req: QuestionRequest):
    if req.session_id not in sessions:
        return {"answer": "Upload document first"}

    chain = sessions[req.session_id]
    return {"answer": chain.invoke(req.question)}

@app.delete("/session/{session_id}")
def delete(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "deleted"}
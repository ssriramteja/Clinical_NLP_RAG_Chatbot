import logging
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

# Ensure src/ is in the module path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_chain import ClinicalRAGChain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rag_chain: Optional[ClinicalRAGChain] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes the RAG chain during startup and retains it in memory.
    """
    global rag_chain
    logger.info("Initializing Clinical RAG API service...")
    try:
        rag_chain = ClinicalRAGChain(k=4)
        logger.info("RAG chain successfully loaded and ready for inference.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}", exc_info=True)
        raise
    
    yield
    logger.info("Service shutting down...")

app = FastAPI(
    title="Clinical RAG Service API",
    description="Microservice providing Retrieval-Augmented Generation over clinical records.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    verbose: bool = Field(default=False)

class ChatMessage(BaseModel):
    role: str = Field(...)
    content: str = Field(...)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    chat_history: List[ChatMessage] = Field(default=[])

class RAGResponse(BaseModel):
    question: str
    answer: str
    model: str
    chunks_retrieved: int
    context: Optional[str] = None
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model: str
    faiss_index: str
    version: str

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to inject execution time into response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", tags=["System"])
async def root():
    return {"message": "Clinical RAG API Service operational.", "docs_url": "/docs", "health_url": "/health"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe endpoint."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain initialization incomplete.")
    return HealthResponse(status="healthy", model="llama-3.3-70b-versatile", faiss_index="loaded", version="1.0.0")

@app.post("/ask", response_model=RAGResponse, tags=["Inference"])
async def ask_question(request: QuestionRequest):
    """Execute a single query against the RAG pipeline."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    
    start_time = time.time()
    try:
        result = rag_chain.ask(question=request.question)
    except Exception as e:
        logger.error(f"Inference error on /ask endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error.")

    latency = round((time.time() - start_time) * 1000, 2)
    return RAGResponse(
        question=result["question"],
        answer=result["answer"],
        model=result["model"],
        chunks_retrieved=result["chunks_retrieved"],
        context=result.get("context") if request.verbose else None,
        latency_ms=latency
    )

@app.post("/chat", response_model=RAGResponse, tags=["Inference"])
async def chat(request: ChatRequest):
    """Execute a conversational query with memory context."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service unavailable.")

    start_time = time.time()
    try:
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        result = rag_chain.ask_with_history(question=request.question, chat_history=history)
    except Exception as e:
        logger.error(f"Inference error on /chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal chat inference error.")

    latency = round((time.time() - start_time) * 1000, 2)
    return RAGResponse(
        question=request.question,
        answer=result["answer"],
        model=result["model"],
        chunks_retrieved=result["chunks_retrieved"],
        latency_ms=latency
    )

@app.get("/risk", response_model=RAGResponse, tags=["Analytics"])
async def risk_triage():
    """Execute organization-wide risk stratification against all embedded records."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service unavailable.")

    start_time = time.time()
    try:
        query = "Identify patients at elevated clinical risk. Categorize them into High, Medium, and Low risk cohorts."
        result = rag_chain.ask(query)
    except Exception as e:
        logger.error("Error executing risk triage", exc_info=True)
        raise HTTPException(status_code=500, detail="Risk analysis failed.")

    latency = round((time.time() - start_time) * 1000, 2)
    return RAGResponse(
        question="Comprehensive Risk Stratification",
        answer=result["answer"],
        model=result["model"],
        chunks_retrieved=result["chunks_retrieved"],
        latency_ms=latency
    )

@app.get("/patient/{patient_id}", response_model=RAGResponse, tags=["Analytics"])
async def get_patient_summary(patient_id: str):
    """Retrieve synthesized clinical summary for a specific Patient ID."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service unavailable.")

    pid = patient_id.upper().strip()
    if not pid.startswith("P") or not pid[1:].isdigit():
        raise HTTPException(status_code=422, detail="Invalid format for Patient ID. Expected format: P[0-9]+")

    start_time = time.time()
    try:
        result = rag_chain.ask(f"Please provide a comprehensive clinical summary for patient {pid}.")
    except Exception as e:
        logger.error(f"Error executing summarize for PID {pid}", exc_info=True)
        raise HTTPException(status_code=500, detail="Patient summarization failed.")

    latency = round((time.time() - start_time) * 1000, 2)
    return RAGResponse(
        question=f"Clinical Summary: {pid}",
        answer=result["answer"],
        model=result["model"],
        chunks_retrieved=result["chunks_retrieved"],
        latency_ms=latency
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled system error on path: {request.url}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"message": "System error occurred.", "path": str(request.url)}
    )

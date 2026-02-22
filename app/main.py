import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Add src/ to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from llm_chain import ClinicalRAGChain

# ── Global RAG chain instance (loaded once at startup) ────────────────────────
rag_chain: Optional[ClinicalRAGChain] = None


# ── Lifespan: startup + shutdown events ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the RAG chain ONCE when the server starts.
    Keeps it in memory for all subsequent requests.
    Avoids reloading ClinicalBERT + FAISS on every request.
    """
    global rag_chain
    print("🚀 Starting ClinicalBot API...")
    print("⏳ Loading RAG chain (ClinicalBERT + FAISS + Groq)...")
    rag_chain = ClinicalRAGChain(k=4)
    print("✅ RAG chain ready! API is live.\n")
    yield
    # Shutdown
    print("🛑 Shutting down ClinicalBot API...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ClinicalBot RAG API",
    description = "Clinical NLP RAG Chatbot powered by ClinicalBERT + FAISS + Groq Llama 3.3",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── CORS Middleware (allows Streamlit UI to call this API) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response Models (Pydantic) ──────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length  = 3,
        max_length  = 500,
        description = "Clinical question to ask",
        example     = "What are the symptoms of the patient with chest pain?"
    )
    verbose: bool = Field(
        default     = False,
        description = "If True, includes retrieved context in response"
    )


class ChatMessage(BaseModel):
    role:    str = Field(..., example="user")
    content: str = Field(..., example="Tell me about stroke patients")


class ChatRequest(BaseModel):
    question:     str              = Field(..., min_length=3, max_length=500)
    chat_history: list[ChatMessage] = Field(
        default     = [],
        max_items   = 10,
        description = "Previous messages for conversational context"
    )


class RAGResponse(BaseModel):
    question:         str
    answer:           str
    model:            str
    chunks_retrieved: int
    context:          Optional[str] = None
    latency_ms:       float


class HealthResponse(BaseModel):
    status:      str
    model:       str
    faiss_index: str
    version:     str


# ── Middleware: Request timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round((time.time() - start) * 1000, 2))
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🏥 ClinicalBot RAG API is running!",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Used by Docker HEALTHCHECK and Kubernetes liveness probe.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    return HealthResponse(
        status      = "healthy",
        model       = "llama-3.3-70b-versatile",
        faiss_index = "loaded",
        version     = "1.0.0"
    )


@app.post("/ask", response_model=RAGResponse, tags=["Clinical Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Single question endpoint.
    Runs full RAG pipeline: retrieve → prompt → Groq → answer.

    Example questions:
    - "What are the symptoms of the cardiac patient?"
    - "Which patients have diabetes?"
    - "What was prescribed to patient P004?"
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    start = time.time()
    try:
        result = rag_chain.ask(
            question = request.question,
            verbose  = False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG chain error: {str(e)}")

    latency = round((time.time() - start) * 1000, 2)

    return RAGResponse(
        question         = result["question"],
        answer           = result["answer"],
        model            = result["model"],
        chunks_retrieved = result["chunks_retrieved"],
        context          = result["context"] if request.verbose else None,
        latency_ms       = latency
    )


@app.post("/chat", response_model=RAGResponse, tags=["Clinical Q&A"])
async def chat(request: ChatRequest):
    """
    Conversational endpoint with memory.
    Pass previous messages in chat_history to maintain context.

    Example:
    {
      "question": "What treatment was given to that patient?",
      "chat_history": [
        {"role": "user",      "content": "Tell me about patient P010"},
        {"role": "assistant", "content": "Patient P010 has acute ischemic stroke..."}
      ]
    }
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    start = time.time()
    try:
        history = [{"role": m.role, "content": m.content}
                   for m in request.chat_history]

        result  = rag_chain.ask_with_history(
            question     = request.question,
            chat_history = history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    latency = round((time.time() - start) * 1000, 2)

    return RAGResponse(
        question         = request.question,
        answer           = result["answer"],
        model            = result["model"],
        chunks_retrieved = result["chunks_retrieved"],
        latency_ms       = latency
    )


@app.get("/risk", response_model=RAGResponse, tags=["Clinical Tools"])
async def risk_triage():
    """
    Runs a full risk triage across all patients in the database.
    Returns HIGH / MEDIUM / LOW risk classifications.
    No input needed — analyzes all loaded clinical records.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    start = time.time()
    try:
        result = rag_chain.ask(
            "Which patients are at highest risk and need urgent attention? "
            "Triage all patients by risk level."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    return RAGResponse(
        question         = "Full patient risk triage",
        answer           = result["answer"],
        model            = result["model"],
        chunks_retrieved = result["chunks_retrieved"],
        latency_ms       = latency
    )


@app.get("/patient/{patient_id}", response_model=RAGResponse, tags=["Clinical Tools"])
async def get_patient_summary(patient_id: str):
    """
    Summarizes a specific patient record by ID.
    Example: GET /patient/P001
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    patient_id = patient_id.upper().strip()

    # Validate format
    if not patient_id.startswith("P") or not patient_id[1:].isdigit():
        raise HTTPException(
            status_code = 422,
            detail      = "Invalid patient ID format. Use format: P001, P002, etc."
        )

    start = time.time()
    try:
        result = rag_chain.ask(f"Summarize patient {patient_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    return RAGResponse(
        question         = f"Summary for patient {patient_id}",
        answer           = result["answer"],
        model            = result["model"],
        chunks_retrieved = result["chunks_retrieved"],
        latency_ms       = latency
    )


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":   str(exc),
            "path":    str(request.url),
            "message": "Internal server error. Check logs."
        }
    )

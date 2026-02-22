import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

INDEX_PATH = Path("faiss_index")
EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


@dataclass
class RetrievedChunk:
    content: str
    patient_id: str
    risk_level: str
    score: float = 0.0

    def __repr__(self):
        return f"[{self.patient_id} | Risk: {self.risk_level}]\n{self.content[:150]}..."


class ClinicalRetriever:
    """Manages vector embeddings and FAISS index retrieval operations."""

    def __init__(self):
        logger.info(f"Initializing ClinicalRetriever with model: {EMBED_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        if not INDEX_PATH.exists():
            logger.error(f"FAISS index path not found: {INDEX_PATH}")
            raise FileNotFoundError(f"FAISS index not found at '{INDEX_PATH}'.")

        logger.info("Loading FAISS index into memory...")
        self.vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")

    def similarity_search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Execute a standard cosine similarity search."""
        docs_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(
                content=doc.page_content,
                patient_id=doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level=doc.metadata.get("risk_level", "UNKNOWN"),
                score=round(float(score), 4)
            )
            for doc, score in docs_scores
        ]

    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[RetrievedChunk]:
        """Execute a Max Marginal Relevance (MMR) search for diversity."""
        docs = self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        return [
            RetrievedChunk(
                content=doc.page_content,
                patient_id=doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level=doc.metadata.get("risk_level", "UNKNOWN")
            )
            for doc in docs
        ]

    def format_context_for_llm(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a standardized context block for LLM inference."""
        formatted_chunks = [
            f"[Source: {chunk.patient_id} | Risk: {chunk.risk_level}]\n{chunk.content}"
            for chunk in chunks
        ]
        return "\n\n".join(formatted_chunks)

    def search_by_risk_level(self, risk_level: str) -> List[RetrievedChunk]:
        """Filter the vector store by specific risk level metadata."""
        docs = self.vectorstore.similarity_search(
            query=risk_level,
            k=20,
            filter={"risk_level": risk_level.upper()}
        )
        return [
            RetrievedChunk(
                content=doc.page_content,
                patient_id=doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level=doc.metadata.get("risk_level", "UNKNOWN")
            )
            for doc in docs
        ]

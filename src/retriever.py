import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Paths & Config ────────────────────────────────────────────────────────────
INDEX_PATH  = Path("faiss_index")
EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"   # same model used in indexer!

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    content:    str
    patient_id: str
    risk_level: str
    score:      float = 0.0

    def __repr__(self):
        return (
            f"[{self.patient_id} | Risk: {self.risk_level}]\n"
            f"{self.content[:200]}..."
        )


# ── Main Retriever Class ──────────────────────────────────────────────────────
class ClinicalRetriever:
    """
    Loads the FAISS index once and exposes two search methods:
      1. similarity_search()  → pure vector similarity
      2. mmr_search()         → diverse results (recommended for clinical)
    """

    def __init__(self):
        print("⏳ Loading ClinicalBERT embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(f"⏳ Loading FAISS index from {INDEX_PATH}...")
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"❌ FAISS index not found at '{INDEX_PATH}'. "
                "Run src/indexer.py first!"
            )

        # allow_dangerous_deserialization=True is required by LangChain
        # for loading local FAISS indexes (safe since WE created it)
        self.vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Retriever ready!\n")


    def similarity_search(
        self,
        query:   str,
        k:       int = 4
    ) -> list[RetrievedChunk]:
        """
        Pure cosine similarity — returns the k most similar chunks.
        Fast but may return multiple chunks from the same patient.
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs_and_scores:
            results.append(RetrievedChunk(
                content    = doc.page_content,
                patient_id = doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level = doc.metadata.get("risk_level", "UNKNOWN"),
                score      = round(float(score), 4)
            ))
        return results


    def mmr_search(
        self,
        query:       str,
        k:           int   = 4,
        fetch_k:     int   = 20,
        lambda_mult: float = 0.5
    ) -> list[RetrievedChunk]:
        """
        Max Marginal Relevance search.
        - fetch_k    : how many candidates FAISS retrieves first (wider net)
        - lambda_mult: 0.0 = max diversity | 1.0 = max similarity
        - 0.5        : balanced (recommended for clinical notes)

        This ensures we don't get 4 chunks all from the same patient record.
        """
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k           = k,
            fetch_k     = fetch_k,
            lambda_mult = lambda_mult
        )

        results = []
        for doc in docs:
            results.append(RetrievedChunk(
                content    = doc.page_content,
                patient_id = doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level = doc.metadata.get("risk_level", "UNKNOWN"),
            ))
        return results


    def format_context_for_llm(self, chunks: list[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into a single context string
        that will be injected into the LLM prompt in Step 3.

        Output format:
        ─────────────────────────────────────
        [Source: P001 | Risk: HIGH]
        Patient has chest pain radiating to left arm...

        [Source: P007 | Risk: HIGH]
        Acute decompensated heart failure...
        ─────────────────────────────────────
        """
        formatted = []
        for chunk in chunks:
            formatted.append(
                f"[Source: {chunk.patient_id} | Risk: {chunk.risk_level}]\n"
                f"{chunk.content}"
            )
        return "\n\n".join(formatted)


    def search_by_risk_level(self, risk_level: str) -> list[RetrievedChunk]:
        """
        Metadata filter — returns all chunks for a specific risk level.
        Useful for: 'show me all HIGH risk patients'
        """
        # LangChain FAISS supports simple metadata filtering
        docs = self.vectorstore.similarity_search(
            query  = risk_level,
            k      = 20,
            filter = {"risk_level": risk_level.upper()}
        )

        results = []
        for doc in docs:
            results.append(RetrievedChunk(
                content    = doc.page_content,
                patient_id = doc.metadata.get("patient_id", "UNKNOWN"),
                risk_level = doc.metadata.get("risk_level", "UNKNOWN"),
            ))
        return results


# ── Test / Demo ───────────────────────────────────────────────────────────────
def main():
    retriever = ClinicalRetriever()

    # ── Test 1: Similarity Search ─────────────────────────────────────────────
    print("=" * 55)
    print("TEST 1: Similarity Search — 'cardiac chest pain'")
    print("=" * 55)
    results = retriever.similarity_search("cardiac chest pain", k=3)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i} (score={r.score}):")
        print(r)

    # ── Test 2: MMR Search ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 2: MMR Search — 'patient with neurological symptoms'")
    print("=" * 55)
    results = retriever.mmr_search("patient with neurological symptoms", k=3)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(r)

    # ── Test 3: Metadata Filter ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 3: Metadata Filter — all HIGH risk patients")
    print("=" * 55)
    results = retriever.search_by_risk_level("HIGH")
    seen_patients = set()
    for r in results:
        if r.patient_id not in seen_patients:
            print(f"  → {r.patient_id} | {r.risk_level}")
            seen_patients.add(r.patient_id)

    # ── Test 4: Format context for LLM ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 4: Formatted context (what LLM will receive in Step 3)")
    print("=" * 55)
    chunks  = retriever.mmr_search("stroke symptoms weakness", k=2)
    context = retriever.format_context_for_llm(chunks)
    print(context)

    print("\n✅ Step 2 Complete! Run step 3 next: llm_chain.py")


if __name__ == "__main__":
    main()

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/sample_clinical_notes.txt")
INDEX_PATH = Path("faiss_index")

# ── Embedding model (ClinicalBERT via HuggingFace, runs locally, FREE) ───────
EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def load_documents(path: Path) -> list[dict]:
    """
    Load the clinical notes file and split into
    individual patient records on the '---' separator.
    Returns a list of dicts: {page_content, metadata}
    """
    raw_text = path.read_text(encoding="utf-8")
    records  = [r.strip() for r in raw_text.split("---") if r.strip()]

    documents = []
    for record in records:
        # Extract PATIENT_ID for metadata
        match = re.search(r"PATIENT_ID:\s*(\w+)", record)
        patient_id = match.group(1) if match else "UNKNOWN"

        # Extract RISK_LEVEL for metadata
        risk_match = re.search(r"RISK_LEVEL:\s*(\w+)", record)
        risk_level = risk_match.group(1) if risk_match else "UNKNOWN"

        documents.append({
            "page_content": record,
            "metadata": {
                "patient_id": patient_id,
                "risk_level": risk_level,
                "source": str(path)
            }
        })

    print(f"✅ Loaded {len(documents)} clinical records")
    return documents


def chunk_documents(documents: list[dict]) -> list:
    """
    Split each clinical record into smaller overlapping chunks.
    chunk_size=300  → ~300 characters per chunk
    chunk_overlap=50 → 50-char overlap so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", ". ", " "]  # split on newlines first, then sentences
    )

    from langchain_core.documents import Document
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["page_content"])
        for split in splits:
            chunks.append(
                Document(
                    page_content=split,
                    metadata=doc["metadata"]
                )
            )

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} records")
    return chunks


def build_faiss_index(chunks: list) -> FAISS:
    """
    Embed all chunks using ClinicalBERT and store in FAISS.
    This runs LOCALLY — no API call needed for embeddings.
    """
    print(f"⏳ Loading embedding model: {EMBED_MODEL}")
    print("   (First run downloads ~400MB model — cached after that)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    print("⏳ Building FAISS index (embedding all chunks)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index to disk so we don't rebuild every time
    vectorstore.save_local(str(INDEX_PATH))
    print(f"✅ FAISS index saved to: {INDEX_PATH}/")

    return vectorstore


def main():
    print("=" * 55)
    print("  Clinical RAG — Step 1: Indexing Pipeline")
    print("=" * 55)

    # Step A: Load raw clinical notes
    docs   = load_documents(DATA_PATH)

    # Step B: Chunk into smaller pieces
    chunks = chunk_documents(docs)

    # Step C: Embed + store in FAISS
    vs     = build_faiss_index(chunks)

    # Quick sanity-check search
    print("\n🔍 Sanity check — searching: 'chest pain cardiac'")
    results = vs.similarity_search("chest pain cardiac", k=2)
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i} [{r.metadata['patient_id']} | {r.metadata['risk_level']}]:")
        print(f"  {r.page_content[:120]}...")

    print("\n✅ Step 1 Complete! Run step 2 next: retriever.py")


if __name__ == "__main__":
    main()

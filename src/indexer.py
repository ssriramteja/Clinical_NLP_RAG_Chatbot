import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DATA_PATH = Path("data/sample_clinical_notes.txt")
INDEX_PATH = Path("faiss_index")
EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def load_documents(path: Path) -> List[Dict[str, Any]]:
    """Load clinical notes and extract individual patient records."""
    if not path.exists():
        logger.error(f"Data file not found at {path}")
        raise FileNotFoundError(f"Data file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    records = [r.strip() for r in raw_text.split("---") if r.strip()]

    documents = []
    for record in records:
        match = re.search(r"PATIENT_ID:\s*(\w+)", record)
        patient_id = match.group(1) if match else "UNKNOWN"

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

    logger.info(f"Loaded {len(documents)} clinical records from {path}")
    return documents


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Document]:
    """Split clinical records into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", ". ", " "]
    )

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

    logger.info(f"Created {len(chunks)} document chunks")
    return chunks


def build_faiss_index(chunks: List[Document]) -> FAISS:
    """Embed chunks and generate a FAISS vector index."""
    logger.info(f"Initializing embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    logger.info(f"FAISS index persisted to {INDEX_PATH}")

    return vectorstore


def main():
    logger.info("Starting indexing pipeline")
    
    try:
        docs = load_documents(DATA_PATH)
        chunks = chunk_documents(docs)
        vectorstore = build_faiss_index(chunks)
        
        results = vectorstore.similarity_search("chest pain cardiac", k=1)
        if results:
            logger.info(f"Index validation successful. Top match patient ID: {results[0].metadata.get('patient_id')}")
            
    except Exception as e:
        logger.error(f"Indexing pipeline failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

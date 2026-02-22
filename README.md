# 🏥 Clinical NLP RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) system specifically designed for clinical and healthcare data. This application enables healthcare professionals to query clinical notes naturally, perform risk stratification, and fetch patient summaries seamlessly using local embeddings and a powerful open-source Large Language Model (LLM).

**Live Demo**: [Hugging Face Space Deployment](https://huggingface.co/spaces/tejacse888/Clinical_nlp_rag_chatbot)

---

## 🧠 Conceptual Architecture

Retrieval-Augmented Generation (RAG) bridges the gap between static LLMs and private data. In a healthcare context, this means we can ask questions about our specific, proprietary patient records without needing to fine-tune a model from scratch.

### The Pipeline Workflow:
1. **Ingestion & Chunking**: Clinical records are loaded and split into overlapping textual chunks to maintain context without exceeding token limits.
2. **Semantic Embedding**: Each chunk is transformed into a high-dimensional vector using `emilyalsentzer/Bio_ClinicalBERT`, an embedding model specifically pre-trained on MIMIC-III clinical records.
3. **Vector Database**: These vectors are stored in a local **FAISS** index, allowing for blazing-fast similarity searches.
4. **Retrieval**: When a user asks a question, the query is also embedded. The system searches FAISS for the most clinically relevant chunks using either standard Cosine Similarity or Maximal Marginal Relevance (MMR) for diverse context.
5. **Generation**: The retrieved context and the user query are injected into a strict, clinically-focused prompt template and sent to the LLM (**Groq Llama 3.3 70B**) to generate an accurate, grounded response.

---

## 🛠️ Steps Performed & System Components

This project was built systematically through the following modules:

### Step 1: Data Preparation & Indexing (`src/indexer.py`)
- Created synthetic but highly realistic clinical notes featuring Chief Complaints, History, Vitals, Assessments, Plans, and Risk Levels.
- Utilized LangChain's `RecursiveCharacterTextSplitter` to optimally segment the documents.
- Generated offline vector embeddings using Hugging Face's `Bio_ClinicalBERT` for privacy-first healthcare encoding.
- Persisted the encoded documents to disk via a local `faiss-cpu` vector store.

### Step 2: Advanced Retrieval Engine (`src/retriever.py`)
- Configured the `ClinicalRetriever` class to load the FAISS index efficiently out of memory.
- Implemented **Similarity Search** for direct question answering.
- Implemented **MMR (Maximal Marginal Relevance) Search** to retrieve diverse chunks and prevent context duplication.
- Implemented **Risk-Level Metadata Filtering** to allow querying specific patient cohorts (e.g., exclusively "HIGH" risk patients).

### Step 3: LLM Orchestration (`src/llm_chain.py` & `src/prompts.py`)
- Engineered strict **System Prompts** preventing the LLM from hallucinating clinical data.
- Implemented a dynamic Prompt Router (`get_prompt_for_query`) that selects specialized prompt templates (Triage, QA, Patient Summary, Treatment Plan) based on keyword intent.
- Integrated **Groq Inference Engine** for sub-second, lightning-fast text generation utilizing the powerful `llama-3.3-70b-versatile` model via LangChain's Expression Language (LCEL).

### Step 4: RESTful API Backend (`app/main.py`)
- Built a robust, production-ready microservice using **FastAPI** and Pydantic.
- Created heavily-typed functional endpoints:
  - `/ask`: Direct single-turn QA.
  - `/chat`: Multi-turn conversational endpoint respecting historical context.
  - `/risk`: Automated batch risk stratification across all loaded patients.
  - `/patient/{id}`: Specific patient chart summarization.
- Incorporated API middleware to track and inject execution latency headers.

### Step 5: Frontend Interface (`ui/streamlit_app.py`)
- Engineered a modern, state-of-the-art **Streamlit UI** heavily customized with raw CSS injections for an enterprise SaaS feel.
- Structured the interface around discrete modes: *Chat Interface*, *Risk Stratification*, and *Chart Review*.
- Added robust session-state management to preserve conversational flow and track real-time telemetry (API latency, query counts).

### Step 6: Containerization & Deployment
- Wrote an optimized `Dockerfile` leveraging a lightweight Python 3.11 image.
- Created an orchestrated shell entrypoint (`start.sh`) that dynamically generates the FAISS index and immediately spawns both the FastAPI backend and Streamlit frontend.
- Successfully deployed the system via **Hugging Face Spaces**.

---

## 🚀 Local Quick Start

If you wish to run the pipeline locally:

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set Environment Variables**:
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

3. **Start the Application**:
```bash
# This will build the index and launch both Uvicorn and Streamlit
./start.sh
```

4. **Access Endpoints**:
- UI Interface: `http://localhost:8501`
- Backend API Docs: `http://localhost:8000/docs`

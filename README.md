# 🏥 Clinical RAG Chatbot

> Clinical NLP Q&A system powered by **ClinicalBERT + FAISS + Groq Llama 3.3**

## 🔄 Architecture
```
Clinical Notes → ClinicalBERT Embeddings → FAISS Index
User Question  → Semantic Search → LangChain Prompt → Groq Llama 3.3 → Answer
```

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python src/indexer.py          # build FAISS index
uvicorn app.main:app --reload  # start API
streamlit run ui/streamlit_app.py  # start UI
```

## 🛠️ Tech Stack
- **Embeddings**: ClinicalBERT (HuggingFace, local)
- **Vector Store**: FAISS
- **LLM**: Groq Llama 3.3-70b (free)
- **Chain**: LangChain LCEL
- **API**: FastAPI + Pydantic
- **UI**: Streamlit

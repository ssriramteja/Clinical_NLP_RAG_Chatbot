#!/bin/bash
# Generate the FAISS index
echo "Building FAISS Index..."
python src/indexer.py

# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to start
sleep 5

# Start the Streamlit frontend
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

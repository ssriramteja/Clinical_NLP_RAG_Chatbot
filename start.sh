#!/bin/bash
echo "Starting Clinical RAG Services..."

# Start the FastAPI backend in the background
echo "Launching FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &

# Wait for the backend to start
echo "Waiting for services to initialize..."
sleep 5

# Start the Streamlit frontend on HF standard port 7860
echo "Launching Streamlit UI on port 7860..."
streamlit run ui/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

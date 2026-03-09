# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to start
sleep 2

# Start the Streamlit frontend on HF standard port 7860
streamlit run ui/streamlit_app.py --server.port 7860 --server.address 0.0.0.0

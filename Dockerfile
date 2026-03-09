FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p faiss_index

# Copy everything
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Pre-build FAISS index during image creation
RUN python src/indexer.py

# Expose port (Hugging Face standard)
EXPOSE 7860

# Run the start script
CMD ["./start.sh"]

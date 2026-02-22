FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Run the start script
CMD ["./start.sh"]

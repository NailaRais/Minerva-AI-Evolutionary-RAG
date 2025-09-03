FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Minerva
COPY . .
RUN pip install --no-cache-dir -e .

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Download models
RUN ollama pull phi3:3.8b-q4_0

# Create directories
RUN mkdir -p /app/data /app/models /app/index

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=0.0.0.0

# Expose ports
EXPOSE 11434  # Ollama
EXPOSE 8000   # Optional web interface

# Default command
CMD ["python", "-m", "minerva.cli", "--help"]

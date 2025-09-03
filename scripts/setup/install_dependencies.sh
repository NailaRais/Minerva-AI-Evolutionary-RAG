#!/bin/bash
set -e

echo "🔧 Installing Minerva Dependencies"
echo "================================="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Install system dependencies
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-pip build-essential
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv minerva-env
source minerva-env/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -e .[dev]

# Download embedding model
echo "Downloading embedding model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model ready!')
"

# Check Ollama installation
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Download Phi-3 model
echo "Downloading Phi-3 model..."
ollama pull phi3:3.8b-q4_0

echo "✅ Installation complete!"
echo "Activate virtual environment: source minerva-env/bin/activate"

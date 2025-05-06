#!/bin/bash

# Setup script for the RAG application
# Make it executable with: chmod +x setup.sh

echo "Setting up RAG application with pgvector and Gradio (using local Ollama)..."

# Create necessary directories
mkdir -p app data

# Check if local Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "WARNING: Your local Ollama doesn't seem to be running."
  echo "Please start Ollama on your host machine before using this application."
  echo "Run: ollama serve"
  exit 1
fi

# Check if gemma3:1b is pulled in local Ollama
if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3:1b"; then
  echo "The model 'gemma3:1b' doesn't seem to be pulled in your local Ollama."
  echo "Do you want to pull it now? (y/n)"
  read -r answer
  if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo "Pulling gemma3:1b model..."
    ollama pull gemma3:1b
  else
    echo "Please pull the model manually using: ollama pull gemma3:1b"
    echo "Then run this script again."
    exit 1
  fi
fi

# Clean up any previous containers
echo "Cleaning up previous containers..."
docker compose down

# Add local pip cache directory to speed up builds
mkdir -p ~/.pip-cache

echo "Building Docker containers with pip cache..."
# Use build with pip cache to speed up builds
docker compose build --build-arg PIP_CACHE_DIR=/tmp/pip-cache

echo "Starting Docker containers..."
docker compose up -d

# Check if containers are running
if [ "$(docker ps -q -f name=webapp)" ] && [ "$(docker ps -q -f name=postgres)" ]; then
  echo "Setup complete! Access the Gradio interface at: http://localhost:7860"
  echo "Place your text files in the 'data' directory and use the 'Document Ingestion' tab to process them."
  echo ""
  echo "Using your local Ollama instance with gemma3:1b model at http://localhost:11434"
else
  echo "ERROR: Containers failed to start. Check the logs for more information:"
  echo "docker compose logs"
fi
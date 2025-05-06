#!/bin/bash

# This script pulls the necessary Ollama model
# Make it executable with: chmod +x init-ollama.sh

# Set variables
MODEL_NAME=${1:-"gemma3:4b"}

echo "Initializing Ollama with model: $MODEL_NAME"

# Pull the model
curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$MODEL_NAME\"}"

echo "Model $MODEL_NAME has been pulled successfully!"
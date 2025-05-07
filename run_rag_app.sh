#!/bin/bash

# Script to run the complete RAG application

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if the database container is running
if ! docker ps | grep -q postgres; then
    echo "Starting vector database containers..."

    # Run the setup script
    bash setup_vector_db.sh

    # Wait for database to be fully initialized
    echo "Waiting for the database to be ready..."
    sleep 10
else
    echo "Vector database is already running."
fi

# Check if required Python packages are installed
if ! pip list | grep -q gradio; then
    echo "Installing required Python packages..."
    pip install -r requirements.txt
fi

# Launch the Gradio app
echo "Starting the RAG application interface..."
cd app
python gradio_app.py

# Show a helpful message when the application exits
echo "
==================================================
  RAG application has been stopped.

  - To restart, run this script again
  - To stop the database containers, run:
      docker-compose -f pgvector-docker-compose.yml down
==================================================
"
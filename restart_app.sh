#!/bin/bash

# Script to restart the RAG application with fixes

echo "Restarting the RAG application with fixes..."

# Copy the fixed vector_db.py file to replace the original
cp vector_db.py app/vector_db.py

# Stop any running Gradio processes
echo "Stopping any running Gradio processes..."
pkill -f "python gradio_app.py" || true

# Launch the Gradio app
echo "Starting the fixed RAG application interface..."
cd app
python gradio_app.py

echo "
==================================================
  RAG application has been stopped.

  - To restart, run this script again
  - To stop the database containers, run:
      docker-compose -f pgvector-docker-compose.yml down
==================================================
"
import os
import gradio as gr
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import requests
import json
from pathlib import Path

# Environment variables
DB_CONNECTION_STRING = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/ragdb")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma3:1b")
COLLECTION_NAME = "documents"
EMBEDDING_DIM = 384  # Dimension for 'all-MiniLM-L6-v2'

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize database connection and create extension if needed
def init_db():
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()

    # Create pgvector extension if it doesn't exist
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create tables for storing documents and embeddings
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB NOT NULL,
        embedding vector({EMBEDDING_DIM})
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully")

# Simple wrapper for Ollama API
def query_ollama(prompt, model_name=MODEL_NAME, temperature=0.5):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json().get("response", "No response from model")
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return f"Error: {str(e)}"

# Get embedding for text
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Process documents from a directory
def process_documents(directory_path):
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Check if directory exists
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Error: Directory {directory_path} does not exist"

        # Find all text files
        file_count = 0
        chunk_count = 0

        for file_path in dir_path.glob("**/*.txt"):
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Simple text chunking (split by paragraphs with some overlap)
                paragraphs = [p for p in content.split('\n\n') if p.strip()]
                chunks = []

                # Process each paragraph as a chunk
                for paragraph in paragraphs:
                    if len(paragraph.strip()) > 10:  # Ignore very short paragraphs
                        chunks.append(paragraph)

                # Add each chunk to the database
                for chunk in chunks:
                    # Get embedding for the chunk
                    embedding = get_embedding(chunk)

                    # Store in database
                    cursor.execute(
                        """
                        INSERT INTO documents (content, metadata, embedding)
                        VALUES (%s, %s, %s)
                        """,
                        (
                            chunk,
                            json.dumps({"source": str(file_path)}),
                            embedding
                        )
                    )

                chunk_count += len(chunks)
                file_count += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        return f"Successfully processed {file_count} files with {chunk_count} chunks"
    except Exception as e:
        return f"Error processing documents: {str(e)}"

# Search for relevant documents
def search_similar_documents(query, top_k=5):
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)

        # Connect to database
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Search for similar documents using cosine similarity
        cursor.execute(
            """
            SELECT content, metadata,
                   1 - (embedding <=> %s) as similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (query_embedding, top_k)
        )

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

# Query the RAG system
def query_rag(query, use_retrieval=True):
    try:
        if use_retrieval:
            # Find relevant documents
            start_time = time.time()
            documents = search_similar_documents(query)

            if not documents:
                return "No relevant documents found", "No documents retrieved"

            # Format context from retrieved documents
            context = "\n\n".join([doc[0] for doc in documents])

            # Create prompt with context
            prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {query}

Answer:"""

            # Get response from Ollama
            result = query_ollama(prompt)
            end_time = time.time()

            # Format source information
            sources = []
            for i, doc in enumerate(documents):
                metadata = json.loads(doc[1])
                source = metadata.get('source', 'Unknown')
                similarity = doc[2]
                sources.append(f"Source {i+1}: {source} (similarity: {similarity:.4f})")

            sources_text = "\n".join(sources)
            response_time = end_time - start_time

            return (
                result,
                f"Retrieved from {len(documents)} documents in {response_time:.2f} seconds.\n\n{sources_text}"
            )
        else:
            # Use LLM directly without retrieval
            start_time = time.time()
            result = query_ollama(query)
            end_time = time.time()

            response_time = end_time - start_time
            return (
                result,
                f"Direct LLM response (no retrieval) in {response_time:.2f} seconds."
            )
    except Exception as e:
        return f"Error: {str(e)}", "An error occurred"

# Check status of Ollama and available models
def check_status():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            return f"Ollama is running. Available models: {', '.join(models)}"
        else:
            return "Ollama is running but couldn't retrieve model list."
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Initialize database
init_db()

# Gradio Interface
with gr.Blocks(title="RAG with pgvector & Ollama") as demo:
    gr.Markdown("# Simple RAG Application")

    with gr.Tab("Document Ingestion"):
        gr.Markdown("## Ingest Documents")
        doc_dir = gr.Textbox(label="Document Directory", value="/data")
        ingest_btn = gr.Button("Ingest Documents")
        ingest_output = gr.Textbox(label="Ingestion Result")

        ingest_btn.click(process_documents, inputs=doc_dir, outputs=ingest_output)

    with gr.Tab("Query System"):
        gr.Markdown("## Query the System")
        with gr.Row():
            query_input = gr.Textbox(label="Your Question", lines=2)
            use_rag = gr.Checkbox(label="Use RAG (Retrieval)", value=True)

        query_btn = gr.Button("Ask")

        with gr.Row():
            answer_output = gr.Textbox(label="Answer", lines=10)
            sources_output = gr.Textbox(label="Sources & Info", lines=10)

        query_btn.click(
            query_rag,
            inputs=[query_input, use_rag],
            outputs=[answer_output, sources_output]
        )

    with gr.Tab("System Status"):
        gr.Markdown("## System Status")
        status_btn = gr.Button("Check Status")
        status_output = gr.Textbox(label="Status")

        status_btn.click(check_status, inputs=None, outputs=status_output)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
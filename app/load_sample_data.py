#!/usr/bin/env python
"""
Load sample documents into vector database for testing.
This script adds some basic vector database and RAG content for testing.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from vector_db import VectorDB

def main():
    """Load sample documents into vector database."""
    print("Loading sample documents into vector database...")

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize vector database
    connection_string = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"
    )
    db = VectorDB(
        connection_string=connection_string,
        embedding_dim=384  # Dimension for 'all-MiniLM-L6-v2'
    )

    # Check if there are already documents in the database
    doc_count = db.count_documents()
    print(f"Database currently has {doc_count} documents.")

    if doc_count > 0:
        choice = input("Database already has documents. Load more sample data? (y/n): ")
        if choice.lower() != 'y':
            print("Exiting without adding documents.")
            return

    # Sample documents about vector databases and RAG
    sample_docs = [
        "Vector databases store and query vector embeddings efficiently using specialized indexing structures.",
        "PostgreSQL with pgvector extension supports similarity search using various metrics like cosine, L2, and inner product.",
        "RAG (Retrieval-Augmented Generation) enhances language models with external knowledge without retraining.",
        "Embeddings represent text as numerical vectors in high-dimensional space, capturing semantic meaning.",
        "Cosine similarity measures the angle between two vectors, making it useful for comparing document similarity.",
        "HNSW (Hierarchical Navigable Small World) is an indexing algorithm that creates a layered graph for fast approximate nearest neighbor search.",
        "IVF (Inverted File Index) divides the vector space into clusters, speeding up search by examining only relevant clusters.",
        "Chunking documents into smaller segments helps maintain context while reducing vector dimensionality for more effective retrieval.",
        "Reranking retrieved documents can improve RAG quality by sorting results based on relevance to the query.",
        "Vector databases excel at semantic search, finding conceptually similar items even when exact keywords don't match."
    ]

    # Generate embeddings and insert documents
    inserted_count = 0
    for i, doc in enumerate(sample_docs):
        try:
            # Generate embedding
            embedding = embedding_model.encode(doc).tolist()

            # Insert with metadata
            metadata = {
                "source": "sample_data",
                "topic": "vector_databases_and_rag",
                "document_id": f"sample_{i+1}"
            }

            doc_id = db.insert_document(doc, embedding, metadata)
            print(f"Inserted document {doc_id}: {doc[:50]}...")
            inserted_count += 1

        except Exception as e:
            print(f"Error inserting document: {e}")

    print(f"\nSuccessfully inserted {inserted_count} sample documents.")
    print(f"Database now has {db.count_documents()} total documents.")

if __name__ == "__main__":
    main()
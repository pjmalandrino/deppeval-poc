#!/usr/bin/env python
"""
Demo script showing basic usage of the VectorDB class.
This demonstrates how to insert documents, search for similar documents,
and perform other basic operations with the vector database.
"""

from vector_db import VectorDB
from sentence_transformers import SentenceTransformer
import os
import time

def main():
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize vector database
    print("Initializing vector database...")
    db = VectorDB(
        connection_string=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"),
        embedding_dim=384  # Dimension for 'all-MiniLM-L6-v2'
    )

    # Create some sample documents
    sample_docs = [
        "Vector databases store and query vector embeddings efficiently.",
        "PostgreSQL with pgvector extension supports similarity search using various metrics.",
        "RAG systems enhance language models with external knowledge without retraining.",
        "Embeddings represent text as numerical vectors in high-dimensional space.",
        "Cosine similarity measures the angle between two vectors, regardless of magnitude."
    ]

    # Insert documents into database
    print("Inserting sample documents...")
    for i, doc in enumerate(sample_docs):
        # Generate embedding for document
        embedding = embedding_model.encode(doc).tolist()

        # Insert document with embedding and metadata
        metadata = {"source": f"sample_document_{i}", "category": "vector_db_info"}
        doc_id = db.insert_document(doc, embedding, metadata)
        print(f"Inserted document {doc_id}: {doc[:30]}...")

    # Count documents in database
    doc_count = db.count_documents()
    print(f"Total documents in database: {doc_count}")

    # Perform a similarity search
    query = "How do vector databases help with RAG applications?"
    print(f"\nSearching for documents similar to: '{query}'")

    # Generate embedding for query
    query_embedding = embedding_model.encode(query).tolist()

    # Search for similar documents
    start_time = time.time()
    results = db.search_similar(query_embedding, top_k=3)
    search_time = time.time() - start_time

    # Display results
    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Found {len(results)} similar documents:")

    for i, result in enumerate(results):
        print(f"\nResult {i+1} (similarity: {result['similarity']:.4f}):")
        print(f"ID: {result['id']}")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
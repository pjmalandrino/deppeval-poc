#!/usr/bin/env python
"""
Extremely minimal DeepEval 2.5.0 test for RAG application.
Focusing on bare minimum functionality to ensure compatibility.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import vector database class
sys.path.append(str(Path(__file__).parent))
from vector_db import VectorDB

# Import only the absolutely necessary DeepEval components
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def main():
    """Run an extremely minimal DeepEval RAG test."""
    print("\n==== Minimal DeepEval 2.5.0 RAG Test ====\n")

    # 1. Connect to vector database with minimal setup
    connection_string = "postgresql://postgres:postgres@localhost:5432/ragdb"
    if "DATABASE_URL" in os.environ:
        connection_string = os.environ["DATABASE_URL"]

    print("Connecting to vector database...")
    db = VectorDB(connection_string=connection_string, embedding_dim=384)

    # 2. Get document count
    doc_count = db.count_documents()
    print(f"Found {doc_count} documents in the database.")

    # 3. Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Simple query
    query = "What are vector databases?"
    print(f"Query: '{query}'")

    # 5. Get embedding and search
    query_embedding = embedding_model.encode(query).tolist()
    results = db.search_similar(query_embedding, top_k=2)

    # 6. Extract content for context
    retrieval_context = [result["content"] for result in results]
    print(f"Retrieved {len(retrieval_context)} documents")

    # 7. Sample answer (in real system, this would come from an LLM)
    answer = "Vector databases are specialized databases designed to store and search vector embeddings efficiently."

    # 8. Create test case with MINIMAL parameters (specifically for v2.5.0)
    print("\nCreating test case...")
    try:
        # The most minimal approach using v2.5.0 expected parameter names
        test_case = LLMTestCase(
            query=query,  # Using 'query' instead of 'input'
            output=answer,  # Using 'output' instead of 'actual_output'
            retrieval_context=retrieval_context
        )
        print("Successfully created test case!")
    except Exception as e:
        print(f"Error creating test case: {e}")
        return

    # 9. Run single metric
    print("\nRunning evaluation...")
    try:
        metric = AnswerRelevancyMetric(threshold=0.5)
        metric.measure(test_case)

        # Get results
        score = metric.score
        print(f"\nEvaluation complete!")
        print(f"Answer Relevancy Score: {score:.4f}")

        # Try to get explanation if available
        try:
            if hasattr(metric, 'reasoning'):
                print(f"Reasoning: {metric.reasoning}")
            elif hasattr(metric, 'reason'):
                print(f"Reason: {metric.reason}")
            elif hasattr(metric, 'explanation'):
                print(f"Explanation: {metric.explanation}")
        except Exception:
            pass

        # Show pass/fail
        if score >= 0.5:
            print("✅ PASSED")
        else:
            print("❌ FAILED")

    except Exception as e:
        print(f"Error during evaluation: {e}")

    print("\n==== Test Complete ====")

if __name__ == "__main__":
    main()
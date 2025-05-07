#!/usr/bin/env python
"""
Minimal DeepEval test for RAG application.
This is a standalone script that only requires deepeval and your existing vector_db.py
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import your vector database class
# Make sure the path to vector_db.py is correct
sys.path.append(str(Path(__file__).parent))  # Add current directory to path
from vector_db import VectorDB

# Import deepeval components
try:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    print("Successfully imported DeepEval!")
except ImportError as e:
    print(f"Error importing DeepEval: {e}")
    print("Please make sure DeepEval is installed: pip install 'deepeval>=2.0.0,<3.0.0'")
    sys.exit(1)

def main():
    """Run a simple DeepEval test on the RAG system."""
    print("\n==== Simple DeepEval RAG Test ====\n")

    # 1. Connect to the vector database
    connection_string = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"
    )

    # Initialize vector database
    print("Connecting to vector database...")
    db = VectorDB(
        connection_string=connection_string,
        embedding_dim=384  # Dimension for 'all-MiniLM-L6-v2'
    )

    # Check if there are documents in the database
    doc_count = db.count_documents()
    print(f"Found {doc_count} documents in the database.")

    if doc_count == 0:
        print("No documents found in the database. Please add some documents first.")
        return

    # 2. Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Define a test query
    query = "How do vector databases store embeddings efficiently?"
    print(f"\nTest query: '{query}'")

    # 4. Get query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # 5. Retrieve documents from vector database
    print("Retrieving similar documents...")
    results = db.search_similar(query_embedding, top_k=3)

    # 6. Create retrieval context from results
    retrieval_context = [result["content"] for result in results]

    # Print retrieved documents
    print(f"\nRetrieved {len(retrieval_context)} documents:")
    for i, doc in enumerate(retrieval_context):
        print(f"\nDocument {i+1} (first 100 chars):")
        print(f"{doc[:100]}...")

    # 7. For testing purposes, we'll use a predefined answer
    # In a real RAG system, this would come from your LLM
    expected_output = "Vector databases store embeddings efficiently using specialized data structures like HNSW or IVF indexes that enable similarity search operations."
    actual_output = expected_output  # For demonstration purposes

    # 8. Create DeepEval test case
    print("\nCreating DeepEval test case...")
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    print("Created LLM test case")

    # 9. Initialize metrics
    print("\nInitializing DeepEval metrics...")
    metrics = []

    try:
        answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
        metrics.append(answer_relevancy)
        print("Added AnswerRelevancyMetric")
    except Exception as e:
        print(f"Error initializing AnswerRelevancyMetric: {e}")

    try:
        faithfulness = FaithfulnessMetric(threshold=0.7)
        metrics.append(faithfulness)
        print("Added FaithfulnessMetric")
    except Exception as e:
        print(f"Error initializing FaithfulnessMetric: {e}")

    # 10. Run metrics
    print("\n==== Evaluation Results ====")

    for metric in metrics:
        try:
            print(f"\nRunning {metric.__class__.__name__}...")
            metric.measure(test_case)
            score = metric.score

            # Try to get explanation (newer versions use explanation, older use reason)
            explanation = None
            try:
                explanation = metric.explanation
            except AttributeError:
                try:
                    explanation = metric.reason
                except AttributeError:
                    explanation = "No explanation available"

            print(f"Score: {score:.4f}")
            print(f"Explanation: {explanation}")

            # Print pass/fail based on threshold
            threshold = getattr(metric, 'threshold', 0.5)
            if score >= threshold:
                print(f"✅ PASSED (threshold: {threshold:.2f})")
            else:
                print(f"❌ FAILED (threshold: {threshold:.2f})")

        except Exception as e:
            print(f"Error running {metric.__class__.__name__}: {str(e)}")

    print("\n==== Test Complete ====")

if __name__ == "__main__":
    main()
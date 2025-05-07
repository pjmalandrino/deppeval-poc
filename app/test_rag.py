#!/usr/bin/env python
"""
PyTest file for testing RAG application with DeepEval.
This allows you to run automated tests in a CI/CD pipeline.
"""

import os
import pytest
from sentence_transformers import SentenceTransformer
from vector_db import VectorDB

# Import DeepEval components
from deepeval.test_runner import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset


# Initialize embedding model and vector database for all tests
@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@pytest.fixture(scope="module")
def vector_db():
    connection_string = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"
    )
    return VectorDB(connection_string=connection_string, embedding_dim=384)


# Create a fixture for test data
@pytest.fixture(scope="module")
def test_dataset(embedding_model, vector_db):
    # Sample test cases
    sample_cases = [
        {
            "query": "How do vector databases store embeddings?",
            "expected_output": "Vector databases store embeddings efficiently using specialized data structures like HNSW or IVF that enable similarity search."
        },
        {
            "query": "What is the purpose of pgvector extension?",
            "expected_output": "The pgvector extension adds vector similarity search capabilities to PostgreSQL databases."
        },
        {
            "query": "How does RAG enhance language models?",
            "expected_output": "RAG enhances language models by providing external knowledge retrieval capabilities without requiring model retraining."
        }
    ]

    # Create test cases with actual retrieval context
    test_cases = []
    for case in sample_cases:
        query = case["query"]
        expected = case["expected_output"]

        # Get embeddings for query
        query_embedding = embedding_model.encode(query).tolist()

        # Search for relevant documents
        results = vector_db.search_similar(query_embedding, top_k=3)
        retrieval_context = [result["content"] for result in results]

        # In a real test, you would call your RAG application to get actual_output
        # For demonstration, we'll use expected_output as a placeholder
        # actual_output = your_rag_application(query)
        actual_output = expected  # Placeholder

        test_case_params = LLMTestCaseParams(
            input=query,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context
        )
        test_case = LLMTestCase(**test_case_params.dict())
        test_cases.append(test_case)

    return EvaluationDataset(test_cases=test_cases)


# Define metrics as fixtures
@pytest.fixture
def relevancy_metric():
    return AnswerRelevancyMetric(threshold=0.7)


@pytest.fixture
def faithfulness_metric():
    return FaithfulnessMetric(threshold=0.7)


@pytest.fixture
def contextual_relevancy_metric():
    return ContextualRelevancyMetric(threshold=0.7)


# Parametrized test to run for each test case
@pytest.mark.parametrize("test_case_index", [0, 1, 2])
def test_answer_relevancy(test_dataset, relevancy_metric, test_case_index):
    """Test if answers are relevant to the query."""
    test_case = test_dataset.test_cases[test_case_index]
    assert_test(test_case, [relevancy_metric])


@pytest.mark.parametrize("test_case_index", [0, 1, 2])
def test_faithfulness(test_dataset, faithfulness_metric, test_case_index):
    """Test if answers are faithful to the retrieval context."""
    test_case = test_dataset.test_cases[test_case_index]
    assert_test(test_case, [faithfulness_metric])


@pytest.mark.parametrize("test_case_index", [0, 1, 2])
def test_contextual_relevancy(test_dataset, contextual_relevancy_metric, test_case_index):
    """Test if retrieved context is relevant to the query."""
    test_case = test_dataset.test_cases[test_case_index]
    assert_test(test_case, [contextual_relevancy_metric])


# Test that runs all metrics at once
@pytest.mark.parametrize("test_case_index", [0, 1, 2])
def test_all_metrics(
        test_dataset,
        relevancy_metric,
        faithfulness_metric,
        contextual_relevancy_metric,
        test_case_index
):
    """Test all metrics at once."""
    test_case = test_dataset.test_cases[test_case_index]
    all_metrics = [
        relevancy_metric,
        faithfulness_metric,
        contextual_relevancy_metric
    ]
    assert_test(test_case, all_metrics)


# Custom test for your specific RAG implementation
def test_custom_rag_implementation(embedding_model, vector_db):
    """Custom test for your specific RAG implementation."""
    # Replace this with your actual RAG application logic
    query = "What is the advantage of using PostgreSQL with pgvector?"

    # 1. Get query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # 2. Retrieve relevant documents
    results = vector_db.search_similar(query_embedding, top_k=3)
    retrieval_context = [result["content"] for result in results]

    # 3. In a real test, generate answer using LLM
    # actual_output = your_llm_function(query, retrieval_context)
    # For demonstration, use a placeholder
    actual_output = "PostgreSQL with pgvector offers efficient vector similarity search while maintaining the benefits of a robust, mature relational database system."

    # 4. Define expected output
    expected_output = "PostgreSQL with pgvector combines vector similarity search with the features of a full relational database."

    # 5. Create test case
    test_case_params = LLMTestCaseParams(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    test_case = LLMTestCase(**test_case_params.dict())

    # 6. Define and run metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7)
    ]

    assert_test(test_case, metrics)
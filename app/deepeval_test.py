#!/usr/bin/env python
"""
DeepEval test for RAG application using Ollama instead of OpenAI.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from vector_db import VectorDB

# Import deepeval components
try:
    from deepeval.metrics import ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics.base_metric import BaseMetric
    print("Successfully imported DeepEval!")
except ImportError as e:
    print(f"Error importing DeepEval: {e}")
    print("Please make sure DeepEval is installed: pip install 'deepeval>=2.5.0'")
    sys.exit(1)

# Custom metric class using Ollama instead of OpenAI
class OllamaFaithfulnessMetric(BaseMetric):
    def __init__(self, threshold=0.5, model="gemma3:4b"):
        # Initialize the base class correctly
        super().__init__()
        # Set properties after base initialization
        self.threshold = threshold
        self.name = "Faithfulness Metric (Ollama)"
        self.model = model
        self.score = 0.0
        self.explanation = ""

    def measure(self, test_case):
        """Measure faithfulness using Ollama."""
        # Extract data from test case
        query = test_case.input
        actual_output = test_case.actual_output
        context = "\n\n".join(test_case.retrieval_context)

        # Construct prompt for Ollama
        prompt = f"""
        You are an evaluator measuring if an answer is faithful to the provided context.
        
        Context:
        {context}
        
        Query: {query}
        
        Answer: {actual_output}
        
        On a scale of 0.0 to 1.0, how faithful is the answer to the given context?
        A score of 1.0 means the answer is completely supported by the context.
        A score of 0.0 means the answer has significant hallucinations or makes claims not in the context.
        
        First, explain your reasoning in detail.
        Then on the final line, provide only a score in this exact format: "SCORE: 0.X"
        """

        # Call Ollama API
        try:
            response = self._call_ollama(prompt)
            # Extract score and explanation
            self._parse_response(response)
            return self.score >= self.threshold
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            self.explanation = f"Error: {str(e)}"
            self.score = 0.0
            return False

    def _call_ollama(self, prompt):
        """Call Ollama API with the given prompt."""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        return response.json()["response"]

    def _parse_response(self, response):
        """Parse the response to extract score and explanation."""
        # Find the score line (should be last line or somewhere in the text)
        lines = response.strip().split('\n')
        score_line = None
        score = 0.0

        # First try to find a line with "SCORE: X.X"
        for line in reversed(lines):
            if "SCORE:" in line:
                score_line = line
                try:
                    score = float(line.split("SCORE:")[1].strip())
                    break
                except (ValueError, IndexError):
                    pass

        # If we found a score, use it
        if score_line and score > 0:
            self.score = score
            # Try to get the explanation (everything before the score line)
            try:
                self.explanation = "\n".join(lines[:lines.index(score_line)])
            except (ValueError, IndexError):
                self.explanation = "Explanation extraction failed, but score was found."
            return

        # If no explicit score found, try to find a number between 0 and 1
        import re
        for line in reversed(lines):
            # Look for patterns like "0.8" or "0.75" or "score is 0.9" etc.
            score_matches = re.findall(r'(?:score\s*(?:is|of|:)?\s*)?(\d?\.\d+)', line.lower())
            for match in score_matches:
                try:
                    potential_score = float(match)
                    if 0 <= potential_score <= 1:
                        self.score = potential_score
                        self.explanation = f"Score extracted from text: '{line}'. Full response:\n{response}"
                        return
                except ValueError:
                    pass

        # If all else fails, try to interpret the whole response
        if "high" in response.lower() or "good" in response.lower() or "strong" in response.lower():
            self.score = 0.8
            self.explanation = f"Score inferred from positive language. Full response:\n{response}"
        elif "moderate" in response.lower() or "partial" in response.lower() or "somewhat" in response.lower():
            self.score = 0.5
            self.explanation = f"Score inferred from moderate language. Full response:\n{response}"
        elif "low" in response.lower() or "poor" in response.lower() or "weak" in response.lower():
            self.score = 0.2
            self.explanation = f"Score inferred from negative language. Full response:\n{response}"
        else:
            # Default fallback
            self.score = 0.5
            self.explanation = f"Could not extract score. Using default 0.5. Full response:\n{response}"

# Same as above but for answer relevancy
class OllamaAnswerRelevancyMetric(BaseMetric):
    def __init__(self, threshold=0.5, model="gemma3:4b"):
        # Initialize the base class correctly
        super().__init__()
        # Set properties after base initialization
        self.threshold = threshold
        self.name = "Answer Relevancy Metric (Ollama)"
        self.model = model
        self.score = 0.0
        self.explanation = ""

    def measure(self, test_case):
        """Measure answer relevancy using Ollama."""
        # Extract data from test case
        query = test_case.input
        actual_output = test_case.actual_output

        # Construct prompt for Ollama
        prompt = f"""
        You are an evaluator measuring if an answer is relevant to the given query.
        
        Query: {query}
        
        Answer: {actual_output}
        
        On a scale of 0.0 to 1.0, how relevant is this answer to the query?
        A score of 1.0 means the answer directly and completely addresses the query.
        A score of 0.0 means the answer is completely irrelevant to the query.
        
        First, explain your reasoning in detail.
        Then on the final line, provide only a score in this exact format: "SCORE: 0.X"
        """

        # Call Ollama API
        try:
            response = self._call_ollama(prompt)
            # Extract score and explanation
            self._parse_response(response)
            return self.score >= self.threshold
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            self.explanation = f"Error: {str(e)}"
            self.score = 0.0
            return False

    def _call_ollama(self, prompt):
        """Call Ollama API with the given prompt."""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        return response.json()["response"]

    def _parse_response(self, response):
        """Parse the response to extract score and explanation."""
        # Find the score line (should be last line or somewhere in the text)
        lines = response.strip().split('\n')
        score_line = None
        score = 0.0

        # First try to find a line with "SCORE: X.X"
        for line in reversed(lines):
            if "SCORE:" in line:
                score_line = line
                try:
                    score = float(line.split("SCORE:")[1].strip())
                    break
                except (ValueError, IndexError):
                    pass

        # If we found a score, use it
        if score_line and score > 0:
            self.score = score
            # Try to get the explanation (everything before the score line)
            try:
                self.explanation = "\n".join(lines[:lines.index(score_line)])
            except (ValueError, IndexError):
                self.explanation = "Explanation extraction failed, but score was found."
            return

        # If no explicit score found, try to find a number between 0 and 1
        import re
        for line in reversed(lines):
            # Look for patterns like "0.8" or "0.75" or "score is 0.9" etc.
            score_matches = re.findall(r'(?:score\s*(?:is|of|:)?\s*)?(\d?\.\d+)', line.lower())
            for match in score_matches:
                try:
                    potential_score = float(match)
                    if 0 <= potential_score <= 1:
                        self.score = potential_score
                        self.explanation = f"Score extracted from text: '{line}'. Full response:\n{response}"
                        return
                except ValueError:
                    pass

        # If all else fails, try to interpret the whole response
        if "high" in response.lower() or "good" in response.lower() or "strong" in response.lower():
            self.score = 0.8
            self.explanation = f"Score inferred from positive language. Full response:\n{response}"
        elif "moderate" in response.lower() or "partial" in response.lower() or "somewhat" in response.lower():
            self.score = 0.5
            self.explanation = f"Score inferred from moderate language. Full response:\n{response}"
        elif "low" in response.lower() or "poor" in response.lower() or "weak" in response.lower():
            self.score = 0.2
            self.explanation = f"Score inferred from negative language. Full response:\n{response}"
        else:
            # Default fallback
            self.score = 0.5
            self.explanation = f"Could not extract score. Using default 0.5. Full response:\n{response}"

def main():
    """Run a DeepEval test on the RAG system using Ollama."""
    print("\n==== DeepEval RAG Test with Ollama ====\n")

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

    # 9. Initialize metrics using Ollama
    print("\nInitializing DeepEval metrics with Ollama...")
    metrics = []

    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama server is not running")

        # Check if model is available - the API structure may vary by Ollama version
        try:
            # Try different possible structures of the response
            if "models" in response.json():
                # Newer Ollama versions
                models = [model["name"] for model in response.json()["models"]]
                if "gemma3:4b" not in models:
                    print("Warning: gemma3:4b model not found in Ollama models list.")
                    print("Will try to use it anyway. If this fails, please run: ollama pull gemma3:4b")
            else:
                # Older Ollama versions might have a different structure
                models = [model["name"] for model in response.json().get("models", [])]
                tags = [tag["name"] for tag in response.json().get("models", [])]

                if "gemma3:4b" not in models and "gemma3:4b" not in tags:
                    print("Warning: gemma3:4b model not found in Ollama. Will try to use it anyway.")
                    print("If this fails, please run: ollama pull gemma3:4b")
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not verify if model exists: {e}")
            print("Will attempt to use gemma3:4b anyway. If this fails, please run: ollama pull gemma3:4b")

        # Add custom Ollama metrics
        answer_relevancy = OllamaAnswerRelevancyMetric(threshold=0.7)
        metrics.append(answer_relevancy)
        print("Added OllamaAnswerRelevancyMetric")

        faithfulness = OllamaFaithfulnessMetric(threshold=0.7)
        metrics.append(faithfulness)
        print("Added OllamaFaithfulnessMetric")

    except Exception as e:
        print(f"Error initializing Ollama metrics: {e}")
        print("Please make sure Ollama is running with the gemma3:4b model.")
        print("You can install it with: ollama pull gemma3:4b")
        return

    # 10. Run metrics
    print("\n==== Evaluation Results ====")

    for metric in metrics:
        try:
            print(f"\nRunning {metric.name}...")
            metric.measure(test_case)
            score = metric.score

            print(f"Score: {score:.4f}")
            print(f"Explanation: {metric.explanation}")

            # Print pass/fail based on threshold
            threshold = getattr(metric, 'threshold', 0.5)
            if score >= threshold:
                print(f"✅ PASSED (threshold: {threshold:.2f})")
            else:
                print(f"❌ FAILED (threshold: {threshold:.2f})")

        except Exception as e:
            print(f"Error running {metric.name}: {str(e)}")

    print("\n==== Test Complete ====")

if __name__ == "__main__":
    main()
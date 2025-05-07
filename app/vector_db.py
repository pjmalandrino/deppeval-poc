import os
import psycopg2
from psycopg2.extras import execute_values
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VectorDB:
    """Class to handle basic operations with pgvector database."""

    def __init__(
            self,
            connection_string: str = None,
            table_name: str = "documents",
            embedding_dim: int = 384
    ):
        """Initialize the VectorDB instance.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store documents and embeddings
            embedding_dim: Dimension of the embedding vectors
        """
        self.connection_string = connection_string or os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"
        )
        self.table_name = table_name
        self.embedding_dim = embedding_dim

        # Initialize the database on startup
        self.init_db()

    def init_db(self) -> None:
        """Initialize the database with pgvector extension and required tables."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Create pgvector extension if it doesn't exist
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table for storing documents and embeddings
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding vector({self.embedding_dim})
            )
            """)

            # Create an index for faster vector search (optional but recommended)
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} USING ivfflat (embedding vector_l2_ops) 
            WITH (lists = 100)
            """)

            conn.commit()
            print(f"Database initialized with table '{self.table_name}' and vector index")
        except Exception as e:
            print(f"Error initializing database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def insert_document(
            self,
            content: str,
            embedding: List[float],
            metadata: Dict[str, Any] = None
    ) -> int:
        """Insert a document with its embedding into the database.

        Args:
            content: The text content of the document
            embedding: Vector embedding of the document
            metadata: Additional metadata about the document

        Returns:
            The ID of the inserted document
        """
        conn = None
        doc_id = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            metadata = metadata or {}

            cursor.execute(
                f"""
                INSERT INTO {self.table_name} (content, metadata, embedding)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (content, json.dumps(metadata), embedding)
            )

            doc_id = cursor.fetchone()[0]
            conn.commit()
            return doc_id
        except Exception as e:
            print(f"Error inserting document: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()

    def insert_documents(
            self,
            documents: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> List[int]:
        """Batch insert multiple documents with their embeddings.

        Args:
            documents: List of tuples (content, embedding, metadata)

        Returns:
            List of document IDs that were inserted
        """
        conn = None
        doc_ids = []
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Prepare data for batch insertion
            values = []
            for content, embedding, metadata in documents:
                metadata = metadata or {}
                values.append((content, json.dumps(metadata), embedding))

            # Use executemany for simpler batch insertion
            cursor.executemany(
                f"""
                INSERT INTO {self.table_name} (content, metadata, embedding)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                values
            )

            doc_ids = [row[0] for row in cursor.fetchall()]
            conn.commit()
            return doc_ids
        except Exception as e:
            print(f"Error batch inserting documents: {e}")
            if conn:
                conn.rollback()
            return []
        finally:
            if conn:
                conn.close()

    def search_similar(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            similarity_metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """Search for documents similar to the query embedding.

        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of most similar documents to return
            similarity_metric: Type of similarity metric to use ('cosine', 'l2', 'inner')

        Returns:
            List of dictionaries containing document information and similarity score
        """
        conn = None
        try:
            # Ensure query_embedding is a list of floats
            if not all(isinstance(x, (float, int)) for x in query_embedding):
                query_embedding = [float(x) for x in query_embedding]

            # Ensure embedding has correct dimension
            if len(query_embedding) != self.embedding_dim:
                raise ValueError(f"Query embedding dimension ({len(query_embedding)}) does not match expected dimension ({self.embedding_dim})")

            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Select similarity operator based on metric
            if similarity_metric == "cosine":
                # Cosine similarity: 1 - (vector <=> vector)
                similarity_op = "1 - (embedding <=> %s::vector)"
            elif similarity_metric == "l2":
                # Negative L2 distance (closer to 0 is better)
                similarity_op = "-1 * (embedding <-> %s::vector)"
            elif similarity_metric == "inner":
                # Inner product
                similarity_op = "embedding <#> %s::vector"
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

            # Format query embedding for pgvector
            vector_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Execute the query with proper error handling
            query = f"""
                SELECT id, content, metadata, {similarity_op} as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
            """

            cursor.execute(query, (vector_str, top_k))

            results = []
            for doc_id, content, metadata_json, similarity in cursor.fetchall():
                # Handle metadata parsing more safely
                try:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                except (json.JSONDecodeError, TypeError):
                    # Default to empty dict if parsing fails
                    metadata = {}

                results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity": similarity
                })

            return results
        except Exception as e:
            print(f"Error searching similar documents: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID.

        Args:
            doc_id: ID of the document to retrieve

        Returns:
            Dictionary containing document information or None if not found
        """
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT id, content, metadata, embedding
                FROM {self.table_name}
                WHERE id = %s
                """,
                (doc_id,)
            )

            result = cursor.fetchone()
            if not result:
                return None

            doc_id, content, metadata_json, embedding = result
            try:
                if isinstance(metadata_json, str):
                    metadata = json.loads(metadata_json)
                else:
                    metadata = metadata_json
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            return {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            }
        except Exception as e:
            print(f"Error getting document: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def delete_document(self, doc_id: int) -> bool:
        """Delete a specific document by ID.

        Args:
            doc_id: ID of the document to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE id = %s
                """,
                (doc_id,)
            )

            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting document: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def count_documents(self) -> int:
        """Count the number of documents in the database.

        Returns:
            The number of documents
        """
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            print(f"Error counting documents: {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def clear_table(self) -> bool:
        """Clear all data from the table.

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing table: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
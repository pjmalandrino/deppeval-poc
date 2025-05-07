#!/usr/bin/env python
"""
Simple Gradio interface for RAG application using pgvector database.
Allows uploading documents, asking questions, and viewing similar documents.
"""

import os
import tempfile
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db import VectorDB
import PyPDF2
from pathlib import Path
import time

class RAGApp:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize vector database
        self.db = VectorDB(
            connection_string=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb"),
            embedding_dim=384  # Dimension for 'all-MiniLM-L6-v2'
        )

        # Create Gradio interface
        self.create_interface()

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            return text.strip()
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def process_file(self, file_obj):
        """Process uploaded file (text or PDF) and store in vector DB."""
        if file_obj is None:
            return "No file uploaded."

        try:
            file_path = file_obj.name
            file_name = Path(file_path).name
            file_extension = Path(file_path).suffix.lower()

            # Extract text based on file type
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension in ['.txt', '.md', '.csv']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return f"Unsupported file type: {file_extension}. Please upload .txt, .pdf, .md, or .csv files."

            if not text or len(text.strip()) < 10:
                return "Extracted text is too short or empty."

            # Create metadata
            metadata = {
                "filename": file_name,
                "file_type": file_extension[1:],  # Remove the dot
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": os.path.getsize(file_path)
            }

            # Split text into chunks (simple approach - max 1000 chars per chunk)
            chunks = []
            max_chunk_size = 1000
            for i in range(0, len(text), max_chunk_size):
                chunk = text[i:i+max_chunk_size]
                if len(chunk.strip()) > 50:  # Only add meaningful chunks
                    chunks.append(chunk)

            # Generate embeddings and store each chunk
            result = f"Processing {file_name} into {len(chunks)} chunks:\n"
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = self.embedding_model.encode(chunk).tolist()

                # Add chunk index to metadata
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                # Insert document with embedding and metadata
                doc_id = self.db.insert_document(chunk, embedding, chunk_metadata)
                result += f"- Chunk {i+1}/{len(chunks)}: Document ID {doc_id}\n"

            return result + f"\nSuccessfully processed and stored {file_name}."

        except Exception as e:
            return f"Error processing file: {str(e)}"

    def query_documents(self, query_text, top_k=5):
        """Query vector DB for documents similar to the query."""
        if not query_text or len(query_text.strip()) < 3:
            return "Query text is too short. Please enter a more specific query."

        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Ensure the embedding is in the correct format - cast to float
            query_embedding_float = [float(x) for x in query_embedding]

            # Search for similar documents
            start_time = time.time()
            results = self.db.search_similar(query_embedding_float, top_k=top_k)
            search_time = time.time() - start_time

            if not results:
                return "No relevant documents found for your query."

            # Format results
            output = f"Found {len(results)} relevant documents in {search_time:.4f} seconds:\n\n"

            for i, result in enumerate(results):
                output += f"--- Result {i+1} (Similarity: {result['similarity']:.4f}) ---\n"
                output += f"From: {result['metadata'].get('filename', 'Unknown')}\n"
                output += f"Content: {result['content'][:200]}...\n\n"

            return output

        except Exception as e:
            return f"Error querying documents: {str(e)}"

    def get_db_stats(self):
        """Get database statistics."""
        try:
            doc_count = self.db.count_documents()
            return f"Vector database contains {doc_count} document chunks."
        except Exception as e:
            return f"Error getting database stats: {str(e)}"

    def clear_database(self):
        """Clear all documents from the database."""
        try:
            success = self.db.clear_table()
            if success:
                return "Database cleared successfully."
            else:
                return "Failed to clear database."
        except Exception as e:
            return f"Error clearing database: {str(e)}"

    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="RAG Vector Database App") as self.interface:
            gr.Markdown("# 🔍 RAG Vector Database App")
            gr.Markdown("Upload documents, ask questions, and find similar content using vector similarity search.")

            with gr.Tab("Upload Documents"):
                file_input = gr.File(label="Upload Document (PDF, TXT, MD, CSV)")
                upload_button = gr.Button("Process Document")
                upload_output = gr.Textbox(label="Upload Result", lines=10)
                db_stats = gr.Textbox(label="Database Stats")
                refresh_button = gr.Button("Refresh DB Stats")
                clear_button = gr.Button("Clear Database", variant="stop")

                upload_button.click(self.process_file, inputs=[file_input], outputs=[upload_output])
                refresh_button.click(self.get_db_stats, inputs=[], outputs=[db_stats])
                clear_button.click(self.clear_database, inputs=[], outputs=[db_stats])

            with gr.Tab("Query Documents"):
                query_input = gr.Textbox(label="Ask a question about your documents", lines=3)
                k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
                query_button = gr.Button("Search")
                query_output = gr.Textbox(label="Search Results", lines=20)

                query_button.click(self.query_documents, inputs=[query_input, k_slider], outputs=[query_output])

            # Initialize database stats on load
            self.interface.load(self.get_db_stats, inputs=[], outputs=[db_stats])

    def launch(self, share=False):
        """Launch the Gradio interface."""
        self.interface.launch(share=share)

if __name__ == "__main__":
    app = RAGApp()
    app.launch()
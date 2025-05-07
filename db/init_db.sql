-- Initialize pgvector database for RAG application

-- Create pgvector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table with vector support
CREATE TABLE IF NOT EXISTS documents (
                                         id SERIAL PRIMARY KEY,
                                         content TEXT NOT NULL,
                                         metadata JSONB NOT NULL,
                                         embedding vector(384)  -- Dimension for 'all-MiniLM-L6-v2' model
    );

-- Create index for faster vector search using IVFFlat
-- IVFFlat divides vectors into lists for faster search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);  -- Number of lists affects performance/accuracy tradeoff

-- Create index for metadata to speed up filtering
CREATE INDEX IF NOT EXISTS documents_metadata_idx
    ON documents USING GIN (metadata);

-- Optional: Create table for storing the queries and their results
CREATE TABLE IF NOT EXISTS query_log (
                                         id SERIAL PRIMARY KEY,
                                         query TEXT NOT NULL,
                                         embedding vector(384),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB DEFAULT '{}'::jsonb
                            );

-- Optional: Create table for storing collections/namespaces
CREATE TABLE IF NOT EXISTS collections (
                                           id SERIAL PRIMARY KEY,
                                           name TEXT UNIQUE NOT NULL,
                                           description TEXT,
                                           created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                           metadata JSONB DEFAULT '{}'::jsonb
);

-- Optional: Add collection_id to documents for organization
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS collection_id INTEGER REFERENCES collections(id) ON DELETE SET NULL;

-- Create function to calculate cosine similarity
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS float
LANGUAGE plpgsql
AS $$
BEGIN
    -- Ensure vectors are normalized
RETURN 1 - (a <=> b);
END;
$$;

-- Create function to calculate euclidean distance
CREATE OR REPLACE FUNCTION euclidean_distance(a vector, b vector)
RETURNS float
LANGUAGE plpgsql
AS $$
BEGIN
RETURN a <-> b;
END;
$$;

-- Create function to calculate dot product (inner product)
CREATE OR REPLACE FUNCTION inner_product(a vector, b vector)
RETURNS float
LANGUAGE plpgsql
AS $$
BEGIN
RETURN a <#> b;
END;
$$;

-- Example of a view that might be useful for querying
CREATE OR REPLACE VIEW document_stats AS
SELECT
    COALESCE(c.name, 'Uncategorized') AS collection,
    COUNT(d.id) AS document_count,
    MIN(LENGTH(d.content)) AS min_content_length,
    MAX(LENGTH(d.content)) AS max_content_length,
    AVG(LENGTH(d.content)) AS avg_content_length
FROM
    documents d
        LEFT JOIN
    collections c ON d.collection_id = c.id
GROUP BY
    COALESCE(c.name, 'Uncategorized');

-- Add some helpful comments to explain usage
COMMENT ON TABLE documents IS 'Stores document chunks with their vector embeddings';
COMMENT ON COLUMN documents.embedding IS 'Vector representation of document content';
COMMENT ON TABLE collections IS 'Optional grouping of documents into collections';
COMMENT ON TABLE query_log IS 'Logs of queries for analytics and optimization';
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for storing embeddings
CREATE TABLE IF NOT EXISTS text_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for similarity search
CREATE INDEX IF NOT EXISTS text_embeddings_embedding_idx ON text_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

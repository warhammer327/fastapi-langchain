import asyncio
import os
from typing import List, Optional

import numpy as np
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from langchain_community.llms import Ollama
from nomic import embed
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "embeddings")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


Base.metadata.create_all(bind=engine)


class TextInput(BaseModel):
    content: str


class EmbeddingRespose(BaseModel):
    id: int
    content: str
    created_at: str


class SimilarityQuery(BaseModel):
    query: str
    limit: Optional[int] = 5


app = FastAPI(title="Tragic End")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def process_generation(model_name, prompt, max_tokens, callback):
    try:
        # Initialize the Ollama model with langchain
        llm = Ollama(model=model_name, base_url=OLLAMA_HOST)

        # Generate the response with a timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, prompt, max_tokens=max_tokens),
            timeout=30,  # 30 second timeout, adjust as needed
        )

        # Call the callback with success result
        callback(True, response)
    except Exception as e:
        # Call the callback with error
        callback(False, str(e))


# Add a dictionary to store ongoing requests
generation_tasks = {}


@app.get("/")
def read_root():
    return {"message": "test check check"}


@app.get("/models-list")
def list_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/generate")
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        model = data.get("model", "gemma:latest")
        prompt = data.get("prompt", "hello")
        max_tokens = data.get("max_tokens", 100)

        # Generate a unique task ID
        task_id = str(hash(f"{model}_{prompt}_{max_tokens}_{len(generation_tasks)}"))

        # Create a structure to store the result
        task_result = {"status": "processing", "result": None}
        generation_tasks[task_id] = task_result

        # Define the callback function
        def update_task_result(success, result):
            if success:
                task_result["status"] = "completed"
                task_result["result"] = result
            else:
                task_result["status"] = "failed"
                task_result["error"] = result

        # Start the generation in the background
        background_tasks.add_task(
            process_generation, model, prompt, max_tokens, update_task_result
        )

        # Return the task ID immediately
        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting generation: {str(e)}"
        )


@app.get("/generate/{task_id}")
async def get_generation_result(task_id: str):
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_result = generation_tasks[task_id]

    if task_result["status"] == "processing":
        return {"status": "processing"}
    elif task_result["status"] == "completed":
        # Clean up completed tasks after a delay
        # (In a production app, you'd want to use a proper cache with expiration)
        return {
            "status": "completed",
            "model": task_id.split("_")[0],  # This is a simplification
            "generated_text": task_result["result"],
        }
    else:
        return {"status": "failed", "error": task_result.get("error", "Unknown error")}


@app.post("/embed", response_model=EmbeddingRespose)
async def create_embedding(text_input: TextInput, db: Session = Depends(get_db)):
    try:
        content = text_input.content
        ollama_url = f"{OLLAMA_HOST}/api/embeddings"
        payload = {"model": "nomic-embed-text:latest", "prompt": content}
        response = requests.post(ollama_url, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama embeddings error: {response.text}",
            )

        # Extract the embedding vector from the response
        embedding_data = response.json()
        embedding_vector = embedding_data.get("embedding", [])

        # Create a new TextEmbedding object
        db_embedding = TextEmbedding(content=content, embedding=embedding_vector)

        # Save to database
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)

        # Return the created embedding info - matching your EmbeddingRespose model
        return {
            "id": db_embedding.id,
            "content": db_embedding.content,
            "created_at": str(
                db_embedding.created_at
            ),  # Convert to string to avoid serialization issues
        }
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating embedding: {str(e)}"
        )

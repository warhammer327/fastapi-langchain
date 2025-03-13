import os
from typing import List, Optional

import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException
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
    embedding = Column(Vector(768), nullable=False)
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


@app.get("/")
def read_root():
    return {"message": "check test check check"}


@app.get("/models-list")
def list_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/embed")
def create_embedding(text_input: TextInput, db: Session = Depends(get_db)):
    try:
        print(text_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding: {str(e)}")

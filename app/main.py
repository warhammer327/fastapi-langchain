import requests
from fastapi import FastAPI, HTTPException
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
app = FastAPI(title="Tragic End")


@app.get("/")
def read_root():
    return {"message": "check check check"}

@app.get("/models-list")
def list_models():
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

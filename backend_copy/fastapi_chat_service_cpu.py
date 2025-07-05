"""fastapi_chat_service_cpu.py – TutorAI FastAPI micro‑service (CPU‑only, TinyLlama)

Key differences from the GPU/Ollama version
-------------------------------------------
1. **Self‑contained pipeline** – we build the `chat_engine` right here using the
   same logic as the Flask CPU app (TinyLlama‑1.1B‑Chat v0.4 + MiniLM embeddings).
2. **Runs on plain CPU** – `device_map="cpu"` so no CUDA dependency.
3. **Streaming & non‑streaming endpoints** remain identical, so front‑end code
   doesn’t change.

Run locally:
    export TUTORAI_API_TOKEN="devtoken"
    uvicorn fastapi_chat_service_cpu:app --host 0.0.0.0 --port 8006 --reload
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports – standard library & FastAPI
# ──────────────────────────────────────────────────────────────────────────────
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# 2. Llama‑Index + HF setup (CPU‑friendly)
# ──────────────────────────────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.huggingface.base import HuggingFaceLLM
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding

from transformers import AutoTokenizer
from llm_config import get_llm, get_embedding_model, get_chat_engine


chat_engine = get_chat_engine()
llm = get_llm()
embedding_model = get_embedding_model()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Auth setup – load token from env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_TOKEN = os.getenv("TUTORAI_API_TOKEN")
if not API_TOKEN:
    print("[WARN] TUTORAI_API_TOKEN not set – all requests will 401.")

async def verify_bearer_token(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed auth header")
    if authorization.removeprefix("Bearer ").strip() != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# ──────────────────────────────────────────────────────────────────────────────
# 4. FastAPI app + schemas
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="TutorAI Chat API (CPU)")

class TutorRequest(BaseModel):
    query: str

class TutorResponse(BaseModel):
    answer: str
    done: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# 5. Non‑streaming endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=TutorResponse)
async def chat(req: TutorRequest) -> TutorResponse:
    try:
        # Check if the query contains "done"
        if "done" in req.query.lower():
            return TutorResponse(answer="Session completed.")
        
        resp = chat_engine.chat(req.query)
        # resp = "Hello"
        if hasattr(resp, "response"):
            answer = str(resp.response)
        else:
            answer = str(resp)
        return TutorResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc

if __name__ == "__main__":
    uvicorn.run("fastapi_chat_service_cpu:app", host="0.0.0.0", port=8006, reload=True)
    print("Ready for requests")
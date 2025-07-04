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

# ---------- 2.1  TinyLlama model & tokenizer on CPU ----------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer first so we can patch a minimal chat template if missing
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if _tokenizer.chat_template is None:
    _tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: "
    )

llm = HuggingFaceLLM(
    model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    tokenizer_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    context_window=1024,
    max_new_tokens=256,
    device_map="auto"
)

embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 256

# ---------- 2.2  Load vector index (assumes ./storage built by create_vector_index_cpu.py) ----------
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ---------- 2.3  Chat engine + memory ----------
chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(token_limit=256,
                                        chat_store=chat_store, 
                                        chat_store_key="user1")

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    similarity_top_k=3,
    llm=llm,
    context_prompt=(
        "Answer only in German. "
        "You are a German chatbot who helps with TUB modules and technical information. "
        "Relevant documents:\n{context_str}\n"
    ),
    verbose=False,
)

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
    client_id: str
    conversation_id: str
    query: str

class TutorResponse(BaseModel):
    answer: str
    done: bool = True

# ──────────────────────────────────────────────────────────────────────────────
# 5. Non‑streaming endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=TutorResponse, dependencies=[Depends(verify_bearer_token)])
async def chat(req: TutorRequest) -> TutorResponse:
    try:
        resp = chat_engine.chat(req.query)
        if hasattr(resp, "response"):
            answer = str(resp.response)
        else:
            answer = str(resp)
        return TutorResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc
    return TutorResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run("fastapi_chat_service_cpu:app", host="0.0.0.0", port=8006, reload=True)
    print("Ready for requests")
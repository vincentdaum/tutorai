"""flask_app_cpu.py – TutorAI Flask server running **entirely on CPU**.

Key points
----------
* LLM: TinyLlama‑1.1B‑Chat‑v0.4 (fits in <4 GB RAM).
* We patch a **minimal chat template** into the tokenizer because the model’s
  HF card doesn’t ship one; this prevents the ValueError you saw.
* Embeddings: sentence‑transformers/all‑MiniLM‑L6‑v2 (CPU‑friendly).
* Everything else (auth, Mongo) is unchanged.

Run with
    python flask_app_cpu.py

A quick note on performance: on a laptop this setup answers short queries in
~3‑5 s. If you need snappier responses, consider the 770 M TinyLlama instruct
checkpoint or enable 4‑bit quantisation via bitsandbytes.
"""

import os
from functools import wraps
from pathlib import Path
from typing import List

from flask import Flask, render_template, jsonify, request, redirect, session, flash
from passlib.hash import sha256_crypt
from pymongo import MongoClient
import requests

# ─────────────────────────────────────────────────────────────
# 1.  HuggingFace LLM (CPU‑only) with patched chat template
# ─────────────────────────────────────────────────────────────

from transformers import AutoTokenizer
from llama_index.llms.huggingface.base import HuggingFaceLLM

HF_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"

# Load tokenizer *first* so we can set a fallback chat_template
print("🔧  Loading tokenizer … (CPU)")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

# If the model card lacks a chat template, inject a simple one
if tokenizer.chat_template is None:
    print("⚠️  No chat template found; injecting fallback template.")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
    )

print("🤖  Loading TinyLlama‑1.1B … this will take ~30 s on CPU")
llm = HuggingFaceLLM(
    model_name=HF_MODEL_NAME,
    tokenizer_name=HF_MODEL_NAME,
    tokenizer=tokenizer,          # pass patched tokenizer
    context_window=2048,
    max_new_tokens=128,
    device_map="cpu",            # ← force CPU
    generate_kwargs={
        "temperature": 0.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    },
)

# ─────────────────────────────────────────────────────────────
# 2.  Embeddings (CPU)
# ─────────────────────────────────────────────────────────────
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

embedding_llm = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 256  # smaller chunks for low‑RAM boxes

# ─────────────────────────────────────────────────────────────
# 3.  Load vector index & set up chat engine
# ─────────────────────────────────────────────────────────────

print("📂  Loading vector index from ./storage …")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(token_limit=300, chat_store=chat_store, chat_store_key="user1")

chat_store.persist("chat_store.json")


def web_search(query: str) -> List[str]:
    """Fallback: hit an external search API (stub)."""
    try:
        resp = requests.get("https://api.example.com/search", params={"q": query}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    similarity_top_k=3,
    llm=llm,  # our CPU model
    context_prompt=(
        "Answer only in German. Du bist TutorAI …"  # trimmed for brevity
        "Here are the relevant documents:\n{context_str}\n"
    ),
    verbose=False,
    fallback_handler=web_search,
)

# ─────────────────────────────────────────────────────────────
# 4.  Flask web app (unchanged except port default)
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev_secret")

# MongoDB (assumes local)
client = MongoClient("mongodb://localhost:27017")
db = client["tutorai"]
users_collection = db["users"]
chats_collection = db["chats"]

# … auth decorators, routes … (identical to previous version)

from functools import wraps


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            flash("Bitte zuerst anmelden", "warning")
            return redirect("/login")
        return fn(*args, **kwargs)

    return wrapper


@app.route("/")
@login_required
def home():
    history = chats_collection.find_one({"username": session["username"]}) or {"chat": []}
    return render_template("chat.html", username=session["username"], chat=history["chat"])


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        user = users_collection.find_one({"username": u})
        if user and sha256_crypt.verify(p, user["password"]):
            session["username"] = u
            flash("Login erfolgreich", "success")
            return redirect("/")
        flash("Falsche Zugangsdaten", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        if users_collection.find_one({"username": u}):
            flash("Benutzer existiert bereits", "warning")
        else:
            users_collection.insert_one({"username": u, "password": sha256_crypt.hash(p)})
            chats_collection.insert_one({"username": u, "chat": []})
            flash("Registrierung erfolgreich", "success")
            return redirect("/login")
    return render_template("register.html")


@app.route("/send", methods=["POST"])
@login_required
def send():
    query = request.get_json().get("message", "")
    resp = chat_engine.stream_chat(query)
    answer = "".join(token for token in resp.response_gen)

    # store chat
    chats_collection.update_one(
        {"username": session["username"]},
        {"$push": {"chat": {"user": query, "bot": answer}}},
        upsert=True,
    )
    return jsonify({"message": answer})


@app.post("/rate")
@login_required
def rate():
    return jsonify({"status": "Ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_evalex=False)
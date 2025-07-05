"""create_kg_index_cpu.py – Build a Knowledge‑Graph index on **CPU‑only**.

This mirrors create_vector_index_cpu.py but generates subject‑predicate‑object
triples plus embeddings so they can be queried later via Llama‑Index’s
KnowledgeGraphIndex.

Usage:
    python -m scripts.create_kg_index_cpu \
        --data-dir data \
        --out-dir storage

Switch `--embed-model` and `--device` when you migrate to a GPU box.
"""

import argparse
import glob
import os
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------------------------
# Helper to set global Llama‑Index settings for a given device/model.
# ---------------------------------------------------------------------------

def configure_embeddings(model_name: str, device: str):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
    )
    # KnowledgeGraphIndex does not use an LLM for triple extraction when
    # extract_embed_model is provided, so we don’t need a generator model.


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_kg_index(input_dir: str, output_dir: str, model_name: str, device: str):
    """Parse files, extract triples+embeddings, and persist index."""

    print(f"🗂  Loading documents from {input_dir} …")
    docs = SimpleDirectoryReader(input_dir).load_data()
    print(f"   → {len(docs)} docs loaded")

    configure_embeddings(model_name, device)

    # Disable any generator LLM entirely so no OpenAI key is required
    Settings.llm = None  # 🚫 prevents fallback to OpenAI

    print("🔎  Extracting triples & building KG … (CPU, may take a while)")
    kg_index = KnowledgeGraphIndex.from_documents(docs, llm=None)

    # Persist under output_dir / kg_index
    persist_path = Path(output_dir) / "kg_index"
    persist_path.mkdir(parents=True, exist_ok=True)

    print(f"💾  Saving Knowledge‑Graph index to {persist_path} …")
    kg_index.storage_context.persist(persist_dir=str(persist_path))

    print("✅  Done! Knowledge‑Graph index ready.")


# ---------------------------------------------------------------------------
# CLI Entry‑point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build KG index (CPU)")
    parser.add_argument("--data-dir", required=True, help="Folder with source docs")
    parser.add_argument("--out-dir", required=True, help="Where to persist index")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Torch device string",
    )
    args = parser.parse_args()

    build_kg_index(args.data_dir, args.out_dir, args.embed_model, args.device)


if __name__ == "__main__":
    main()
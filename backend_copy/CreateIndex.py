"""create_vector_index_cpu.py – Builds the dense vector index *without a GPU*.

Usage (from repo root):
    python -m scripts.create_vector_index_cpu \
        --data-dir data \
        --out-dir storage

You can run the same script on a GPU machine later; just change the
--device flag to "cuda" and pick a larger embedding model.
"""

import argparse
import glob
import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------------------------
# 1. CLI arguments -----------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TutorAI vector index on CPU")
    parser.add_argument("--data-dir", type=str, default="data", help="Folder with raw source docs (PDF, txt, md, …)")
    parser.add_argument("--out-dir", type=str, default="storage", help="Where to write the persisted index")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Torch device for embeddings")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model")
    parser.add_argument("--chunk-size", type=int, default=512, help="Token chunk size for document splitting")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Helper: ensure dirs exist ------------------------------------------------
# ---------------------------------------------------------------------------

def check_paths(args: argparse.Namespace) -> None:
    if not Path(args.data_dir).exists():
        raise SystemExit(f"[!] data directory not found: {args.data_dir}")
    os.makedirs(args.out_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. Main build routine ------------------------------------------------------
# ---------------------------------------------------------------------------

def build_index(args: argparse.Namespace) -> None:
    """Read docs → embed → persist vector index."""

    # 3.1  Configure embedding model (CPU‑friendly by default)
    print(f"[•] Loading embedding model '{args.model}' on {args.device} …")
    embed_model = HuggingFaceEmbedding(model_name=args.model, device=args.device)

    Settings.embed_model = embed_model
    Settings.chunk_size = args.chunk_size

    # 3.2  Load documents (SimpleDirectoryReader supports PDF/txt/md)
    print(f"[•] Reading documents from {args.data_dir} …")
    reader = SimpleDirectoryReader(args.data_dir, recursive=True)
    documents = reader.load_data()
    print(f"[✓] Loaded {len(documents)} documents.")

    # 3.3  Build the vector index (no LLM required)
    print("[•] Embedding & indexing … this may take a while on CPU …")
    index = VectorStoreIndex.from_documents(documents)

    # 3.4  Persist to disk
    print(f"[•] Persisting index to {args.out_dir} …")
    index.storage_context.persist(persist_dir=args.out_dir)
    print("[✓] Done! You can now load the index from that folder at runtime.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _args = parse_args()
    check_paths(_args)
    build_index(_args)

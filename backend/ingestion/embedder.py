import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# CPU optimization — use all available cores
torch.set_num_threads(os.cpu_count() or 4)

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[Embedder] Loading model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model = _model.to("cpu")
        print("[Embedder] Model ready.")
    return _model


def embed_chunks(chunks: List[dict], batch_size: int = 64, show_progress: bool = True) -> List[dict]:
    if not chunks:
        return []

    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    print(f"[Embedder] Embedding {len(texts)} chunks...")

    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True  # pre-normalize so FAISS search is faster
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = all_embeddings[i]

    print(f"[Embedder] Done. Dim: {all_embeddings.shape[1]}")
    return chunks


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    return model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
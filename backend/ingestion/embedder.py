# backend/ingestion/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_model = None

def get_model():
    global _model
    if _model is None:
        print("[Embedder] Loading model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model = _model.to("cpu")
        _model.eval()
        print("[Embedder] Model ready.")
    return _model

def embed_chunks(chunks, batch_size=16):
    if not chunks:
        return chunks
    
    model = get_model()
    texts = [c["text"] for c in chunks]
    
    print(f"[Embedder] Embedding {len(texts)} chunks...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    
    gc.collect()
    return chunks

def embed_query(query):
    model = get_model()
    return model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
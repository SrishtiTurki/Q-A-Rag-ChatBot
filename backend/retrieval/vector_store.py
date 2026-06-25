import faiss
import numpy as np
import pickle
import os
from typing import List

# FAISS index and metadata stored in memory
_index = None
_metadata = []  # list of {"text": ..., "source": ..., "chunk_index": ...}

FAISS_INDEX_PATH = "vector_store/index.faiss"
METADATA_PATH = "vector_store/metadata.pkl"


def build_index(embedded_chunks: List[dict]):
    """Builds a FAISS flat index from embedded chunks."""
    global _index, _metadata

    if not embedded_chunks:
        print("[VectorStore] No chunks to index.")
        return

    embeddings = np.vstack([c["embedding"] for c in embedded_chunks]).astype("float32")
    dim = embeddings.shape[1]

    print(f"[VectorStore] Building FAISS index: {len(embedded_chunks)} chunks, dim={dim}")

    faiss.normalize_L2(embeddings)

    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)

    # Store ALL metadata
    _metadata = [
        {
            "text": c["text"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "page_number": c.get("page_number", "unknown"),
            "line_start": c.get("line_start", "unknown"),
            "line_end": c.get("line_end", "unknown"),
            "paragraph_start": c.get("paragraph_start", "unknown"),
            "uid": c.get("uid", f"{c['source']}_{c['chunk_index']}")
        }
        for c in embedded_chunks
    ]

    print(f"[VectorStore] Index built. Total vectors: {_index.ntotal}")


def add_to_index(embedded_chunks: List[dict]):
    """Adds new chunks to an existing index."""
    global _index, _metadata

    if _index is None:
        build_index(embedded_chunks)
        return

    embeddings = np.vstack([c["embedding"] for c in embedded_chunks]).astype("float32")
    faiss.normalize_L2(embeddings)
    _index.add(embeddings)

    for c in embedded_chunks:
        _metadata.append({
            "text": c["text"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "page_number": c.get("page_number", "unknown"),
            "line_start": c.get("line_start", "unknown"),
            "line_end": c.get("line_end", "unknown"),
            "paragraph_start": c.get("paragraph_start", "unknown"),
            "uid": c.get("uid", f"{c['source']}_{c['chunk_index']}")
        })

    print(f"[VectorStore] Added {len(embedded_chunks)} chunks. Total: {_index.ntotal}")


def save_index():
    """Persist FAISS index and metadata to disk."""
    os.makedirs("vector_store", exist_ok=True)
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(_metadata, f)
    print(f"[VectorStore] Saved index to {FAISS_INDEX_PATH}")


def load_index():
    """Load FAISS index and metadata from disk."""
    global _index, _metadata
    if not os.path.exists(FAISS_INDEX_PATH):
        print("[VectorStore] No saved index found.")
        return
    _index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        _metadata = pickle.load(f)
    print(f"[VectorStore] Loaded index. Total vectors: {_index.ntotal}")


def search(query_embedding: np.ndarray, top_k: int = 5):
    """Search the FAISS index for the most similar chunks."""
    if _index is None or _index.ntotal == 0:
        print("[VectorStore] Index is empty.")
        return []

    query = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    scores, indices = _index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((float(score), _metadata[idx]))

    return results


def reset_index():
    """Clear everything."""
    global _index, _metadata
    _index = None
    _metadata = []
    print("[VectorStore] Index reset.")


def get_index_stats():
    """Quick summary of what's in the index."""
    if _index is None:
        return {"status": "empty", "total_chunks": 0, "sources": []}
    sources = list(set(m["source"] for m in _metadata))
    return {
        "status": "ready",
        "total_chunks": _index.ntotal,
        "sources": sources
    }


def get_metadata():
    """Return the current metadata."""
    return _metadata


def delete_source(sources: List[str]) -> int:
    """
    Delete all chunks from the specified sources.
    Returns the number of chunks deleted.
    """
    global _index, _metadata
    
    if not sources or _index is None:
        return 0
    
    # Find indices to remove
    indices_to_remove = []
    for idx, meta in enumerate(_metadata):
        # Check if this chunk belongs to any of the sources to delete
        source = meta["source"]
        clean_source = source.replace(" [images]", "")
        if clean_source in sources or source in sources:
            indices_to_remove.append(idx)
    
    if not indices_to_remove:
        return 0
    
    # Remove duplicates and sort in reverse order
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)
    
    # Remove from metadata (reverse order to maintain indices)
    removed_count = len(indices_to_remove)
    for idx in indices_to_remove:
        if idx < len(_metadata):
            del _metadata[idx]
    
    print(f"[VectorStore] Removed {removed_count} chunks from metadata")
    
    # Rebuild the index from remaining metadata
    if _metadata:
        # Get embeddings for remaining chunks by re-encoding
        from ingestion.embedder import get_model
        model = get_model()
        texts = [m["text"] for m in _metadata]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Create new index
        dim = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(embeddings)
        _index = new_index
        
        print(f"[VectorStore] Rebuilt index with {_index.ntotal} vectors")
    else:
        # No metadata left, reset everything
        _index = None
        print("[VectorStore] No chunks left, index reset")
    
    return removed_count
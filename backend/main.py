# backend/main.py
import sys
import os

# ─── Fix Python Path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Imports ──────────────────────────────────────────────────────────────
import shutil
import tempfile
import logging
import time
import threading
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from ingestion.file_parser import parse_file, parse_images_only, SUPPORTED_EXTENSIONS
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_chunks, embed_query
from retrieval.vector_store import (
    add_to_index, save_index, load_index,
    reset_index, get_index_stats, delete_source, get_metadata
)
from retrieval.retriever import retrieve
from generation.prompt_builder import build_prompt, build_no_context_response
from generation.llm import generate_answer
from feedback.reward_model import reward_model

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_chatbot.log", encoding="utf-8")
    ]
)
log = logging.getLogger("rag")

# ─── State ────────────────────────────────────────────────────────────────
image_jobs: dict = {}
all_indexed_files: list = []
last_query_chunks: list = []

# ─── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global all_indexed_files
    log.info("=== RAG Chatbot starting up ===")
    load_index()
    stats = get_index_stats()
    if stats["status"] == "ready":
        seen = set()
        for m in get_metadata():
            clean = m["source"].replace(" [images]", "")
            if clean not in seen:
                seen.add(clean)
                all_indexed_files.append(clean)
        log.info(f"Loaded index: {stats['total_chunks']} chunks, {len(all_indexed_files)} files")
    else:
        log.info("No existing index — starting fresh")
    
    feedback_stats = reward_model.get_stats()
    log.info(f"Reward model loaded: {feedback_stats['total_feedback']} feedback entries")
    yield

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Q&A Chatbot", lifespan=lifespan)

# ─── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://q-a-rag-chatbot.streamlit.app",
        "http://localhost:8501",
        "http://localhost:8000",
        "https://q-a-rag-chatbot-frontend.onrender.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── Models ────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    selected_files: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    found: bool
    partial: bool = False
    chunks: List[dict] = []

class DeleteRequest(BaseModel):
    files: List[str]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    stars: int
    comment: str = ""
    response_time: Optional[float] = None

# ─── Root Endpoint ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}

# ─── Upload Endpoint ──────────────────────────────────────────────────────
@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    ocr_enabled: bool = False
):
    global all_indexed_files
    
    # Force OCR off to save memory
    ocr_enabled = False
    
    log.info(f"📤 Upload called with {len(files)} files")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    results = []
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        log.info(f"Processing: {file.filename}")
        
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "file": file.filename, 
                "status": "skipped", 
                "reason": f"Unsupported: {ext}"
            })
            continue
        
        # Read file content
        content = await file.read()
        log.info(f"   File size: {len(content)} bytes")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Parse text
            t0 = time.time()
            pages = parse_file(tmp_path)
            log.info(f"   Parsed: {len(pages)} pages ({time.time()-t0:.2f}s)")
            
            if not pages:
                results.append({
                    "file": file.filename, 
                    "status": "skipped", 
                    "reason": "No text extracted."
                })
                os.unlink(tmp_path)
                continue
            
            # Chunk
            chunks = chunk_pages(pages, source=file.filename)
            log.info(f"   Chunked: {len(chunks)} chunks")
            
            # Embed
            t0 = time.time()
            try:
                embedded = embed_chunks(chunks)
                log.info(f"   Embedded: {len(embedded)} vectors ({time.time()-t0:.2f}s)")
                
                add_to_index(embedded)
                save_index()
                
                if file.filename not in all_indexed_files:
                    all_indexed_files.append(file.filename)
                
                results.append({
                    "file": file.filename,
                    "status": "success",
                    "chunks_indexed": len(chunks),
                    "image_job_id": None
                })
            except Exception as embed_error:
                log.error(f"Embedding failed: {embed_error}")
                results.append({
                    "file": file.filename,
                    "status": "error",
                    "reason": f"Embedding failed: {str(embed_error)}"
                })
            
        except Exception as e:
            log.error(f"Failed: {file.filename}: {e}", exc_info=True)
            results.append({
                "file": file.filename, 
                "status": "error", 
                "reason": str(e)
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return {"results": results, "index_stats": get_index_stats()}

# ─── Query Endpoint ──────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    global last_query_chunks
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info(f"Query: '{question}' | files={request.selected_files or 'all'}")

    query_embedding = embed_query(question)

    is_summary = any(
        w in question.lower()
        for w in ["summarize", "summary", "summarise", "overview", "brief", 
                  "what is in", "what does", "tell me about", "describe"]
    )

    result = retrieve(
        query_embedding,
        query_text=question,
        filter_sources=request.selected_files,
        include_all_sources=is_summary
    )

    if not result["found"]:
        return QueryResponse(
            answer=build_no_context_response(), 
            sources=[], 
            found=False, 
            partial=False,
            chunks=[]
        )

    # Apply Reward Model Reranking
    reranked_chunks = reward_model.rerank_chunks(result["chunks"])
    top_k = 15 if is_summary else 8
    final_chunks = reranked_chunks[:top_k]
    result["chunks"] = final_chunks
    
    # Store chunks for feedback
    last_query_chunks = final_chunks

    # Track missing files
    queried_files = request.selected_files or all_indexed_files.copy()
    hit_bases = {c["source"].replace(" [images]", "") for c in final_chunks}
    missing_files = [f for f in queried_files if f not in hit_bases]

    # Build prompt
    prompt = build_prompt(
        question,
        final_chunks,
        partial_match=result["partial"],
        pdf_sources=result.get("pdf_sources", {}),
        all_indexed_files=queried_files,
        missing_files=missing_files
    )

    answer = generate_answer(prompt)
    log.info(f"   Answer: '{answer[:80]}...'")

    return QueryResponse(
        answer=answer,
        sources=result["sources"],
        found=True,
        partial=result["partial"],
        chunks=final_chunks
    )

# ─── Feedback Endpoint ──────────────────────────────────────────────────
@app.post("/feedback")
def add_feedback(feedback: FeedbackRequest):
    global last_query_chunks
    
    log.info(f"[Feedback] Received: {feedback.stars} stars")
    
    if feedback.stars < 1 or feedback.stars > 5:
        raise HTTPException(status_code=400, detail="Stars must be between 1 and 5")
    
    chunks = last_query_chunks
    
    if not chunks:
        log.warning("[Feedback] No chunks available for feedback!")
        return {
            "status": "warning",
            "message": "No chunks found for feedback. Please ask a question first.",
            "stars": feedback.stars
        }
    
    reward_model.add_feedback(
        question=feedback.question,
        answer=feedback.answer,
        chunks=chunks,
        stars=feedback.stars,
        comment=feedback.comment,
        response_time=feedback.response_time
    )
    
    stats = reward_model.get_stats()
    
    return {
        "status": "success",
        "stars": feedback.stars,
        "chunks_updated": len(chunks),
        "total_feedback": stats["total_feedback"],
        "average_rating": stats["average_rating"]
    }

# ─── Feedback Stats Endpoint ────────────────────────────────────────────
@app.get("/feedback/stats")
def get_feedback_stats():
    stats = reward_model.get_stats()
    stats["best_sources"] = reward_model.get_best_sources()
    stats["poor_sources"] = reward_model.get_poor_sources()
    return stats

# ─── Get Files Endpoint ──────────────────────────────────────────────────
@app.get("/files")
def get_files():
    metadata = get_metadata()
    
    counts = {}
    for m in metadata:
        clean = m["source"].replace(" [images]", "")
        is_img = "[images]" in m["source"]
        if clean not in counts:
            counts[clean] = {"text": 0, "images": 0, "pages": set()}
        if is_img:
            counts[clean]["images"] += 1
        else:
            counts[clean]["text"] += 1
        counts[clean]["pages"].add(m.get("page_number", "?"))

    files = []
    for name in all_indexed_files:
        c = counts.get(name, {"text": 0, "images": 0, "pages": set()})
        files.append({
            "name": name,
            "text_chunks": c["text"],
            "image_chunks": c["images"],
            "total_chunks": c["text"] + c["images"],
            "pages": len([p for p in c["pages"] if p != "?"])
        })

    return {"files": files, "count": len(files)}

# ─── Stats Endpoint ──────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    return get_index_stats()

# ─── Reset Endpoint ──────────────────────────────────────────────────────
@app.post("/reset")
def reset():
    global all_indexed_files, last_query_chunks
    reset_index()
    image_jobs.clear()
    all_indexed_files = []
    last_query_chunks = []
    for path in ["vector_store/index.faiss", "vector_store/metadata.pkl"]:
        if os.path.exists(path):
            os.remove(path)
    log.info("Index reset.")
    return {"status": "cleared"}

# ─── Image Status Endpoint ──────────────────────────────────────────────
@app.get("/image-status/{job_id}")
def image_status(job_id: str):
    if job_id not in image_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return image_jobs[job_id]

# ─── Delete Files Endpoint ──────────────────────────────────────────────
@app.delete("/files")
def delete_files(request: DeleteRequest):
    global all_indexed_files

    if not request.files:
        raise HTTPException(status_code=400, detail="No files specified.")

    removed = delete_source(request.files)

    for f in request.files:
        if f in all_indexed_files:
            all_indexed_files.remove(f)
        for jid, job in list(image_jobs.items()):
            if job["file"] == f:
                del image_jobs[jid]

    if removed > 0:
        save_index()

    log.info(f"Deleted {request.files}, removed {removed} chunks")
    return {
        "deleted": request.files,
        "chunks_removed": removed,
        "index_stats": get_index_stats()
    }
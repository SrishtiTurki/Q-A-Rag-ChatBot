import os
import sys
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

# ─── Import Reward Model ──────────────────────────────────────────────────
from feedback.reward_model import reward_model

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
        ),
        logging.FileHandler("rag_chatbot.log", encoding="utf-8")
    ]
)
log = logging.getLogger("rag")

# ─── State ────────────────────────────────────────────────────────────────────

image_jobs: dict = {}          # job_id -> status dict
all_indexed_files: list = []   # tracks filenames in order
last_query_chunks: list = []   # NEW: Store chunks from last query for feedback
st.session_state.ocr_enabled = False
# ─── Lifespan ─────────────────────────────────────────────────────────────────

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
    
    # NEW: Log reward model status
    feedback_stats = reward_model.get_stats()
    log.info(f"Reward model loaded: {feedback_stats['total_feedback']} feedback entries")
    
    yield

app = FastAPI(title="RAG Q&A Chatbot", lifespan=lifespan)
from fastapi.middleware.cors import CORSMiddleware

# Add this after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://q-a-rag-chatbot.streamlit.app",  # Your Streamlit app
        "http://localhost:8501",                   # Local development
        "https://q-a-rag-chatbot-frontend.onrender.com",  # If on Render
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    selected_files: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    found: bool
    partial: bool = False
    chunks: List[dict] = []  # NEW: Return chunks for feedback

class DeleteRequest(BaseModel):
    files: List[str]

# ─── NEW: Feedback Models ────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    stars: int
    comment: str = ""
    response_time: Optional[float] = None

# ─── Background image indexing ────────────────────────────────────────────────

def _run_image_indexing(job_id: str, tmp_path: str, filename: str):
    try:
        log.info(f"[ImageJob {job_id}] Starting: {filename}")
        image_jobs[job_id] = {"status": "processing", "file": filename, "message": "Extracting images..."}

        image_pages = parse_images_only(tmp_path)

        if not image_pages:
            image_jobs[job_id] = {"status": "done", "file": filename, "message": "No images found."}
            return

        image_jobs[job_id]["message"] = f"Indexing {len(image_pages)} image(s)..."
        chunks = chunk_pages(image_pages, source=f"{filename} [images]")
        embedded = embed_chunks(chunks, show_progress=False)
        add_to_index(embedded)
        save_index()

        image_jobs[job_id] = {
            "status": "done",
            "file": filename,
            "message": f"Images ready - {len(chunks)} image chunks indexed."
        }
        log.info(f"[ImageJob {job_id}] Done: {len(chunks)} chunks")

    except Exception as e:
        log.error(f"[ImageJob {job_id}] Failed: {e}", exc_info=True)
        image_jobs[job_id] = {"status": "error", "file": filename, "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), ocr_enabled: bool = True):
    global all_indexed_files

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        log.info(f"Processing: {file.filename}")

        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"file": file.filename, "status": "skipped", "reason": f"Unsupported: {ext}"})
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            pages = parse_file(tmp_path)
            log.info(f"   Parsed: {len(pages)} pages ({time.time()-t0:.2f}s)")

            if not pages:
                results.append({"file": file.filename, "status": "skipped", "reason": "No text extracted."})
                os.unlink(tmp_path)
                continue

            chunks = chunk_pages(pages, source=file.filename)
            log.info(f"   Chunked: {len(chunks)} chunks")

            t0 = time.time()
            embedded = embed_chunks(chunks)
            log.info(f"   Embedded: {len(embedded)} vectors ({time.time()-t0:.2f}s)")

            add_to_index(embedded)
            save_index()

            if file.filename not in all_indexed_files:
                all_indexed_files.append(file.filename)

            job_id = None
            if ocr_enabled and ext in {".pdf", ".docx"}:
                job_id = str(uuid.uuid4())[:8]
                bg_tmp = tmp_path + "_bg"
                shutil.copy(tmp_path, bg_tmp)
                image_jobs[job_id] = {"status": "processing", "file": file.filename, "message": "Starting..."}
                threading.Thread(
                    target=_run_image_indexing,
                    args=(job_id, bg_tmp, file.filename),
                    daemon=True
                ).start()
                log.info(f"   Image job started: {job_id}")

            results.append({
                "file": file.filename,
                "status": "success",
                "chunks_indexed": len(chunks),
                "image_job_id": job_id
            })

        except Exception as e:
            log.error(f"Failed: {file.filename}: {e}", exc_info=True)
            results.append({"file": file.filename, "status": "error", "reason": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return {"results": results, "index_stats": get_index_stats()}


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


@app.get("/image-status/{job_id}")
def image_status(job_id: str):
    if job_id not in image_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return image_jobs[job_id]


# ─── UPDATED: Query Endpoint with Reward Model ──────────────────────────────

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
                  "what is in", "what does", "tell me about", "describe",
                  "what are", "what's in"]
    )

    # Get initial retrieval
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
            chunks=[]  # NEW: Empty chunks
        )

    # ─── Apply Reward Model Reranking ──────────────────────────────────
    reranked_chunks = reward_model.rerank_chunks(result["chunks"])
    
    # Take top chunks (more for summaries)
    top_k = 15 if is_summary else 8
    final_chunks = reranked_chunks[:top_k]
    result["chunks"] = final_chunks
    
    # ─── Store chunks for feedback ──────────────────────────────────────
    last_query_chunks = final_chunks
    log.info(f"Stored {len(final_chunks)} chunks for feedback")

    # ─── Track missing files ──────────────────────────────────────────────
    queried_files = request.selected_files or all_indexed_files.copy()
    hit_bases = {c["source"].replace(" [images]", "") for c in final_chunks}
    missing_files = [f for f in queried_files if f not in hit_bases]

    # Log chunk scores for debugging
    for i, chunk in enumerate(final_chunks[:3]):
        log.info(
            f"   Chunk {i+1}: score={chunk.get('adjusted_score', chunk.get('score', 0)):.4f} "
            f"| {chunk['source'][:30]} | page={chunk.get('page_number', '?')}"
        )

    # ─── Build prompt ──────────────────────────────────────────────────────
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

    # ─── Return with chunks for feedback ──────────────────────────────────
    return QueryResponse(
        answer=answer,
        sources=result["sources"],
        found=True,
        partial=result["partial"],
        chunks=final_chunks  # NEW: Return chunks
    )


# ─── NEW: Feedback Endpoint ──────────────────────────────────────────────────

@app.post("/feedback")
def add_feedback(feedback: FeedbackRequest):
    """
    Endpoint to receive star ratings and update reward model.
    """
    global last_query_chunks
    
    log.info(f"[Feedback] Received: {feedback.stars} stars for: {feedback.question[:50]}...")
    
    if feedback.stars < 1 or feedback.stars > 5:
        raise HTTPException(status_code=400, detail="Stars must be between 1 and 5")
    
    # Get chunks from the last query
    chunks = last_query_chunks
    
    if not chunks:
        log.warning("[Feedback] No chunks available for feedback!")
        return {
            "status": "warning",
            "message": "No chunks found for feedback. Please ask a question first.",
            "stars": feedback.stars
        }
    
    log.info(f"[Feedback] Updating {len(chunks)} chunks with {feedback.stars} stars")
    
    # Update reward model
    reward_model.add_feedback(
        question=feedback.question,
        answer=feedback.answer,
        chunks=chunks,
        stars=feedback.stars,
        comment=feedback.comment,
        response_time=feedback.response_time
    )
    
    # Get updated stats
    stats = reward_model.get_stats()
    log.info(f"[Feedback] Updated stats: total={stats['total_feedback']}, avg={stats['average_rating']:.2f}")
    
    return {
        "status": "success",
        "stars": feedback.stars,
        "chunks_updated": len(chunks),
        "total_feedback": stats["total_feedback"],
        "average_rating": stats["average_rating"]
    }


# ─── NEW: Feedback Stats Endpoint ────────────────────────────────────────────

@app.get("/feedback/stats")
def get_feedback_stats():
    """Get feedback statistics for dashboard."""
    stats = reward_model.get_stats()
    stats["best_sources"] = reward_model.get_best_sources()
    stats["poor_sources"] = reward_model.get_poor_sources()
    
    log.info(f"[Stats] Requested: total={stats['total_feedback']}, avg={stats['average_rating']:.2f}")
    
    return stats


# ─── Existing Endpoints ──────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    return get_index_stats()


@app.post("/reset")
def reset():
    global all_indexed_files, last_query_chunks
    reset_index()
    image_jobs.clear()
    all_indexed_files = []
    last_query_chunks = []  # NEW: Clear stored chunks
    for path in ["vector_store/index.faiss", "vector_store/metadata.pkl"]:
        if os.path.exists(path):
            os.remove(path)
    log.info("Index reset.")
    return {"status": "cleared"}
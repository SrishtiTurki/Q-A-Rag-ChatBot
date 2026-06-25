import streamlit as st
import requests
import time
from datetime import datetime
import os

# ─── Configuration ─────────────────────────────────────────────────────────────

def get_backend_url():
    """Get backend URL from Streamlit secrets or environment."""
    # Try Streamlit Cloud secrets
    try:
        if hasattr(st, 'secrets') and 'BACKEND_URL' in st.secrets:
            return st.secrets['BACKEND_URL']
    except:
        pass
    
    # Try Render environment variable
    render_url = os.environ.get("BACKEND_URL")
    if render_url:
        return render_url
    
    # Local development fallback
    return "http://localhost:8000"

BACKEND_URL = get_backend_url()

st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    .sidebar-header {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
        padding: 10px 0;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .file-item {
        padding: 8px 10px;
        margin: 4px 0;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    .file-item:hover {
        background: #f1f5f9;
        border-color: #94a3b8;
    }
    .file-item.selected {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
    
    .stats-bar {
        display: flex;
        gap: 20px;
        padding: 10px 16px;
        background: #f1f5f9;
        border-radius: 10px;
        margin: 10px 0;
        flex-wrap: wrap;
        font-size: 14px;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-text { background: #d1fae5; color: #065f46; }
    .badge-image { background: #dbeafe; color: #1e40af; }
    .badge-processing { background: #fef3c7; color: #92400e; }
    .badge-error { background: #fecaca; color: #991b1b; }
    
    .processing-badge {
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .source-citation {
        font-size: 13px;
        color: #475569;
        padding: 4px 8px;
        background: #f1f5f9;
        border-radius: 4px;
        margin: 2px 0;
    }
    
    /* Compact feedback styles */
    .feedback-container {
        padding: 8px 12px;
        background: #fafbfc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 8px;
    }
    
    /* Star button hover effects */
    .stButton button {
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chat_history = []
    st.session_state.image_jobs = {}
    st.session_state.image_status = {}
    st.session_state.selected_files = []
    st.session_state.ocr_enabled = True
    st.session_state.files = []
    st.session_state.stats = None
    st.session_state.last_refresh = 0
    st.session_state.data_loaded = False
    st.session_state.feedback_sent = {}
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.last_chunks = []

# ─── Helper Functions ──────────────────────────────────────────────────────

def check_backend():
    """Check if backend is running."""
    try:
        return requests.get(f"{BACKEND_URL}/", timeout=2).status_code == 200
    except:
        return False

def refresh_data(force=False):
    """Refresh file list and stats from backend."""
    current = time.time()
    if not force and current - st.session_state.last_refresh < 5:
        return
    
    try:
        r = requests.get(f"{BACKEND_URL}/files", timeout=5)
        if r.ok:
            st.session_state.files = r.json().get("files", [])
    except:
        pass
    
    try:
        r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if r.ok:
            st.session_state.stats = r.json()
    except:
        pass
    
    st.session_state.last_refresh = current
    st.session_state.data_loaded = True

def check_image_jobs():
    """Check status of background image processing jobs."""
    has_pending = False
    
    for filename, job_id in list(st.session_state.image_jobs.items()):
        current_status = st.session_state.image_status.get(job_id, {}).get("status")
        
        if current_status in ("done", "error"):
            continue
        
        has_pending = True
        try:
            r = requests.get(f"{BACKEND_URL}/image-status/{job_id}", timeout=5)
            if r.ok:
                status_data = r.json()
                st.session_state.image_status[job_id] = status_data
                if status_data.get("status") in ("done", "error"):
                    refresh_data(force=True)
        except:
            pass
    
    return has_pending

def get_badge(filename):
    """Get badge status for a file."""
    jid = st.session_state.image_jobs.get(filename)
    if not jid:
        return "text", "Text only"
    
    s = st.session_state.image_status.get(jid, {}).get("status")
    
    if not s:
        return "text", "Text only"
    if s == "done":
        return "image", "✅ Text + Images"
    if s == "processing":
        return "processing", "⏳ Processing..."
    if s == "error":
        return "error", "❌ Error"
    if s == "pending":
        return "processing", "⏳ Queued..."
    return "text", "Text ready"

def badge_html(label, t):
    """Generate HTML for badge."""
    classes = {
        "text": "badge-text",
        "image": "badge-image",
        "processing": "badge-processing processing-badge",
        "error": "badge-error",
    }
    cls = classes.get(t, "badge-text")
    return f'<span class="badge {cls}">{label}</span>'

def delete_file(filename):
    """Delete a file from the system."""
    try:
        with st.spinner(f"Deleting {filename}..."):
            r = requests.delete(
                f"{BACKEND_URL}/files", json={"files": [filename]}, timeout=60
            )
            if r.ok:
                st.session_state.selected_files = [
                    f for f in st.session_state.selected_files if f != filename
                ]
                jid = st.session_state.image_jobs.pop(filename, None)
                if jid:
                    st.session_state.image_status.pop(jid, None)
                refresh_data(force=True)
                return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
    return False

def send_feedback(question: str, answer: str, stars: int, comment: str = "", response_time: float = None):
    """Send feedback to backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "question": question,
                "answer": answer[:500],
                "stars": stars,
                "comment": comment,
                "response_time": response_time
            },
            timeout=5
        )
        return response.ok
    except Exception as e:
        st.error(f"Failed to send feedback: {e}")
        return False

def render_star_rating(question: str, answer: str, idx: int):
    """Render compact star rating component for feedback."""
    # Check if feedback already sent
    if st.session_state.feedback_sent.get(idx, False):
        st.caption("✅ Feedback submitted")
        return
    
    # Use a compact container
    with st.container():
        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        
        # Compact layout with columns
        col1, col2, col3 = st.columns([0.5, 3.5, 0.5])
        
        with col1:
            st.caption("Rate:")
        
        with col2:
            # Star buttons in a single row
            star_cols = st.columns(5)
            selected_stars = 0
            
            # Star emojis
            star_emojis = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
            
            for i, emoji in enumerate(star_emojis, 1):
                with star_cols[i-1]:
                    # Check if this star was selected
                    is_selected = st.session_state.get(f"star_selected_{idx}", 0) == i
                    
                    if st.button(
                        emoji, 
                        key=f"star_{idx}_{i}", 
                        help=f"{i} stars",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        # Toggle selection
                        if is_selected:
                            st.session_state[f"star_selected_{idx}"] = 0
                            selected_stars = 0
                        else:
                            st.session_state[f"star_selected_{idx}"] = i
                            selected_stars = i
                        st.rerun()
                    
                    # Show if selected
                    if is_selected:
                        st.caption(f"⭐ {i}")
        
        with col3:
            st.caption("")
        
        # If star selected, show compact feedback form
        selected_stars = st.session_state.get(f"star_selected_{idx}", 0)
        
        if selected_stars > 0:
            st.caption(f"Selected: {'⭐' * selected_stars}")
            
            # Compact feedback form
            comment = ""
            if selected_stars <= 3:
                comment_col, submit_col = st.columns([3, 1])
                
                with comment_col:
                    comment = st.text_input(
                        "💬 What went wrong?",
                        key=f"comment_{idx}",
                        placeholder="Missing info? Wrong citations? (optional)",
                        label_visibility="collapsed"
                    )
                
                with submit_col:
                    if st.button("📤 Send", key=f"submit_{idx}", type="primary", use_container_width=True):
                        with st.spinner("..."):
                            if send_feedback(question, answer, selected_stars, comment):
                                st.session_state.feedback_sent[idx] = True
                                st.success(f"✅ {selected_stars}⭐")
                                st.rerun()
            else:
                # For 4-5 stars, simpler submission
                if st.button("📤 Submit", key=f"submit_{idx}", type="primary"):
                    with st.spinner("..."):
                        if send_feedback(question, answer, selected_stars, ""):
                            st.session_state.feedback_sent[idx] = True
                            st.success(f"✅ {selected_stars}⭐ Thanks!")
                            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Bootstrap ──────────────────────────────────────────────────────────────

online = check_backend()

if not st.session_state.data_loaded:
    refresh_data(force=True)

# Check pending image jobs
has_pending_jobs = check_image_jobs()
if has_pending_jobs:
    time.sleep(2)
    st.rerun()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 RAG Q&A System</div>', unsafe_allow_html=True)
    
    # Show backend status
    st.markdown(f"**Status:** {'🟢 Online' if online else '🔴 Offline'}")
    st.caption(f"🔗 Backend: {BACKEND_URL}")
    
    if online and st.session_state.stats:
        s = st.session_state.stats
        st.markdown(
            f'<div class="stats-bar">'
            f"📄 {len(st.session_state.files)} files &nbsp;|&nbsp; "
            f"📦 {s.get('total_chunks', 0)} chunks"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # ─── Settings ──────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    st.session_state.ocr_enabled = st.toggle("🔍 Enable Image OCR", st.session_state.ocr_enabled)
    
    st.markdown("---")
    
    # ─── Upload Section ──────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")
    st.caption("Supported: PDF, DOCX, TXT, CSV, PNG, JPG, JPEG")
    
    uploaded = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
        disabled=not online
    )
    
    if st.button(
        "🚀 Upload & Index",
        type="primary",
        use_container_width=True,
        disabled=not online or not uploaded
    ):
        with st.spinner("Indexing files..."):
            file_objects = [("files", (f.name, f.read(), f.type)) for f in uploaded]
            try:
                r = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=file_objects,
                    params={"ocr_enabled": st.session_state.ocr_enabled},
                    timeout=180
                )
                data = r.json()
                
                for res in data.get("results", []):
                    if res["status"] == "success":
                        st.success(f"✅ {res['file']} — {res.get('chunks_indexed', 0)} chunks")
                        if res.get("image_job_id"):
                            jid = res["image_job_id"]
                            st.session_state.image_jobs[res["file"]] = jid
                            st.session_state.image_status[jid] = {"status": "processing"}
                    elif res["status"] == "skipped":
                        st.warning(f"⚠️ {res['file']} — {res.get('reason', 'Skipped')}")
                    else:
                        st.error(f"❌ {res['file']} — {res.get('reason', 'Error')}")
            except Exception as e:
                st.error(f"Upload failed: {e}")
        
        refresh_data(force=True)
        st.rerun()
    
    st.markdown("---")
    
    # ─── File Selection ──────────────────────────────────────────────────
    st.markdown("### 📄 Select Files")
    st.caption("Leave unchecked = query all files")
    
    files = st.session_state.files
    
    if files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All", use_container_width=True):
                st.session_state.selected_files = [f["name"] for f in files]
                st.rerun()
        with col2:
            if st.button("❌ Deselect All", use_container_width=True):
                st.session_state.selected_files = []
                st.rerun()
        
        st.markdown("---")
        
        # Display file list with selection
        for f in files:
            name = f["name"]
            is_selected = name in st.session_state.selected_files
            btype, blabel = get_badge(name)
            
            col_check, col_info, col_del = st.columns([0.15, 0.65, 0.2])
            
            with col_check:
                checked = st.checkbox(
                    f"Select {name}",
                    value=is_selected,
                    key=f"cb_{name}",
                    label_visibility="collapsed"
                )
                if checked and name not in st.session_state.selected_files:
                    st.session_state.selected_files.append(name)
                elif not checked and name in st.session_state.selected_files:
                    st.session_state.selected_files.remove(name)
            
            with col_info:
                chunks = f.get("text_chunks", 0)
                img = f.get("image_chunks", 0)
                short_name = name[:25] + ("..." if len(name) > 25 else "")
                st.markdown(
                    f'<div class="file-item{" selected" if is_selected else ""}">'
                    f"<strong>{short_name}</strong><br>"
                    f"{badge_html(blabel, btype)}"
                    f' <span style="font-size:11px;color:#64748b;">{chunks} text'
                    f"{f' + {img} img' if img else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col_del:
                if st.button("🗑️", key=f"del_{name}", help=f"Delete {name}"):
                    if delete_file(name):
                        st.success(f"✅ Deleted {name}")
                        st.rerun()
        
        st.markdown("---")
        
        # Selected files summary
        selected_count = len(st.session_state.selected_files)
        if selected_count > 0:
            st.info(f"📌 {selected_count} file(s) selected")
            if st.button(f"🗑️ Delete Selected ({selected_count})", use_container_width=True):
                with st.spinner("Deleting selected files..."):
                    for fname in st.session_state.selected_files.copy():
                        delete_file(fname)
                st.rerun()
        else:
            st.info("📌 Querying all files")
    else:
        st.info("No files indexed yet")
    
    st.markdown("---")
    
    # ─── Actions ──────────────────────────────────────────────────────────
    col_refresh, col_clear = st.columns(2)
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            refresh_data(force=True)
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True, disabled=not online):
            try:
                response = requests.post(f"{BACKEND_URL}/reset", timeout=30)
                if response.status_code == 200:
                    st.session_state.selected_files = []
                    st.session_state.image_jobs = {}
                    st.session_state.image_status = {}
                    st.session_state.chat_history = []
                    st.session_state.files = []
                    st.session_state.stats = None
                    st.session_state.data_loaded = False
                    st.session_state.feedback_sent = {}
                    st.success("✅ All data cleared!")
                    refresh_data(force=True)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    
    # ─── Analytics Section ──────────────────────────────────────────────
    if st.button("📊 View Feedback Analytics", use_container_width=True):
        try:
            r = requests.get(f"{BACKEND_URL}/feedback/stats", timeout=5)
            if r.ok:
                stats = r.json()
                
                st.markdown("#### 📊 System Stats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Feedback", stats.get("total_feedback", 0))
                with col2:
                    avg = stats.get("average_rating", 0)
                    st.metric("⭐ Avg Rating", f"{avg:.1f}")
                
                # Star distribution
                dist = stats.get("stars_distribution", {})
                if dist:
                    st.markdown("#### 📈 Ratings Distribution")
                    for stars in range(1, 6):
                        count = dist.get(stars, 0)
                        if count > 0:
                            st.progress(count / max(dist.values()) if dist.values() else 1)
                            st.text(f"{'⭐' * stars}: {count}")
                
                # Best sources
                best = stats.get("best_sources", [])
                if best:
                    st.markdown("#### ✅ Best Sources")
                    for s in best[:3]:
                        st.success(f"📄 {s['source'][:30]}: {s['score']:.2f}")
                
                # Poor sources
                poor = stats.get("poor_sources", [])
                if poor:
                    st.markdown("#### 🔴 Needs Improvement")
                    for s in poor[:3]:
                        st.warning(f"📄 {s['source'][:30]}: {s['score']:.2f}")
        except:
            st.error("Could not fetch stats")

# ─── Main Chat Interface ──────────────────────────────────────────────────

st.markdown('<div style="padding: 10px 0 20px 0;">', unsafe_allow_html=True)
st.title("💬 Document Q&A")

if st.session_state.selected_files:
    st.caption(f"📌 Querying **{len(st.session_state.selected_files)}** selected file(s)")
else:
    st.caption("📌 Querying **all** files")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Display Chat History ────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show sources for assistant messages
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f'<div class="source-citation">📄 {s}</div>', unsafe_allow_html=True)
        
        # Show feedback for assistant messages (only for recent ones)
        if msg["role"] == "assistant" and idx > 0 and idx >= len(st.session_state.chat_history) - 3:
            question = st.session_state.chat_history[idx-1]["content"] if idx > 0 else ""
            render_star_rating(question, msg["content"], idx)

# ─── Chat Input ──────────────────────────────────────────────────────────

question = st.chat_input("Ask something about your documents...")

if question:
    start_time = time.time()
    
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                payload = {"question": question}
                if st.session_state.selected_files:
                    payload["selected_files"] = st.session_state.selected_files
                
                r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=90)
                data = r.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                found = data.get("found", False)
                
                # Store for feedback
                st.session_state.last_question = question
                st.session_state.last_answer = answer
                
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []
                found = False
        
        response_time = time.time() - start_time
        
        # Display answer
        st.markdown(answer)
        
        # Show sources
        if sources:
            with st.expander("📚 Sources", expanded=True):
                for s in sources:
                    st.markdown(f'<div class="source-citation">📄 {s}</div>', unsafe_allow_html=True)
        elif not found:
            st.info("ℹ️ No relevant documents found. Try rephrasing or selecting different files.")
        
        # Store assistant message
        msg_idx = len(st.session_state.chat_history)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "response_time": response_time,
            "question": question
        })
        
        # Show feedback for the response
        st.markdown("---")
        render_star_rating(question, answer, msg_idx)

# ─── Footer ───────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("💡 Tip: Rate answers with ⭐ to help improve the system!")

# ─── Auto-refresh for background jobs ──────────────────────────────────

if has_pending_jobs:
    with st.sidebar:
        st.info("⏳ Processing images in background...")
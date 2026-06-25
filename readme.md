# 📚 RAG Q&A Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-0052CC?style=for-the-badge&logo=facebook&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral_AI-API-FF6B00?style=for-the-badge&logo=mistralai&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**[Live Demo](https://your-app.streamlit.app)** • **[API Docs](https://your-backend.onrender.com/docs)** • **[Report Bug](https://github.com/yourusername/rag-chatbot/issues)** • **[Request Feature](https://github.com/yourusername/rag-chatbot/issues)**

</div>

---

## 🎯 Overview

**RAG Q&A Chatbot** is a production-ready Retrieval-Augmented Generation system that allows you to upload documents and ask questions with accurate, source-cited answers. It features a self-improving RLHF feedback system that learns from user ratings to provide better responses over time.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Support** | Upload PDF, DOCX, TXT, CSV, PNG, JPG, JPEG |
| 🔍 **Smart Retrieval** | FAISS vector search with intelligent reranking |
| 🤖 **LLM Integration** | Support for Mistral AI |
| ⭐ **RLHF Feedback** | 1-5 star rating system that improves results |
| 📊 **Analytics Dashboard** | Track performance, see what's working |
| 📝 **Source Citations** | Every fact includes (Document, Page, Line) |
| 💬 **Conversation Memory** | Multi-turn follow-up questions |
| 🚀 **Production Ready** | Deployable on Render + Streamlit Cloud |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (Streamlit Cloud)                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Sidebar: Upload, File Selection, Settings, Analytics     │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  Main: Chat Interface with Star Ratings & Feedback        │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTPS API
┌───────────────────────────▼──────────────────────────────────────┐
│                        BACKEND API                               │
│                      (Render.com)                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Endpoints: /upload, /query, /feedback, /stats            │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                     PROCESSING PIPELINE                          │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ File Parser │───▶│  Chunker    │───▶│  Embedder   │         │
│  │ PDF/TXT/    │    │  500 chars  │    │  MiniLM-v2  │         │
│  │ DOCX/CSV/IMG│    │  50 overlap │    │  384 dims   │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                 │
│                                                ▼                 │
│                                     ┌─────────────────────┐     │
│                                     │  Vector Store       │     │
│                                     │  FAISS Index        │     │
│                                     │  Metadata.pkl       │     │
│                                     └─────────────────────┘     │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Retriever  │───▶│  Reranker   │───▶│  LLM        │         │
│  │  FAISS      │    │  Reward     │    │  Mistral/   │         │
│  │  Top-K      │    │  Model      │    │  Gemini     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```
Python 3.9+
Mistral AI API Key
Git
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the backend (local development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# In a new terminal, run the frontend
streamlit run app.py
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required - Choose your LLM
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── backend/                     # FastAPI Backend
│   ├── main.py                 # API endpoints
│   ├── requirements.txt        # Backend dependencies
│   ├── ingestion/              # Document processing
│   │   ├── file_parser.py     # Multi-format parsing
│   │   ├── chunker.py         # Text chunking
│   │   ├── embedder.py        # Embedding generation
│   │   └── ocr.py             # Image OCR
│   ├── retrieval/              # Vector search
│   │   ├── vector_store.py    # FAISS operations
│   │   └── retriever.py       # Search & reranking
│   ├── generation/             # LLM integration
│   │   ├── llm.py             # Mistral/Gemini API
│   │   └── prompt_builder.py  # Prompt engineering
│   └── feedback/               # RLHF system
│       └── reward_model.py    # Star-based learning
│
├── frontend/                    # Streamlit Frontend
│   ├── app.py                  # Main UI
│   ├── requirements.txt        # Frontend dependencies
│   └── .streamlit/
│       └── config.toml         # Streamlit config
│
├── vector_store/                # Auto-created
│   ├── index.faiss             # FAISS vector index
│   └── metadata.pkl            # Chunk metadata
│
├── feedback_data/               # Auto-created
│   ├── star_reward_model.pkl   # Learned weights
│   └── feedback_log.json       # Feedback history
│
├── .env.example                 # Environment template
├── .gitignore
└── README.md
```

---

## 🎯 How It Works

### 1. Document Upload & Indexing
```
📄 Upload PDF → 📝 Extract text → ✂️ Chunk into 500 char pieces
→ 🔢 Generate embeddings → 💾 Store in FAISS
```

### 2. Query & Retrieval
```
❓ User question → 🔍 Search FAISS → 📊 Get top 15 chunks
→ ⭐ Rerank with reward model → 📤 Pass top chunks to LLM
```

### 3. Feedback & Learning
```
✅ User rates answer (1-5⭐) → 🔄 Convert to reward signal
→ 📈 Update chunk scores → 🧠 System improves over time
```

---

## ⭐ RLHF Feedback System

### How Star Ratings Work

| Stars | Reward Signal | Meaning |
|-------|--------------|---------|
| ⭐⭐⭐⭐⭐ | +1.0 | Excellent - Perfect answer |
| ⭐⭐⭐⭐ | +0.5 | Good - Helpful answer |
| ⭐⭐⭐ | 0.0 | Neutral - Acceptable |
| ⭐⭐ | -0.5 | Poor - Needs improvement |
| ⭐ | -1.0 | Very Poor - Not helpful |

### Learning Flow
```
1. User gives 2⭐ → Reward = -0.5
2. System updates each chunk used in the answer
3. Chunks get penalized (-0.5 * position_weight)
4. Source/document gets slightly penalized
5. Future queries rerank these chunks lower
6. System improves over time! 📈
```

---

## 🚀 Deployment

### Backend → Render

1. Push `backend/` to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add `MISTRAL_API_KEY` in environment variables

### Frontend → Streamlit Cloud

1. Push `frontend/` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `BACKEND_URL` in Streamlit secrets to your Render URL

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Backend** | FastAPI, Uvicorn | API server |
| **Frontend** | Streamlit | UI interface |
| **Vector DB** | FAISS | Similarity search |
| **Embeddings** | sentence-transformers (MiniLM-L6-v2) | Text vectorization |
| **LLM** | Mistral AI | Answer generation |
| **OCR** | EasyOCR | Image text extraction |
| **Document** | PyMuPDF, python-docx, Pillow | File parsing |
| **ML** | PyTorch, NumPy, SciPy | ML operations |
| **Deployment** | Render, Streamlit Cloud | Hosting |

---

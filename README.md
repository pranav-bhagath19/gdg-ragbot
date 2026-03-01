# 🤖 Free-Tier RAG Chatbot API

**FastAPI · sentence-transformers · hnswlib · Groq + Gemini Fallback**

A production-ready Retrieval-Augmented Generation (RAG) API that lets you chat with your PDF, Excel, and Word documents — for **$0/month**.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture)
2. [Folder Structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Get API Keys](#get-api-keys)
5. [Local Setup & Run](#local-setup)
6. [API Endpoints](#api-endpoints)
7. [Testing Guide](#testing)
8. [Docker Setup](#docker)
9. [Deploy to Render](#deploy-render)
10. [Deploy to Railway](#deploy-railway)
11. [Cold Start Prevention](#cold-start)
12. [Troubleshooting](#troubleshooting)
13. [Production Checklist](#production-checklist)

---

## 🏗️ Architecture Overview <a name="architecture"></a>

```
User Request
     │
     ▼
FastAPI (main.py)
     │
     ├── POST /ingest ──────► parse.py → chunk.py → embed.py → vectorstore.py
     │                                                              │
     │                                                         hnswlib index
     │                                                         (disk persist)
     │
     ├── POST /chat ────────► embed query → vectorstore search → prompt.py
     │                                                              │
     │                                                         llm.py
     │                                                         │
     │                                                    ┌────┴────┐
     │                                                    │         │
     │                                               Groq API  Gemini Flash
     │                                               (primary) (fallback)
     │
     ├── GET /documents ────► vectorstore.list_documents()
     └── DELETE /documents/{id} ► vectorstore.delete_document()
```

**LLM Failover Flow:**
```
Call Groq → Fail? → Retry Groq (1s delay) → Fail? → Call Gemini Flash → Response
```

---

## 📁 Folder Structure <a name="folder-structure"></a>

```
rag-chatbot/
├── main.py                  # FastAPI app entry point
├── config.py                # Environment configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .env                     # YOUR secrets (never commit this!)
├── Dockerfile               # Docker image with model pre-bake
├── .dockerignore
├── render.yaml              # Render deployment config
├── railway.toml             # Railway deployment config
├── pytest.ini               # Test configuration
├── postman_collection.json  # API test collection
│
├── routers/
│   ├── __init__.py
│   ├── ingest.py            # POST /ingest, GET /ingest/status/{id}
│   ├── chat.py              # POST /chat
│   └── documents.py         # GET /documents, DELETE /documents/{id}
│
├── services/
│   ├── __init__.py
│   ├── parse.py             # PDF, XLSX, DOCX parsers
│   ├── chunk.py             # Text chunking with overlap
│   ├── embed.py             # sentence-transformers embeddings
│   ├── vectorstore.py       # hnswlib vector store + persistence
│   ├── retriever.py         # Query → ANN search → context
│   ├── prompt.py            # Prompt builder
│   └── llm.py               # Groq client + Gemini fallback
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── test_ingest.py       # Ingestion endpoint tests
│   ├── test_chat.py         # Chat endpoint tests
│   ├── test_documents.py    # Document management tests
│   └── test_parsers.py      # Unit tests for parsers/chunker
│
├── data/                    # Auto-created: hnswlib index + metadata
└── logs/                    # Auto-created: app logs
```

---

## 📦 Prerequisites <a name="prerequisites"></a>

- **Python 3.10 or 3.11** (3.12 not fully tested with hnswlib)
- **pip** (Python package manager)
- **Git**
- **Docker** (optional, for containerized deployment)

Check your Python version:
```bash
python --version
# Should show Python 3.10.x or 3.11.x
```

---

## 🔑 Get API Keys <a name="get-api-keys"></a>

You need **two free API keys**:

### Groq API Key (Primary LLM — FREE)
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up / Log in
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key — it starts with `gsk_...`

### Google Gemini API Key (Fallback LLM — FREE)
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key — it starts with `AIza...`

---

## 🚀 Local Setup & Run <a name="local-setup"></a>

### Step 1: Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd rag-chatbot

# Or just download and unzip, then:
cd rag-chatbot
```

### Step 2: Create a virtual environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ This installs PyTorch + sentence-transformers (~500MB). It may take 2-5 minutes.

### Step 4: Set up environment variables

```bash
# Copy the template
cp .env.example .env

# Open .env in a text editor and fill in your API keys:
# GROQ_API_KEY=gsk_your_actual_groq_key
# GEMINI_API_KEY=AIza_your_actual_gemini_key
```

**On Mac/Linux:**
```bash
nano .env
# Edit the file, save with Ctrl+X, then Y, then Enter
```

**On Windows:**
```bash
notepad .env
```

Your `.env` file should look like:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx
STORAGE_PATH=./data
MAX_FILE_SIZE_MB=10
MAX_DOCS=100
```

### Step 5: Create required directories

```bash
mkdir -p data logs
```

### Step 6: Run the server

```bash
python main.py
```

You should see output like:
```
2024-01-01 12:00:00 | INFO | 🚀 Starting RAG Chatbot API v1.0.0
2024-01-01 12:00:00 | INFO | 📦 Loading sentence-transformers model...
2024-01-01 12:00:15 | INFO | ✅ Embedding model loaded
2024-01-01 12:00:15 | INFO | 📁 Initializing vector store at: ./data
2024-01-01 12:00:15 | INFO | ✅ App started in 15.2s — Ready to serve requests
```

> ℹ️ First startup takes 10-15s to download the embedding model (~90MB). Subsequent starts are ~2s.

### Step 7: Verify it's running

Open your browser: **[http://localhost:8000/health](http://localhost:8000/health)**

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vector_store": {"total_documents": 0, "total_chunks": 0}
}
```

Interactive API docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📡 API Endpoints <a name="api-endpoints"></a>

### POST /ingest — Upload a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "queued",
  "filename": "document.pdf",
  "message": "Document queued for processing. Poll /ingest/status/{job_id}"
}
```

### GET /ingest/status/{job_id} — Check Ingestion Progress

```bash
curl http://localhost:8000/ingest/status/abc-123-def
```

**Response (completed):**
```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "filename": "document.pdf",
  "doc_id": "xyz-789",
  "chunks_added": 42,
  "message": "Successfully ingested 42 chunks"
}
```

### POST /chat — Chat with Documents

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?"}'
```

**Response:**
```json
{
  "answer": "The main findings are... [Source 1]",
  "citations": [
    {
      "source_num": 1,
      "document": "document.pdf",
      "page": "3",
      "relevance_score": 0.924
    }
  ],
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "fallback_used": false,
  "latency_ms": 850,
  "chunks_retrieved": 5
}
```

**With conversation history:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me more about that",
    "history": [
      {"role": "user", "content": "What are the findings?"},
      {"role": "assistant", "content": "The findings show..."}
    ]
  }'
```

### GET /documents — List All Documents

```bash
curl http://localhost:8000/documents
```

### DELETE /documents/{doc_id} — Delete a Document

```bash
curl -X DELETE http://localhost:8000/documents/xyz-789
```

---

## 🧪 Testing Guide <a name="testing"></a>

### Run All Tests

```bash
# Make sure your virtual environment is active
pytest tests/ -v
```

### Run Specific Test Files

```bash
pytest tests/test_ingest.py -v       # Ingestion tests
pytest tests/test_chat.py -v         # Chat tests
pytest tests/test_documents.py -v    # Document management tests
pytest tests/test_parsers.py -v      # Parser unit tests
```

### Test with a Real File (Manual)

```bash
# Start the server in one terminal
python main.py

# In another terminal:
# 1. Upload a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf" | python -m json.tool

# 2. Copy the job_id from the response, then poll:
curl http://localhost:8000/ingest/status/YOUR_JOB_ID | python -m json.tool

# 3. Wait for status: "completed", then chat:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize this document"}' | python -m json.tool
```

### Use the Postman Collection

1. Download [Postman](https://www.postman.com/downloads/)
2. Click **Import** → upload `postman_collection.json`
3. Set the `base_url` variable to `http://localhost:8000`
4. Run requests in order (1 → 6)

---

## 🐳 Docker Setup <a name="docker"></a>

### Build the Image

```bash
docker build -t rag-chatbot .
```

> ⚠️ First build takes 5-10 minutes (downloads PyTorch + pre-bakes the embedding model).

### Run the Container

```bash
docker run -d \
  --name rag-chatbot \
  -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_key \
  -e GEMINI_API_KEY=your_gemini_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  rag-chatbot
```

### Check Logs

```bash
docker logs rag-chatbot -f
```

### Stop the Container

```bash
docker stop rag-chatbot
docker rm rag-chatbot
```

---

## 🚀 Deploy to Render (Free Tier) <a name="deploy-render"></a>

Render is recommended — it offers a persistent disk on the free tier.

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial RAG chatbot"

# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/rag-chatbot.git
git push -u origin main
```

> ⚠️ Make sure `.env` is in `.gitignore` — NEVER commit your API keys!

### Step 2: Create .gitignore

```bash
# Create .gitignore if it doesn't exist:
cat > .gitignore << 'EOF'
.env
__pycache__/
*.pyc
*.pyo
data/
logs/
.venv/
venv/
*.egg-info/
.pytest_cache/
EOF
```

### Step 3: Deploy on Render

1. Go to [https://render.com](https://render.com) and sign up
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select your `rag-chatbot` repo
4. Configure:
   - **Name:** `rag-chatbot-api`
   - **Runtime:** `Docker`
   - **Plan:** `Free`
5. Click **"Advanced"** → **"Add Disk"**:
   - **Name:** `rag-data`
   - **Mount Path:** `/var/data`
   - **Size:** `1 GB`
6. Click **"Environment"** → Add variables:
   ```
   GROQ_API_KEY    = gsk_your_actual_key
   GEMINI_API_KEY  = AIza_your_actual_key
   STORAGE_PATH    = /var/data
   MAX_FILE_SIZE_MB = 10
   MAX_DOCS        = 100
   ```
7. Click **"Create Web Service"**
8. Wait ~5 minutes for first deploy

Your API will be at: `https://rag-chatbot-api.onrender.com`

---

## 🚆 Deploy to Railway (Alternative) <a name="deploy-railway"></a>

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### Step 2: Deploy

```bash
railway init        # Creates a new project
railway up          # Deploys your app
```

### Step 3: Set Environment Variables

```bash
railway variables set GROQ_API_KEY=gsk_your_key
railway variables set GEMINI_API_KEY=AIza_your_key
railway variables set STORAGE_PATH=/app/data
railway variables set MAX_FILE_SIZE_MB=10
railway variables set MAX_DOCS=100
```

### Step 4: Add Persistent Volume (Railway Dashboard)

1. Open your Railway project dashboard
2. Click your service → **"Volumes"**
3. Add volume: mount path `/app/data`

---

## ❄️ Cold Start Prevention <a name="cold-start"></a>

Free tier services sleep after 15-30 minutes of inactivity. The first request after sleep takes 20-30s.

### Solution: UptimeRobot (FREE)

1. Go to [https://uptimerobot.com](https://uptimerobot.com) and sign up free
2. Click **"Add New Monitor"**
3. Configure:
   - **Monitor Type:** HTTP(s)
   - **Friendly Name:** RAG Chatbot Ping
   - **URL:** `https://your-app.onrender.com/health`
   - **Monitoring Interval:** Every 5 minutes
4. Click **"Create Monitor"**

This pings your `/health` endpoint every 5 minutes, keeping it warm.

---

## 🔧 Troubleshooting <a name="troubleshooting"></a>

### "ModuleNotFoundError: No module named 'xxx'"
```bash
# Make sure your virtual environment is active:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies:
pip install -r requirements.txt
```

### "GROQ_API_KEY not configured"
```bash
# Check your .env file exists and has the key:
cat .env
# Should show: GROQ_API_KEY=gsk_xxx...
```

### Embedding model download fails
```bash
# Try manual download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# This will show progress bar and any errors
```

### "No documents ingested yet" on /chat
- You must ingest at least one document first
- Check ingestion job status: `GET /ingest/status/{job_id}`
- Job status "failed" means parsing failed — check the `message` field

### Groq 429 Rate Limit Error
- Groq free tier: 30 req/min, 6000 tokens/min
- The system automatically falls back to Gemini when Groq rate-limits
- If Gemini also fails: add delay between requests or upgrade Groq plan

### hnswlib import error on Apple Silicon (M1/M2)
```bash
# Install with specific flags:
ARCHFLAGS="-arch arm64" pip install hnswlib
```

### Docker build fails at PyTorch installation
```bash
# PyTorch is large (~2GB). Ensure you have:
# - At least 4GB free disk space
# - Stable internet connection
# Retry: docker build --no-cache -t rag-chatbot .
```

### Large PDF takes too long
- Files over 10MB are rejected (configurable via MAX_FILE_SIZE_MB)
- Ingestion runs in the background — poll /ingest/status/{job_id}
- Very large PDFs (50+ pages) may take 30-60s in the background

---

## ✅ Production Checklist <a name="production-checklist"></a>

Before going live, verify each item:

**Security**
- [ ] `.env` is NOT committed to git (check `.gitignore`)
- [ ] CORS origins restricted to your frontend domain (in `main.py`)
- [ ] API keys are set as environment variables, NOT hardcoded

**Configuration**
- [ ] `GROQ_API_KEY` is set and valid
- [ ] `GEMINI_API_KEY` is set and valid
- [ ] `STORAGE_PATH` points to persistent disk (`/var/data` on Render)
- [ ] `DEBUG=false` in production

**Deployment**
- [ ] App starts without errors (check logs)
- [ ] `/health` returns `{"status": "healthy", "model_loaded": true}`
- [ ] `model_loaded: true` in health check (model pre-loaded at startup)
- [ ] Persistent disk mounted and accessible (index survives restarts)

**Functionality**
- [ ] Can upload PDF via POST /ingest
- [ ] Can upload DOCX via POST /ingest
- [ ] Can upload XLSX via POST /ingest
- [ ] Ingestion job reaches "completed" status
- [ ] POST /chat returns an answer with citations
- [ ] Groq → Gemini failover tested (temporarily use wrong Groq key)
- [ ] GET /documents lists ingested docs
- [ ] DELETE /documents/{id} removes a document

**Performance**
- [ ] Warm response time under 4 seconds
- [ ] UptimeRobot configured to ping /health every 5 minutes
- [ ] Tested with 10+ documents concurrently

**Monitoring**
- [ ] Logs visible in hosting platform dashboard
- [ ] Provider used (groq/gemini) and `fallback_used` field in responses

---

## 📊 Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Query embedding | ~15ms | MiniLM-L6-v2 on CPU |
| Vector search | ~5ms | hnswlib kNN |
| Groq LLM (TTFT) | ~300ms | Llama 3.1 8B |
| Total warm request | < 1.5s | Typical |
| Total cold start | 20-30s | After idle on free tier |

---

## 💡 Quick Reference Commands

```bash
# Start server locally
python main.py

# Run all tests
pytest tests/ -v

# Build Docker image
docker build -t rag-chatbot .

# Run with Docker
docker run -p 8000:8000 --env-file .env rag-chatbot

# Check health
curl http://localhost:8000/health

# Ingest a file
curl -X POST http://localhost:8000/ingest -F "file=@doc.pdf"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?"}'
```

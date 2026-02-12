# SEFS - Semantic File System

AI-powered file organization using vector embeddings and LLM clustering.

## Architecture

- **Embeddings**: MiniLM-L6-v2 (384-dim vectors, GPU-accelerated)
- **Vector Search**: FAISS for fast similarity search
- **LLM**: Llama 3.3 70B (Cerebras) for folder naming & summaries
- **Clustering**: Connected components based on chunk similarity
- **Backend**: FastAPI + SQLite + FAISS
- **Frontend**: React + Force Graph

## Demo

<video src="assets/IntelligentFileManager.mp4" controls="controls" style="max-width: 100%;">
</video>

## Screenshots

### System Architecture
![System Architecture](assets/SEFS_System_Architecture.jpeg)

### User Interface
![Dashboard UI](assets/UI.jpeg)

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python start.bat  # or: uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## How It Works

### 1. File Processing
```
File → Chunk (512 tokens) → Embed (MiniLM) → Store (SQLite + FAISS)
```

### 2. Clustering
```
All chunks → FAISS similarity search → Connected components → Clusters
```

### 3. Folder Naming
```
Cluster formed → Sample 3 files → LLM generates name → Folder created
```

### 4. File Summaries (Lazy)
```
User clicks file → Check cache → Generate with LLM (map-reduce) → Cache 24h
```

## Key Features

- **Chunk-level change detection**: Only re-embed modified chunks
- **FAISS indexing**: Fast similarity search across millions of vectors
- **LLM caching**: Folder names cached indefinitely, summaries for 24h
- **GPU acceleration**: Embeddings run on CUDA if available
- **Real-time updates**: WebSocket broadcasts file movements
- **Debounced clustering**: Batches changes to avoid constant re-clustering

## API Endpoints

- `GET /files` - File structure
- `GET /stats` - System metrics
- `GET /system-info` - Configuration
- `POST /upload` - Upload & auto-categorize
- `POST /recluster` - Force re-clustering
- `GET /file-summary/{filename}` - Get/generate summary
- `GET /similar/{filename}` - Find similar files
- `GET /clusters` - Current cluster info

## Configuration

Edit `backend/config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b"
CHUNK_SIZE = 512
SIMILARITY_THRESHOLD = 0.7
USE_GPU = True
```

## Troubleshooting

### Backend won't start
```bash
cd backend
python -c "from semantic_engine import semantic_service"
python -c "from main import app"
```

### FAISS GPU issues
Install CPU version:
```bash
pip uninstall faiss-gpu
pip install faiss-cpu
```

### Frontend can't connect
Check backend is running on http://localhost:8000
Check CORS is enabled in main.py

## Performance

- **Embedding**: ~1000 chunks/sec on GPU
- **Clustering**: ~100 files in <1 second
- **LLM calls**: ~5 per cluster (not per file)
- **Summary generation**: ~2-5 seconds per file (cached)

## File Structure

```
backend/
  ├── main.py              # FastAPI app
  ├── semantic_engine.py   # Core vector/LLM logic
  ├── watcher.py           # File system watcher
  ├── config.py            # Settings
  ├── vectors.db           # SQLite vector store
  └── faiss_index.bin      # FAISS index

frontend/
  ├── src/
  │   ├── Dashboard.jsx    # Main UI
  │   └── main.jsx
  └── package.json

monitored_root/          # Files to organize
```

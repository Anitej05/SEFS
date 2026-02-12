# Quick Start Guide

## Step 1: Install Dependencies

### Backend
```bash
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 2: Start Backend

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or use the batch file:
```bash
cd backend
start.bat
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
INFO:     Watcher started on D:/SEFS/monitored_root (Recursive Mode)
```

## Step 3: Start Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

## Step 4: Open Browser

Navigate to: http://localhost:5173

You should see the SEFS dashboard with:
- File graph visualization
- System metrics
- Semantic layers view
- Knowledge base explorer

## Step 5: Test It

### Upload a file
1. Click the Upload button
2. Select a text file
3. Watch it get automatically categorized

### Or drop files directly
1. Navigate to `D:/SEFS/monitored_root` (or your configured path)
2. Drop some files there
3. Watch the dashboard update in real-time

## Troubleshooting

### Backend Issues

**"Module not found" errors**
```bash
cd backend
pip install -r requirements.txt
```

**"Port 8000 already in use"**
```bash
# Kill the process or use a different port
python -m uvicorn main:app --reload --port 8001
# Then update frontend/src/Dashboard.jsx: SEFS_API = "http://localhost:8001"
```

**"CUDA not available" warnings**
- This is fine! The system will use CPU for embeddings
- To use GPU, install: `pip install faiss-gpu`

### Frontend Issues

**"Cannot connect to backend"**
- Make sure backend is running on port 8000
- Check browser console for CORS errors
- Verify `SEFS_API` in Dashboard.jsx matches backend URL

**Graph not showing**
- Check browser console for errors
- Make sure files exist in monitored_root
- Try refreshing the page

**No metrics showing**
- Backend might not be fully started
- Check backend logs for errors
- Try accessing http://localhost:8000/stats directly

### Common Issues

**Empty dashboard**
- No files in monitored_root yet
- Upload a file or drop files into the folder

**Files not being categorized**
- Check backend logs for errors
- LLM API might be rate-limited (uses Cerebras)
- Try manual re-clustering: click "Recluster" button

**Slow performance**
- First run downloads the embedding model (~80MB)
- Subsequent runs are much faster
- GPU acceleration helps significantly

## Next Steps

1. **Configure**: Edit `backend/config.py` to customize settings
2. **Add files**: Drop files into monitored_root
3. **Explore**: Click files to view summaries
4. **Organize**: Let the AI cluster your files semantically

## Architecture Overview

```
User drops file → Watcher detects → Chunk & Embed → Store in FAISS
                                                          ↓
                                                    Cluster files
                                                          ↓
                                                    LLM names folder
                                                          ↓
                                                    Move to folder
                                                          ↓
User clicks file → Generate summary (lazy) → Cache → Display
```

## Performance Tips

1. **Use GPU**: Install `faiss-gpu` and `torch` with CUDA support
2. **Batch uploads**: Upload multiple files at once
3. **Adjust chunk size**: Smaller chunks = more granular but slower
4. **Cache summaries**: Summaries are cached for 24h
5. **Debounce clustering**: System waits 5s before re-clustering

## Support

Check the logs:
- Backend: Terminal where uvicorn is running
- Frontend: Browser console (F12)
- System: Dashboard → Stats tab

For issues, check:
1. Backend is running (http://localhost:8000)
2. Frontend is running (http://localhost:5173)
3. monitored_root folder exists
4. Dependencies are installed

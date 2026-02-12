import os
import json
import time
import shutil
import logging
import tempfile
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from semantic_engine import get_semantic_service
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SEFS API")
startup_time = time.time()

# Get semantic service lazily
semantic_service = None

def get_service():
    global semantic_service
    if semantic_service is None:
        semantic_service = get_semantic_service()
        semantic_service.log_callback = lambda data: None  # Will be set when ws connects
        
        # Initialize TTS service
        from tts_service import TTSService
        semantic_service.tts_service = TTSService()
    return semantic_service


class StatsTracker:
    def __init__(self):
        self.files_processed = 0
        self.files_moved = 0
        self.chunks_embedded = 0
        self.clusters_created = 0
        self.errors = 0
        self.last_activity = None
    
    def record_process(self, chunks: int = 0):
        self.files_processed += 1
        self.chunks_embedded += chunks
        self.last_activity = datetime.now().isoformat()
    
    def record_move(self):
        self.files_moved += 1
        self.last_activity = datetime.now().isoformat()
    
    def record_cluster(self):
        self.clusters_created += 1
        self.last_activity = datetime.now().isoformat()
    
    def record_error(self):
        self.errors += 1
        self.last_activity = datetime.now().isoformat()

stats_tracker = StatsTracker()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def list_connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
file_watcher = None


def on_file_update(data):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(manager.broadcast(data))


@app.on_event("startup")
async def startup_event():
    # Ensure directories exist
    os.makedirs(settings.UPLOAD_STORAGE, exist_ok=True)
    os.makedirs(settings.MONITORED_ROOT, exist_ok=True)
    
    # Start file watcher for OS→SEFS bidirectional sync
    from watcher import FileWatcher
    global file_watcher
    file_watcher = FileWatcher(update_callback=on_file_update)
    file_watcher.start()
    
    logger.info("SEFS Backend started (Bidirectional Sync Mode)")
    get_service().log_event("system", message="SEFS Semantic Intelligence System Online — Bidirectional Sync Active")

@app.on_event("shutdown")
async def shutdown_event():
    global file_watcher
    if file_watcher:
        file_watcher.stop()
        logger.info("File watcher stopped")


@app.get("/")
async def root():
    return {"status": "SEFS Backend Running", "mode": "upload-driven"}


@app.get("/system-info")
async def system_info():
    uptime = int(time.time() - startup_time)
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "upload_storage": settings.UPLOAD_STORAGE,
        "status": "active",
        "uptime_seconds": uptime,
        "websocket_connections": len(manager.active_connections),
    }


@app.get("/logs")
async def get_logs():
    """Return recent logs. Logs are primarily delivered via WebSocket."""
    return []


@app.get("/search")
async def search_files(q: str, limit: int = 10):
    """Semantic search across all indexed files."""
    try:
        service = get_service()
        results = service.semantic_search(q, limit=limit)
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def get_files():
    """Return DB-driven virtual file structure based on cluster assignments."""
    service = get_service()
    
    # Get clusters from DB
    clusters = service.vector_store.get_clusters()
    
    # Get all file paths from DB
    all_file_paths = service.vector_store.get_all_file_paths()
    
    # Track which files are in clusters
    clustered_files = set()
    structure = []
    
    for cluster_name, data in clusters.items():
        folder_files = []
        for fp, score in data["members"]:
            clustered_files.add(fp)
            filename = os.path.basename(fp)
            file_info = {"name": filename, "score": round(score, 4)}
            
            if os.path.exists(fp):
                stat = os.stat(fp)
                file_info["size"] = stat.st_size
                file_info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            else:
                file_info["size"] = 0
                file_info["modified"] = None
            
            folder_files.append(file_info)
        
        structure.append({
            "name": cluster_name,
            "type": "folder",
            "files": [ff["name"] for ff in folder_files],
            "file_details": folder_files,
            "file_count": len(folder_files)
        })
    
    # Add unclustered files as root-level items
    for fp in all_file_paths:
        if fp not in clustered_files:
            filename = os.path.basename(fp)
            file_info = {
                "name": filename,
                "type": "file",
                "folder": None,
            }
            if os.path.exists(fp):
                stat = os.stat(fp)
                file_info["size"] = stat.st_size
                file_info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            else:
                file_info["size"] = 0
                file_info["modified"] = None
            structure.append(file_info)
    
    return structure


@app.get("/stats")
async def get_stats():
    """Return DB-driven statistics."""
    service = get_service()
    
    clusters = service.vector_store.get_clusters()
    file_count = service.vector_store.get_document_count()
    chunk_count = service.vector_store.get_chunk_count()
    
    # Calculate total size from upload storage
    total_size = 0
    upload_dir = settings.UPLOAD_STORAGE
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            fp = os.path.join(upload_dir, f)
            if os.path.isfile(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    
    uptime = int(time.time() - startup_time)
    return {
        "folders": len(clusters),
        "files": file_count,
        "chunks": chunk_count,
        "total_size_bytes": total_size,
        "uptime_seconds": uptime,
        "files_processed": stats_tracker.files_processed,
        "files_moved": stats_tracker.files_moved,
        "chunks_embedded": stats_tracker.chunks_embedded,
        "clusters_created": stats_tracker.clusters_created,
        "errors": stats_tracker.errors,
        "last_activity": stats_tracker.last_activity,
        "websocket_connections": len(manager.active_connections),
    }


def _resolve_file_path(filename: str, folder: str = None) -> str:
    """Resolve a filename (with optional virtual folder) to its actual path.
    
    Looks up the file in the DB by matching the basename, optionally filtered by cluster.
    Falls back to checking upload storage directly.
    """
    service = get_service()
    
    # If folder is given, look it up from cluster membership
    if folder and folder not in (None, "Root", "null"):
        clusters = service.vector_store.get_clusters()
        if folder in clusters:
            for fp, score in clusters[folder]["members"]:
                if os.path.basename(fp) == filename:
                    return fp
    
    # Search all documents for a matching basename
    all_paths = service.vector_store.get_all_file_paths()
    for fp in all_paths:
        if os.path.basename(fp) == filename:
            return fp
    
    # Fallback: check upload storage directly
    direct_path = os.path.join(settings.UPLOAD_STORAGE, filename)
    if os.path.exists(direct_path):
        return direct_path
    
    return None


@app.get("/file/{folder}/{filename}")
async def get_file_content(folder: str, filename: str):
    file_path = _resolve_file_path(filename, folder)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    stat = os.stat(file_path)
    content = None
    is_text = False
    text_extensions = {'.txt', '.py', '.js', '.jsx', '.css', '.html', '.json', '.md', '.csv', '.log', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.toml', '.sh', '.bat', '.ps1'}
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in text_extensions and stat.st_size < 50000:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            is_text = True
        except Exception:
            content = None
    
    return {
        "name": filename,
        "folder": folder,
        "path": file_path,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "extension": ext,
        "is_text": is_text,
        "content": content
    }


@app.get("/file-root/{filename}")
async def get_root_file_content(filename: str):
    file_path = _resolve_file_path(filename)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    stat = os.stat(file_path)
    content = None
    is_text = False
    text_extensions = {'.txt', '.py', '.js', '.jsx', '.css', '.html', '.json', '.md', '.csv', '.log', '.xml', '.yaml', '.yml'}
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in text_extensions and stat.st_size < 50000:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            is_text = True
        except Exception:
            content = None
    
    return {
        "name": filename,
        "folder": None,
        "path": file_path,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": ext,
        "is_text": is_text,
        "content": content
    }


class MoveRequest(BaseModel):
    filename: str
    source_folder: str | None = None
    target_folder: str

@app.post("/move")
async def move_file(req: MoveRequest):
    """Move a file to a different semantic cluster."""
    svc = get_service()
    
    # Find the full file path by filename
    all_paths = svc.vector_store.get_all_file_paths()
    file_path = None
    for fp in all_paths:
        if os.path.basename(fp) == req.filename:
            file_path = fp
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail=f"File '{req.filename}' not found")
    
    # Get current cluster for logging
    old_cluster = svc.vector_store.get_cluster_for_file(file_path)
    old_name = old_cluster[0] if old_cluster else "Unknown"
    
    # Actually reassign the file
    success = svc.vector_store.reassign_file_to_cluster(file_path, req.target_folder)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reassign file")
    
    # Record the move for entropy tracking
    svc.vector_store.save_file_move(file_path, old_name, req.target_folder)
    svc.log_event("move", file=req.filename, folder=req.target_folder, message=f"Moved {req.filename}: {old_name} → {req.target_folder}")
    stats_tracker.record_move()
    
    # Notify WebSocket clients
    await manager.broadcast({
        "type": "file_moved",
        "file": req.filename,
        "from": old_name,
        "to": req.target_folder
    })
    
    return {"status": "ok", "file": req.filename, "from": old_name, "destination": req.target_folder}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), auto_categorize: bool = True):
    """Upload a file, process it, and auto-cluster."""
    os.makedirs(settings.MONITORED_ROOT, exist_ok=True)
    
    # Read file content
    content = await file.read()
    
    # Check if it's a zip file
    if file.filename.lower().endswith('.zip'):
        import zipfile
        import io
        
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                # Suppress watcher events during extraction to prevent race conditions
                extract_path = settings.MONITORED_ROOT
                extracted_names = zip_ref.namelist()
                if file_watcher:
                    for name in extracted_names:
                        if not name.endswith('/'):
                            file_watcher.handler.suppress_path(
                                os.path.join(extract_path, name), duration=30.0
                            )
                zip_ref.extractall(extract_path)
                
                # Get list of extracted files
                extracted_files = zip_ref.namelist()
                logger.info(f"Extracted {len(extracted_files)} files from {file.filename}")
                
                result = {
                    "status": "extracted",
                    "file": file.filename,
                    "extracted_count": len(extracted_files),
                    "size": len(content)
                }
                
                if auto_categorize:
                    processed = 0
                    for extracted_file in extracted_files:
                        if not extracted_file.endswith('/'):
                            extracted_path = os.path.join(extract_path, extracted_file)
                            if os.path.exists(extracted_path):
                                try:
                                    result_processed = get_service().process_file(extracted_path)
                                    if "error" not in result_processed:
                                        processed += 1
                                        stats_tracker.record_process(result_processed.get("chunks_processed", 0))
                                except Exception as e:
                                    logger.error(f"Failed to process {extracted_file}: {e}")
                    
                    # Trigger re-clustering
                    if processed > 0:
                        clusters = get_service().recluster_all()
                        stats_tracker.record_cluster()
                        result["processed"] = processed
                        result["categorized"] = True
                
                await manager.broadcast({
                    "type": "upload",
                    "file": file.filename,
                    "extracted": len(extracted_files)
                })
                
                return result
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")
    
    # Regular file upload - save to monitored root for processing
    file_path = os.path.join(settings.MONITORED_ROOT, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Suppress watcher for this path to prevent race condition
    if file_watcher:
        file_watcher.handler.suppress_path(file_path, duration=30.0)
    with open(file_path, "wb") as f:
        f.write(content)
    
    logger.info(f"Uploaded file: {file.filename} ({len(content)} bytes)")
    
    result = {"status": "uploaded", "file": file.filename, "size": len(content)}
    
    if auto_categorize:
        try:
            result_processed = get_service().process_file(file_path)
            if "error" not in result_processed:
                stats_tracker.record_process(result_processed.get("chunks_processed", 0))
                result["chunks_processed"] = result_processed.get("chunks_processed")
                
                # Trigger re-clustering
                clusters = get_service().recluster_all()
                stats_tracker.record_cluster()
                result["categorized"] = True
                result["clusters"] = clusters.get("clusters", [])
                
                await manager.broadcast({
                    "type": "upload",
                    "file": file.filename,
                    "chunks": result_processed.get("chunks_processed", 0)
                })
        except Exception as e:
            logger.error(f"Auto-categorize failed: {e}")
            stats_tracker.record_error()
            result["categorize_error"] = str(e)
    
    return result


@app.post("/upload-bulk")
async def upload_bulk(files: list[UploadFile] = File(...), auto_categorize: bool = True):
    """Upload multiple files at once."""
    os.makedirs(settings.MONITORED_ROOT, exist_ok=True)
    results = []
    
    for file in files:
        try:
            file_path = os.path.join(settings.MONITORED_ROOT, file.filename)
            content = await file.read()
            
            # Suppress watcher during write
            if file_watcher:
                file_watcher.handler.suppress_path(file_path, duration=30.0)
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Uploaded: {file.filename} ({len(content)} bytes)")
            results.append({"file": file.filename, "status": "uploaded", "size": len(content)})
            
            if auto_categorize:
                try:
                    result_processed = get_service().process_file(file_path)
                    if "error" not in result_processed:
                        stats_tracker.record_process(result_processed.get("chunks_processed", 0))
                except Exception as e:
                    logger.error(f"Failed to process {file.filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            results.append({"file": file.filename, "status": "failed", "error": str(e)})
    
    # Trigger re-clustering after all uploads
    if auto_categorize and any(r["status"] == "uploaded" for r in results):
        try:
            clusters = get_service().recluster_all()
            stats_tracker.record_cluster()
        except Exception as e:
            logger.error(f"Re-clustering failed: {e}")
    
    await manager.broadcast({
        "type": "bulk_upload",
        "count": len(results)
    })
    
    return {
        "status": "completed",
        "total": len(files),
        "successful": len([r for r in results if r["status"] == "uploaded"]),
        "results": results
    }


class DeleteRequest(BaseModel):
    filename: str
    folder: str | None = None

@app.post("/delete-file")
async def delete_file(req: DeleteRequest):
    """Delete a file from DB and upload storage."""
    file_path = _resolve_file_path(req.filename, req.folder)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found in database")
    
    # Delete from DB
    get_service().delete_file(file_path)
    
    # Delete physical file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Re-cluster after deletion
    try:
        get_service().recluster_all()
    except Exception as e:
        logger.warning(f"Recluster after delete failed: {e}")
    
    logger.info(f"Deleted file: {req.filename}")
    
    await manager.broadcast({
        "type": "delete",
        "file": req.filename,
        "folder": req.folder
    })
    
    return {"status": "deleted", "file": req.filename}


@app.post("/clear")
async def clear_all():
    """Clear all data: DB, FAISS index, and uploaded files."""
    try:
        result = get_service().clear_all()
        stats_tracker.__init__()  # Reset stats
        
        await manager.broadcast({
            "type": "clear",
            "message": "All data cleared"
        })
        
        return result
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-folder/{folder_name}")
async def create_folder(folder_name: str):
    """Folders are now virtual (cluster-based). This is a no-op kept for compatibility."""
    return {"status": "ok", "folder": folder_name, "note": "Virtual folders are auto-created by clustering"}


@app.delete("/folder/{folder_name}")
async def delete_folder(folder_name: str):
    """Folders are virtual. This is a no-op kept for compatibility."""
    return {"status": "ok", "folder": folder_name, "note": "Virtual folders are auto-managed by clustering"}


class ScanDirectoryRequest(BaseModel):
    path: str

@app.post("/scan-directory")
async def scan_directory(req: ScanDirectoryRequest):
    """Scan a local directory and organize its files IN-PLACE into semantic clusters.
    This modifies the actual folder structure at the given path."""
    directory = os.path.abspath(req.path)
    
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.path}")
    
    try:
        logger.info(f"Starting in-place organization of: {directory}")
        result = get_service().scan_and_organize_directory(directory)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Update the watcher to also monitor this directory
        global file_watcher
        if file_watcher:
            try:
                file_watcher.observer.schedule(
                    file_watcher.handler, directory, recursive=True
                )
                logger.info(f"Watcher now also monitoring: {directory}")
            except Exception as e:
                logger.warning(f"Could not add watcher for {directory}: {e}")
        
        stats_tracker.record_cluster()
        
        # Broadcast to frontend
        await manager.broadcast({
            "type": "scan_directory",
            "directory": directory,
            "clusters": result.get("clusters", []),
            "files_moved": result.get("files_moved", 0)
        })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan directory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recluster")
async def recluster():
    """Force re-clustering of all documents."""
    try:
        result = get_service().recluster_all()
        stats_tracker.record_cluster()
        
        await manager.broadcast({
            "type": "recluster",
            "clusters": result.get("clusters", []),
        })
        
        return {"status": "ok", "clusters": result.get("clusters", [])}
    except Exception as e:
        logger.error(f"Recluster failed: {e}")
        stats_tracker.record_error()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{filename}")
async def find_similar(filename: str, folder: str = None, limit: int = 5):
    """Find similar files based on Top-K chunk-pair similarity."""
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        similar = get_service().find_similar(file_path, limit=limit)
        return {
            "file": filename,
            "similar": [
                {"file": os.path.basename(fp), "similarity": round(sim, 4)}
                for fp, sim in similar
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/file-chunks/{filename}")
async def get_file_chunks(filename: str, folder: str = None):
    """Get chunks for a file."""
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        chunks = get_service().get_file_chunks(file_path)
        return {
            "file": filename,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/file-summary/{filename}")
async def get_file_summary(filename: str, folder: str = None):
    """Get or generate summary for a file."""
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        summary = await get_service().get_file_summary(file_path)
        return {
            "file": filename,
            "summary": summary
        }
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="File not found (likely moved during processing)")
    except Exception as e:
        logger.error(f"File summary error for {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read-summary/{filename}")
async def read_summary(filename: str, folder: str = None):
    """Generate and stream audio for file summary."""
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        service = get_service()
        
        # Get summary text first
        summary_text = await service.get_file_summary(file_path)
        if not summary_text or "[File not found" in summary_text:
             raise HTTPException(status_code=404, detail="Summary not available")

        # Generate audio
        file_hash = service._hash_file(file_path)
        audio_path = service.tts_service.generate_audio(summary_text, file_hash)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Audio generation failed")
            
        return FileResponse(audio_path, media_type="audio/wav", filename=f"summary_{filename}.wav")

    except Exception as e:
        logger.error(f"TTS error for {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters")
async def get_clusters():
    """Get current cluster information."""
    try:
        clusters = get_service().vector_store.get_clusters()
        return {
            "clusters": [
                {
                    "name": name,
                    "member_count": data["member_count"],
                    "members": [{"file": os.path.basename(fp), "score": round(score, 4)} for fp, score in data["members"]]
                }
                for name, data in clusters.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download")
async def download_file(filename: str, folder: str = None):
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)


@app.get("/preview")
async def preview_file(filename: str, folder: str = None):
    """Serve a file inline for iframe/browser preview (no forced download)."""
    file_path = _resolve_file_path(filename, folder)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    import mimetypes
    media_type, _ = mimetypes.guess_type(file_path)
    media_type = media_type or "application/octet-stream"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Content-Disposition": "inline"}
    )


# ─── Standout Feature Endpoints ───────────────────────────────────

class NLCommandRequest(BaseModel):
    command: str

@app.post("/nl-command")
async def nl_command(req: NLCommandRequest):
    """Execute a natural language command using gpt-oss-120b."""
    try:
        result = await get_service().execute_nl_command(req.command)
        
        # Broadcast to frontend so UI refreshes after AI-triggered changes
        await manager.broadcast({
            "type": "nl_command",
            "command": req.command,
            "action": result.get("action", "unknown"),
            "message": result.get("message", "")
        })
        
        return result
    except Exception as e:
        logger.error(f"NL command error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster-history")
async def cluster_history(limit: int = 50):
    """Get cluster snapshot history for Time Travel."""
    try:
        snapshots = get_service().vector_store.get_cluster_snapshots(limit)
        return {"snapshots": snapshots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster-snapshot/{snapshot_id}")
async def cluster_snapshot(snapshot_id: int):
    """Get a specific cluster snapshot."""
    try:
        snapshot = get_service().vector_store.get_snapshot_by_id(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folder-summary/{folder_name}")
async def folder_summary(folder_name: str):
    """Get folder intelligence: description, coherence score, suggested name."""
    try:
        result = await get_service().get_folder_intelligence(folder_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/duplicates")
async def get_duplicates():
    """Find near-duplicate files."""
    try:
        duplicates = get_service().find_duplicates()
        return {"duplicates": duplicates, "count": len(duplicates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cross-edges")
async def get_cross_edges():
    """Get cross-cluster semantic relationships."""
    try:
        edges = get_service().get_cross_cluster_edges()
        return {"edges": edges, "count": len(edges)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvalMoveRequest(BaseModel):
    filename: str
    target_folder: str

@app.post("/evaluate-move")
async def evaluate_move_endpoint(req: EvalMoveRequest):
    """Evaluate the impact of moving a file to a different cluster."""
    try:
        result = get_service().evaluate_move(req.filename, req.target_folder)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entropy")
async def get_entropy():
    """Get file entropy/stability data for heatmap."""
    try:
        entropy = get_service().get_entropy_data()
        return {"entropy": entropy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export-report")
async def export_report():
    """Generate and download a semantic organization report."""
    try:
        report = get_service().generate_export_report()
        # Write to temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
        tmp.write(report)
        tmp.close()
        return FileResponse(tmp.name, filename="sefs_semantic_report.md", media_type="text/markdown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.list_connect(websocket)
    
    # Update service callback to use manager
    try:
        service = get_service()
        service.log_callback = lambda data: None  # Sync callback not needed, ws is async
    except Exception:
        pass
    
    # Send recent logs
    try:
        recent_logs = get_service().vector_store.get_recent_logs(20)
        for log in reversed(recent_logs):
            await websocket.send_json(log)
    except Exception as e:
        logger.error(f"Failed to send initial logs: {e}")
        
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

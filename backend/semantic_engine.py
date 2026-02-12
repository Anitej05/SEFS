import os
import shutil
import json
import time
import hashlib
import logging
import asyncio
import threading
import numpy as np
import faiss
import sqlite3
import contextlib
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

# ─── Global watcher handler reference for suppression ───
# Set by main.py or watcher.py at startup so that sync_file_to_disk
# can suppress watcher events for SEFS-triggered file moves.
_watcher_handler = None

def set_watcher_handler(handler):
    """Register the watcher handler for suppression during SEFS→OS sync."""
    global _watcher_handler
    _watcher_handler = handler

def _suppress_watcher_path(path: str):
    """Suppress watcher events for a path during SEFS-initiated moves."""
    if _watcher_handler and hasattr(_watcher_handler, 'suppress_path'):
        _watcher_handler.suppress_path(path)


class FileLockManager:
    def __init__(self):
        self._lock = threading.RLock()
    
    @contextlib.contextmanager
    def lock(self):
        with self._lock:
            yield


class VectorStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    last_modified REAL,
                    chunk_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    faiss_idx INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    centroid BLOB,
                    member_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_clusters (
                    document_id INTEGER NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    membership_score REAL DEFAULT 0.0,
                    PRIMARY KEY (document_id, cluster_id),
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    file_hash TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    expires_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    file TEXT,
                    folder TEXT,
                    message TEXT,
                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_created ON system_logs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_idx)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_clusters ON document_clusters(document_id)")
            # ─── Standout Feature Tables ───
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cluster_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot JSON NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    cluster_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    from_cluster TEXT,
                    to_cluster TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS folder_summaries (
                    cluster_name TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    coherence_score REAL DEFAULT 0.0,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            # Migration: add source_dir column if not present
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN source_dir TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
    
    def _get_source_dir(self, file_path: str) -> str | None:
        """Get the stored source directory for a file."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT source_dir FROM documents WHERE file_path = ?", (file_path,)
            ).fetchone()
            return row[0] if row and row[0] else None
    
    def set_source_dir(self, file_path: str, source_dir: str):
        """Set the source directory for a file (used for in-place organization)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE documents SET source_dir = ? WHERE file_path = ?",
                (source_dir, file_path)
            )
    
    def get_document(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT id, file_hash, last_modified, chunk_count FROM documents WHERE file_path = ?",
                (file_path,)
            ).fetchone()
    
    def upsert_document(self, file_path: str, file_hash: str, last_modified: float, chunk_count: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO documents (file_path, file_hash, last_modified, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    last_modified = excluded.last_modified,
                    chunk_count = excluded.chunk_count
            """, (file_path, file_hash, last_modified, chunk_count))
            return conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", (file_path,)
            ).fetchone()[0]
    
    def delete_document(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
    
    def get_chunks(self, document_id: int):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT chunk_index, content_hash, faiss_idx FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            ).fetchall()
    
    def save_chunks(self, document_id: int, chunks: list[tuple], embeddings: np.ndarray):
        with sqlite3.connect(self.db_path) as conn:
            for i, (content, content_hash) in enumerate(chunks):
                embedding_blob = embeddings[i].astype(np.float32).tobytes()
                conn.execute("""
                    INSERT INTO chunks (document_id, chunk_index, content, content_hash, embedding, faiss_idx)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, chunk_index) DO UPDATE SET
                        content = excluded.content,
                        content_hash = excluded.content_hash,
                        embedding = excluded.embedding,
                        faiss_idx = excluded.faiss_idx
                """, (document_id, i, content, content_hash, embedding_blob, -1))  # faiss_idx set later
    
    def update_faiss_indices(self, document_id: int, indices: list[int]):
        with sqlite3.connect(self.db_path) as conn:
            for i, idx in enumerate(indices):
                conn.execute(
                    "UPDATE chunks SET faiss_idx = ? WHERE document_id = ? AND chunk_index = ?",
                    (idx, document_id, i)
                )
    
    def delete_chunks_after(self, document_id: int, chunk_index: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM chunks WHERE document_id = ? AND chunk_index > ?",
                (document_id, chunk_index)
            )
    
    def get_all_document_chunks(self):
        """Returns list of (file_path, embedding_blob, content) for all chunks."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT d.file_path, c.embedding, c.content, c.chunk_index
                FROM chunks c JOIN documents d ON c.document_id = d.id
                ORDER BY d.file_path, c.chunk_index
            """).fetchall()
    
    def get_document_embeddings(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT c.chunk_index, c.embedding, c.content
                FROM chunks c JOIN documents d ON c.document_id = d.id
                WHERE d.file_path = ?
                ORDER BY c.chunk_index
            """, (file_path,)).fetchall()
        return rows
    
    def get_all_file_paths(self):
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT file_path FROM documents").fetchall()]
    
    def clear_all_data(self):
        """Delete all data from all tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM document_clusters")
            conn.execute("DELETE FROM clusters")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM summaries")
            conn.execute("DELETE FROM system_logs")
        logger.info("Cleared all data from VectorStore")
    
    def get_document_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    
    def get_chunk_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    
    def save_clusters(self, clusters: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM document_clusters")
            conn.execute("DELETE FROM clusters")
            
            for name, data in clusters.items():
                centroid = data["centroid"].astype(np.float32).tobytes() if data["centroid"] is not None else None
                conn.execute(
                    "INSERT INTO clusters (name, centroid, member_count) VALUES (?, ?, ?)",
                    (name, centroid, len(data["members"]))
                )
                cluster_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                
                for file_path, score in data["members"]:
                    doc_id = conn.execute(
                        "SELECT id FROM documents WHERE file_path = ?", (file_path,)
                    ).fetchone()
                    if doc_id:
                        conn.execute(
                            "INSERT INTO document_clusters (document_id, cluster_id, membership_score) VALUES (?, ?, ?)",
                            (doc_id[0], cluster_id, score)
                        )
    
    def get_clusters(self):
        with sqlite3.connect(self.db_path) as conn:
            clusters = {}
            rows = conn.execute("SELECT id, name, centroid, member_count FROM clusters").fetchall()
            for cid, name, centroid, count in rows:
                members = conn.execute("""
                    SELECT d.file_path, dc.membership_score
                    FROM document_clusters dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.cluster_id = ?
                """, (cid,)).fetchall()
                clusters[name] = {
                    "id": cid,
                    "centroid": np.frombuffer(centroid, dtype=np.float32) if centroid else None,
                    "member_count": count,
                    "members": members
                }
        return clusters
    
    def get_cluster_for_file(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT c.name, dc.membership_score
                FROM document_clusters dc
                JOIN clusters c ON dc.cluster_id = c.id
                JOIN documents d ON dc.document_id = d.id
                WHERE d.file_path = ?
                ORDER BY dc.membership_score DESC
                LIMIT 1
            """, (file_path,)).fetchone()
            return result

    def sync_file_to_disk(self, file_path: str, target_cluster_name: str, base_dir: str = None) -> str:
        """Physically move a file into <base_dir>/<cluster_name>/ and update DB path.
        If base_dir is None, auto-detects from the file's current location.
        Returns the new file path, or the original if move not needed/failed.
        This is the SEFS→OS direction of bidirectional sync."""
        filename = os.path.basename(file_path)
        
        # Determine base directory for cluster subfolders
        if base_dir is None:
            # Auto-detect: use the source directory the file lives in
            # If file is in a cluster subfolder already, go up one level to the root
            parent = os.path.dirname(file_path)
            grandparent = os.path.dirname(parent)
            # Check if parent looks like a cluster folder inside a known base
            if os.path.normpath(grandparent) == os.path.normpath(settings.MONITORED_ROOT):
                base_dir = settings.MONITORED_ROOT
            elif os.path.normpath(parent) == os.path.normpath(settings.MONITORED_ROOT):
                base_dir = settings.MONITORED_ROOT
            else:
                # In-place mode: file is in an external directory
                # Use the source_dir stored in DB, or fall back to parent
                source_dir = self._get_source_dir(file_path)
                base_dir = source_dir if source_dir else parent
        
        target_dir = os.path.join(base_dir, target_cluster_name)
        new_path = os.path.join(target_dir, filename)
        
        # Skip if already in the right place
        if os.path.normpath(file_path) == os.path.normpath(new_path):
            return file_path
        
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            # Handle name collision in target folder
            if os.path.exists(new_path) and os.path.normpath(file_path) != os.path.normpath(new_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1
            
            # Physically move the file
            if os.path.exists(file_path):
                # Suppress watcher events for both old and new paths to avoid re-processing
                _suppress_watcher_path(file_path)
                _suppress_watcher_path(new_path)
                
                shutil.move(file_path, new_path)
                logger.info(f"OS sync (SEFS→OS): {os.path.basename(file_path)} → {target_cluster_name}/")
                
                # Update DB file_path
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("UPDATE documents SET file_path = ? WHERE file_path = ?", (new_path, file_path))
                
                # Clean up old empty parent directory (if it was a cluster folder)
                old_dir = os.path.dirname(file_path)
                self._cleanup_empty_folder(old_dir)
                
                return new_path
            else:
                logger.warning(f"OS sync: source file not found: {file_path}")
                return file_path
                
        except Exception as e:
            logger.error(f"OS sync failed for {file_path} -> {new_path}: {e}")
            return file_path
    
    def _cleanup_empty_folder(self, folder_path: str):
        """Remove a folder if it's empty and is a cluster subfolder of MONITORED_ROOT."""
        try:
            root = os.path.normpath(settings.MONITORED_ROOT)
            folder_norm = os.path.normpath(folder_path)
            
            # Only clean up direct children of MONITORED_ROOT (cluster folders)
            if (os.path.isdir(folder_norm) and 
                os.path.dirname(folder_norm) == root and
                folder_norm != root):
                # Check if empty (no files, ignore hidden)
                contents = [f for f in os.listdir(folder_norm) if not f.startswith('.')]
                if not contents:
                    _suppress_watcher_path(folder_norm)
                    os.rmdir(folder_norm)
                    logger.info(f"Cleaned up empty cluster folder: {folder_norm}")
        except Exception as e:
            logger.debug(f"Folder cleanup skipped for {folder_path}: {e}")
    
    def reassign_file_to_cluster(self, file_path: str, target_cluster_name: str) -> bool:
        """Move a file from its current cluster to a target cluster. Creates the target if needed.
        Also physically moves the file on disk into MONITORED_ROOT/<cluster_name>/."""
        with sqlite3.connect(self.db_path) as conn:
            # Get document id
            doc = conn.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,)).fetchone()
            if not doc:
                return False
            doc_id = doc[0]

            # Get or create target cluster
            target = conn.execute("SELECT id FROM clusters WHERE name = ?", (target_cluster_name,)).fetchone()
            if not target:
                conn.execute("INSERT INTO clusters (name, centroid, member_count) VALUES (?, NULL, 0)", (target_cluster_name,))
                target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            else:
                target_id = target[0]

            # Get current cluster id and name (to clean up later)
            current = conn.execute("SELECT cluster_id FROM document_clusters WHERE document_id = ?", (doc_id,)).fetchone()
            old_cluster_id = current[0] if current else None

            # Remove old assignment
            conn.execute("DELETE FROM document_clusters WHERE document_id = ?", (doc_id,))

            # Insert new assignment
            conn.execute(
                "INSERT INTO document_clusters (document_id, cluster_id, membership_score) VALUES (?, ?, ?)",
                (doc_id, target_id, 1.0)
            )

            # Update member counts
            conn.execute("UPDATE clusters SET member_count = (SELECT COUNT(*) FROM document_clusters WHERE cluster_id = ?) WHERE id = ?", (target_id, target_id))
            if old_cluster_id:
                remaining = conn.execute("SELECT COUNT(*) FROM document_clusters WHERE cluster_id = ?", (old_cluster_id,)).fetchone()[0]
                if remaining == 0:
                    conn.execute("DELETE FROM clusters WHERE id = ?", (old_cluster_id,))
                else:
                    conn.execute("UPDATE clusters SET member_count = ? WHERE id = ?", (remaining, old_cluster_id))

        # Physically move the file on disk (outside the DB transaction)
        new_path = self.sync_file_to_disk(file_path, target_cluster_name)
        
        return True
    
    def save_summary(self, file_hash: str, summary: str):
        expires_at = time.time() + settings.SUMMARY_CACHE_TTL
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO summaries (file_hash, summary, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    summary = excluded.summary,
                    created_at = strftime('%s', 'now'),
                    expires_at = excluded.expires_at
            """, (file_hash, summary, expires_at))
    
    def get_summary(self, file_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT summary FROM summaries WHERE file_hash = ? AND expires_at > ?",
                (file_hash, time.time())
            ).fetchone()
            return result[0] if result else None
    
    def cleanup_expired_summaries(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM summaries WHERE expires_at < ?", (time.time(),))

    def save_log(self, log_type: str, file: str = None, folder: str = None, message: str = None, metadata: dict = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_logs (type, file, folder, message, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (log_type, file, folder, message, json.dumps(metadata) if metadata else None))

    def get_recent_logs(self, limit: int = 30):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT type, file, folder, message, metadata, created_at 
                FROM system_logs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            logs = []
            for r in rows:
                try:
                    ts = float(r[5])
                except:
                    ts = time.time()
                    
                logs.append({
                    "type": r[0],
                    "file": r[1],
                    "folder": r[2],
                    "message": r[3],
                    "metadata": json.loads(r[4]) if r[4] else None,
                    "timestamp": ts
                })
            return logs

    # ─── Standout Feature Methods ──────────────────────────────────────

    def save_cluster_snapshot(self, clusters: dict):
        """Save a snapshot of current cluster state for Time Travel."""
        snapshot_data = {}
        file_count = 0
        for name, data in clusters.items():
            members = [{"file": os.path.basename(fp), "score": round(s, 4)} for fp, s in data.get("members", [])]
            snapshot_data[name] = members
            file_count += len(members)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO cluster_snapshots (snapshot, file_count, cluster_count) VALUES (?, ?, ?)",
                (json.dumps(snapshot_data), file_count, len(clusters))
            )

    def get_cluster_snapshots(self, limit: int = 50):
        """Get cluster snapshots for Time Travel timeline."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, file_count, cluster_count, created_at FROM cluster_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r[0], "file_count": r[1], "cluster_count": r[2], "timestamp": r[3]} for r in rows]

    def get_snapshot_by_id(self, snapshot_id: int):
        """Get a specific cluster snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT snapshot, file_count, cluster_count, created_at FROM cluster_snapshots WHERE id = ?",
                (snapshot_id,)
            ).fetchone()
            if row:
                return {"snapshot": json.loads(row[0]), "file_count": row[1], "cluster_count": row[2], "timestamp": row[3]}
            return None

    def save_file_move(self, file_path: str, from_cluster: str, to_cluster: str):
        """Record a file move for entropy tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO file_moves (file_path, from_cluster, to_cluster) VALUES (?, ?, ?)",
                (file_path, from_cluster, to_cluster)
            )

    def get_file_move_counts(self):
        """Get move counts per file for entropy heatmap."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT file_path, COUNT(*) as move_count,
                       MAX(created_at) as last_move
                FROM file_moves
                GROUP BY file_path
            """).fetchall()
            return {os.path.basename(r[0]): {"moves": r[1], "last_move": r[2]} for r in rows}

    def save_folder_summary(self, cluster_name: str, summary: str, coherence_score: float):
        """Cache a folder summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO folder_summaries (cluster_name, summary, coherence_score) VALUES (?, ?, ?)",
                (cluster_name, summary, coherence_score)
            )

    def get_folder_summary(self, cluster_name: str):
        """Get cached folder summary."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT summary, coherence_score, created_at FROM folder_summaries WHERE cluster_name = ?",
                (cluster_name,)
            ).fetchone()
            if row:
                return {"summary": row[0], "coherence_score": row[1], "created_at": row[2]}
            return None


class FAISSIndex:
    def __init__(self, index_path: str, dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self._load_or_create()
    
    def _load_or_create(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
    
    def add(self, embeddings: np.ndarray, ids: list[int] = None):
        if ids is None:
            ids = list(range(self.index.ntotal, self.index.ntotal + len(embeddings)))
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
        return ids
    
    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]
    
    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[list, list]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        distances, indices = self.index.search(queries, k)
        return distances.tolist(), indices.tolist()
    
    def remove_ids(self, ids: list[int]):
        if hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(np.array(ids, dtype=np.int64))
    
    def save(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
    
    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0


class ChunkingService:
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None):
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def chunk_for_llm(self, text: str, max_chars: int = 3000):
        """Chunk for LLM summarization (larger chunks, fewer calls)."""
        if len(text) <= max_chars:
            return [text] if text else []
        
        # Split by paragraphs first, then by size
        paragraphs = text.split('\n\n')
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += para + '\n\n'
            else:
                if current:
                    chunks.append(current.strip())
                current = para + '\n\n'
        
        if current:
            chunks.append(current.strip())
        
        return chunks


class EmbeddingService:
    def __init__(self, model_name: str = None, use_gpu: bool = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        
        self.device = "cuda" if use_gpu else "cpu"
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load model on {self.device}: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model = SentenceTransformer(model_name, device="cpu")
            
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def encode_parallel(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size)


class LLMService:
    def __init__(self):
        self.api_url = settings.CEREBRAS_API_URL
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT
        self.max_tokens = settings.LLM_MAX_TOKENS
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cluster_name_cache = {}
        self._last_cache_cleanup = 0
        self._key_index = 0
        self._keys = settings.CEREBRAS_API_KEYS
    
    def _get_current_key(self):
        if not self._keys:
            return ""
        return self._keys[self._key_index % len(self._keys)]
    
    def _rotate_key(self):
        self._key_index += 1
        new_key = self._get_current_key()
        logger.warning(f"Rotating to API key index {self._key_index % len(self._keys)}")
        return new_key
    
    def _cleanup_cache(self):
        now = time.time()
        if now - self._last_cache_cleanup > settings.CLUSTER_NAME_CACHE_TTL:
            self._cluster_name_cache.clear()
            self._last_cache_cleanup = now
    
    def generate_cluster_name(self, file_paths: list[str], sample_contents: list[str]) -> str:
        self._cleanup_cache()
        
        cache_key = tuple(sorted(file_paths))
        if cache_key in self._cluster_name_cache:
            return self._cluster_name_cache[cache_key]
        
        if not file_paths:
            return "General"
        
        filenames = [os.path.basename(fp) for fp in file_paths]
        content_section = ""
        if sample_contents:
            content_section = "\n\nFile content samples:\n" + "\n---\n".join(
                f"[{filenames[i] if i < len(filenames) else 'file'}]:\n{c[:600]}"
                for i, c in enumerate(sample_contents)
            )
        prompt = (
            f"You are a file organizer. Based on these filenames AND their contents, "
            f"generate a short, descriptive folder name (2-3 words, snake_case, lowercase).\n\n"
            f"Files: {', '.join(filenames)}"
            f"{content_section}\n\n"
            f"Return ONLY the folder name, nothing else."
        )
        
        from cerebras.cloud.sdk import Cerebras
        
        for attempt in range(len(self._keys) or 1):
            key = self._get_current_key()
            try:
                client = Cerebras(api_key=key)
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    timeout=30.0
                )
                
                content = response.choices[0].message.content.strip()
                name = "".join(c for c in content if c.isalnum() or c in (' ', '_')).strip()
                name = name.replace(" ", "_")
                result = name or self._fallback_name(file_paths)
                self._cluster_name_cache[cache_key] = result
                logger.info(f"LLM generated cluster name: '{result}' for {len(file_paths)} files")
                return result

            except Exception as e:
                logger.warning(f"LLM API error (Cluster Name) with key ending ...{key[-4:] if key else 'None'}: {e}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    logger.warning(f"Rate limit hit for key index {self._key_index % len(self._keys)}")
                self._rotate_key()

        fallback = self._fallback_name(file_paths)
        logger.info(f"Fallback cluster name: '{fallback}' (LLM failed for all keys)")
        return fallback
    
    def _fallback_name(self, file_names: list[str]) -> str:
        extensions = {}
        for f in file_names:
            ext = os.path.splitext(f)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        if extensions:
            most_common = max(extensions.items(), key=lambda x: x[1])[0]
            return most_common[1:].upper() if most_common else "Mixed"
        return "General"
    
    async def generate_file_summary(self, content: str, file_hash: str = None) -> str:
        # Check cache
        if file_hash:
            cached = vector_store.get_summary(file_hash)
            if cached:
                return cached
        
        summary = await self._generate_summary_async(content, file_hash)
        return summary
    
    async def _generate_summary_async(self, content: str, file_hash: str = None):
        import aiohttp
        
        chunks = chunking_service.chunk_for_llm(content)
        
        if not chunks:
            return "[Empty file]"
        
        if len(chunks) == 1:
            summary = await self._summarize_text(chunks[0])
        else:
            # Map: summarize each chunk
            chunk_summaries = await asyncio.gather(
                *[self._summarize_text(chunk) for chunk in chunks[:5]]  # Limit to 5 chunks
            )
            # Reduce: combine summaries
            combined = "\n".join(chunk_summaries)
            summary = await self._summarize_text(combined, is_combined=True)
        
        # Cache if we have file hash
        if file_hash and summary:
            vector_store.save_summary(file_hash, summary)
        
        return summary
    
    async def _summarize_text(self, text: str, is_combined: bool = False):
        if is_combined:
            prompt = f"Combine these section summaries into one coherent summary (2-3 sentences). Focus on the high-level purpose and functionality:\n\n{text[:4000]}"
        else:
            prompt = f"Provide a concise summary (2-3 sentences) of the following content. If it is code, explain WHAT it does and its main components. Do NOT output the code itself:\n\n{text[:3000]}"
        
        from cerebras.cloud.sdk import AsyncCerebras
        
        for attempt in range(len(self._keys) or 1):
            key = self._get_current_key()
            try:
                async with AsyncCerebras(api_key=key) as client:
                    print(f"DEBUG: Calling Cerebras for summary... Prompt length: {len(prompt)}")
                    response = await client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a code analysis expert. Your task is to provide a plain-text summary of what the code does. CRITICAL RULE: NEVER include source code, snippets, or the word 'def' in your output. Talk ONLY about purpose and functionality."
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        model=self.model,
                        temperature=0.1,
                        max_tokens=self.max_tokens,
                        timeout=30.0
                    )
                    
                    content = response.choices[0].message.content.strip()
                    print(f"DEBUG: LLM Success. Content: {content[:100]}...")
                    return content

            except Exception as e:
                logger.warning(f"LLM API error (Summary) with key ending ...{key[-4:] if key else 'None'}: {e}")
                
                if "429" in str(e) or "Too Many Requests" in str(e):
                    logger.warning(f"Rate limit hit for key index {self._key_index % len(self._keys)}")
                
                self._rotate_key()

        msg = "AI Summary Unavailable: The system could not generate a summary at this time. Please try again later or check the logs for details."
        print(f"DEBUG: LLM Failed. Returning fallback: {msg}")
        return msg

    async def generate_folder_summary(self, folder_name: str, file_names: list[str], sample_contents: list[str]) -> dict:
        """Generate a folder intelligence summary using LLM."""
        files_str = ", ".join(file_names[:10])
        samples_str = "\n---\n".join(s[:500] for s in sample_contents[:3])
        
        prompt = f"""Analyze this semantic folder and its files.
Folder: "{folder_name}"
Files: {files_str}
Sample content:
{samples_str}

Respond in EXACTLY this JSON format (nothing else):
{{"description": "2-sentence description of what this folder contains and why these files are grouped",
"suggested_name": "a better snake_case folder name if the current one could be improved, otherwise repeat current name"}}"""
        
        from cerebras.cloud.sdk import AsyncCerebras
        
        for attempt in range(len(self._keys) or 1):
            key = self._get_current_key()
            try:
                async with AsyncCerebras(api_key=key) as client:
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        temperature=0.2,
                        max_tokens=256,
                        timeout=20.0
                    )
                    content = response.choices[0].message.content.strip()
                    # Try to parse JSON
                    try:
                        result = json.loads(content)
                        return result
                    except json.JSONDecodeError:
                        return {"description": content[:200], "suggested_name": folder_name}
            except Exception as e:
                logger.warning(f"Folder summary LLM error: {e}")
                self._rotate_key()
        
        return {"description": f"Folder containing {len(file_names)} semantically related files.", "suggested_name": folder_name}


class NLCommandService:
    """Natural Language Command processor using gpt-oss-120b reasoning model."""
    
    def __init__(self):
        self.model = settings.NL_MODEL
        self.api_url = settings.NL_API_URL
        self.api_key = settings.NL_API_KEY
        self.max_tokens = settings.NL_MAX_TOKENS
        self.timeout = settings.NL_TIMEOUT
    
    @staticmethod
    def strip_thinking(text: str) -> str:
        """Strip <think>...</think> reasoning tokens from model output."""
        import re
        # Remove <think>...</think> blocks (including multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Also handle unclosed <think> tags (model might not close them)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()
    
    async def execute(self, command: str, context: dict) -> dict:
        """
        Execute a natural language command.
        context = {files: [...], folders: [...], clusters: {...}}
        Returns: {action, result, explanation}
        """
        folders_str = ", ".join(context.get("folders", []))
        files_str = ", ".join(context.get("files", [])[:30])
        
        system_prompt = f"""You are SEFS AI — a semantic file system assistant. You help users manage their files using natural language commands.

Current file system state:
- Folders: {folders_str}
- Files (up to 30): {files_str}

You can perform these actions:
1. SEARCH: Find files semantically. Params: {{"query": "search text"}}
2. MOVE: Move files to a folder. Params: {{"filename": "file.txt", "target_folder": "folder_name"}} OR for bulk: {{"source_folder": "old_folder", "target_folder": "new_folder"}} OR {{"filenames": ["f1.txt","f2.txt"], "target_folder": "folder"}}
3. DELETE: Delete files permanently. Params: {{"filename": "file.txt"}} OR {{"filenames": ["f1.txt","f2.txt"]}}
4. EXPLAIN: Explain why a file is in a specific folder. Params: {{}}
5. ORGANIZE: Suggest and execute reorganization. Params: {{"moves": [{{"filename": "f.txt", "target_folder": "folder"}}]}}
6. QUERY: Answer questions about the file system. Params: {{}}

IMPORTANT RULES:
- Use EXACT filenames from the list above (case-sensitive, including extension).
- For move/delete, prefer "filenames" array when multiple files are involved.
- If the user says "move all files from X to Y", use source_folder + target_folder.
- For delete, ONLY delete files the user explicitly names or describes. Never delete files speculatively.

Respond ONLY with this JSON (no extra text):
{{"action": "search|move|delete|explain|organize|query",
"params": {{...}},
"explanation": "brief natural language explanation"}}"""

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_url
            )
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            raw_content = response.choices[0].message.content or ""
            content = self.strip_thinking(raw_content)
            
            # Try to parse JSON
            try:
                # Extract JSON from possible markdown code block
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Fallback: return as plain explanation
            return {
                "action": "query",
                "params": {},
                "explanation": content
            }
            
        except Exception as e:
            logger.error(f"NL Command error: {e}")
            return {
                "action": "error",
                "params": {},
                "explanation": f"I couldn't process that command: {str(e)}"
            }


class ClusteringService:
    def __init__(self):
        self.faiss_index = FAISSIndex(settings.FAISS_INDEX_PATH)
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.top_k_sim = settings.TOP_K_SIMILARITY
        self.min_cluster_size = settings.MIN_CLUSTER_SIZE
    
    def _compute_file_similarity(self, vecs_a: list[np.ndarray], vecs_b: list[np.ndarray]) -> float:
        """
        Compute similarity between two files using Top-K chunk-pair average.
        For all chunk pairs (a_i, b_j), compute cosine similarity,
        take the top-K highest, and return their average.
        """
        if not vecs_a or not vecs_b:
            return 0.0
        
        # Stack into matrices
        mat_a = np.array(vecs_a, dtype=np.float32)
        mat_b = np.array(vecs_b, dtype=np.float32)
        
        # Normalize rows
        norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
        norms_a[norms_a == 0] = 1
        norms_b[norms_b == 0] = 1
        mat_a = mat_a / norms_a
        mat_b = mat_b / norms_b
        
        # Cosine similarity matrix: (len_a, len_b)
        sim_matrix = mat_a @ mat_b.T
        
        # Flatten and take top-K
        all_sims = sim_matrix.flatten()
        k = min(self.top_k_sim, len(all_sims))
        top_k_indices = np.argpartition(all_sims, -k)[-k:]
        top_k_sims = all_sims[top_k_indices]
        
        return float(np.mean(top_k_sims))
    
    def cluster_documents(self, file_paths: list[str], embeddings_dict: dict) -> dict:
        """
        Cluster documents using Top-K chunk-pair average similarity.
        For each pair of files, compute cosine similarity between ALL chunk pairs,
        average the top-K similarities, and use connected components to form clusters.
        Returns: {cluster_key: {centroid: np.ndarray, members: [(file_path, score)]}}
        """
        if not file_paths:
            return {}
        
        # Build file-to-chunks mapping (only files with embeddings)
        file_to_chunks = {}
        for fp in file_paths:
            if fp in embeddings_dict and embeddings_dict[fp]:
                file_to_chunks[fp] = embeddings_dict[fp]
        
        if len(file_to_chunks) < self.min_cluster_size:
            clusters = {}
            for fp in file_to_chunks:
                centroid = np.mean(file_to_chunks[fp], axis=0)
                clusters[fp] = {"centroid": centroid, "members": [(fp, 1.0)]}
            return clusters
        
        file_list = list(file_to_chunks.keys())
        n = len(file_list)
        
        # Compute pairwise file-file similarity matrix using Top-K chunk-pair average
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sim_matrix[i][i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                sim = self._compute_file_similarity(
                    file_to_chunks[file_list[i]],
                    file_to_chunks[file_list[j]]
                )
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
        
        logger.info(f"Computed {n}x{n} similarity matrix using Top-K={self.top_k_sim} chunk-pair average")
        
        # Build connected components using Union-Find
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path compression
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Union files that exceed the similarity threshold
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.similarity_threshold:
                    union(i, j)
        
        # Group by connected components
        clusters_dict = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(i)
        
        # Build final clusters with membership scores
        clusters = {}
        for root_idx, member_indices in clusters_dict.items():
            members_fps = [file_list[i] for i in member_indices]
            
            if len(members_fps) < self.min_cluster_size and n > self.min_cluster_size:
                # Small clusters - keep as singletons
                for fp in members_fps:
                    centroid = np.mean(file_to_chunks[fp], axis=0)
                    clusters[fp] = {"centroid": centroid, "members": [(fp, 1.0)]}
            else:
                # Compute cluster centroid from all member chunk centroids
                all_member_centroids = [np.mean(file_to_chunks[fp], axis=0) for fp in members_fps]
                cluster_centroid = np.mean(all_member_centroids, axis=0)
                
                # Membership score = average Top-K similarity to other cluster members
                members_with_scores = []
                for i_idx in member_indices:
                    if len(member_indices) == 1:
                        score = 1.0
                    else:
                        scores = [sim_matrix[i_idx][j_idx] for j_idx in member_indices if j_idx != i_idx]
                        score = float(np.mean(scores))
                    members_with_scores.append((file_list[i_idx], max(0, score)))
                
                cluster_key = f"cluster_{root_idx}"
                clusters[cluster_key] = {
                    "centroid": cluster_centroid,
                    "members": members_with_scores
                }
        
        return clusters
    
    @staticmethod
    def generate_extension_name(file_paths: list[str]) -> str:
        """Generate cluster name based on dominant file extension."""
        extensions = {}
        for fp in file_paths:
            ext = os.path.splitext(os.path.basename(fp))[1].lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
        
        if not extensions:
            return "General_Files"
        
        # Most common extension
        dominant = max(extensions.items(), key=lambda x: x[1])[0]
        name = dominant[1:].upper()  # Remove dot, uppercase
        
        # Check if mixed (dominant is < 60% of total)
        total = sum(extensions.values())
        if extensions[dominant] / total < 0.6 and len(extensions) > 1:
            return "Mixed_Files"
        
        return f"{name}_Files"
    
    def rebalance_clusters(self, clusters: dict, embeddings_dict: dict) -> dict:
        """Remove low-membership files and try to reassign them."""
        rebalanced = {}
        
        for name, data in clusters.items():
            strong_members = [(fp, score) for fp, score in data["members"] if score >= 0.5]
            
            if len(strong_members) >= self.min_cluster_size:
                rebalanced[name] = {
                    "centroid": data["centroid"],
                    "members": strong_members
                }
            else:
                pass  # Singletons stay as-is
        
        return rebalanced


class SemanticService:
    def __init__(self):
        self.vector_store = VectorStore(settings.VECTOR_DB_PATH)
        self.chunker = ChunkingService()
        self.embedder = EmbeddingService()
        self.llm_service = LLMService()
        self.clusterer = ClusteringService()
        self.nl_service = NLCommandService()
        self.lock_manager = FileLockManager()
        self.log_callback = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._rebalance_timer = None

    def log_event(self, log_type: str, file: str = None, folder: str = None, message: str = None, metadata: dict = None):
        """Save log to DB and trigger broadcast callback."""
        # Save to DB
        self.vector_store.save_log(log_type, file, folder, message, metadata)
        
        # Trigger real-time broadcast
        if self.log_callback:
            log_data = {
                "type": log_type,
                "file": file,
                "folder": folder,
                "message": message,
                "metadata": metadata,
                "timestamp": time.time()
            }
            self.log_callback(log_data)
    
    def semantic_search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Semantic search: LLM expands query → embed → FAISS search → return ranked file results.
        """
        # Step 1: LLM query expansion for better recall
        expanded_query = query
        try:
            from cerebras.cloud.sdk import Cerebras
            key = self.llm_service._get_current_key()
            client = Cerebras(api_key=key)
            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"Rewrite this search query to be more descriptive for semantic similarity search. Add related terms and synonyms. Return ONLY the expanded query, nothing else.\n\nOriginal query: {query}"
                }],
                model=self.llm_service.model,
                temperature=0.3,
                max_tokens=100,
                timeout=10.0
            )
            expanded_query = response.choices[0].message.content.strip()
            logger.info(f"Search query expanded: '{query}' → '{expanded_query}'")
        except Exception as e:
            logger.warning(f"LLM query expansion failed, using original: {e}")
            expanded_query = query
        
        # Step 2: Embed the query
        query_embedding = self.embedder.encode_single(expanded_query)
        
        # Step 3: Search FAISS
        k = min(limit * 5, self.clusterer.faiss_index.ntotal)  # Search more chunks to get enough unique files
        if k == 0:
            return []
        
        distances, indices = self.clusterer.faiss_index.search(query_embedding, k)
        
        # Step 4: Map FAISS indices to files
        file_scores = {}  # file_path -> best_score
        with sqlite3.connect(self.vector_store.db_path) as conn:
            for dist, faiss_idx in zip(distances, indices):
                if faiss_idx == -1:
                    continue
                row = conn.execute("""
                    SELECT d.file_path, c.content
                    FROM chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.faiss_idx = ?
                """, (int(faiss_idx),)).fetchone()
                
                if row:
                    file_path, chunk_content = row
                    # Convert L2 distance to similarity score (0-1 range)
                    similarity = max(0, 1.0 - (float(dist) / 4.0))
                    
                    if file_path not in file_scores or similarity > file_scores[file_path]["score"]:
                        file_scores[file_path] = {
                            "score": similarity,
                            "snippet": chunk_content[:200] if chunk_content else "",
                        }
        
        # Step 5: Get cluster membership and build results
        clusters = self.vector_store.get_clusters() or {}
        file_to_cluster = {}
        for cluster_name, data in clusters.items():
            for member in data.get("members", []):
                file_to_cluster[member[0]] = cluster_name
        
        results = []
        for file_path, info in sorted(file_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]:
            results.append({
                "file": os.path.basename(file_path),
                "path": file_path,
                "folder": file_to_cluster.get(file_path, None),
                "score": round(info["score"], 4),
                "snippet": info["snippet"],
            })
        
        logger.info(f"Semantic search '{query}' → {len(results)} results")
        return results
    
    def _read_file_content(self, file_path: str) -> str | None:
        """Extract text content from a file. Supports PDF and common text formats."""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                import PyPDF2
                text_parts = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                content = "\n\n".join(text_parts)
                if content.strip():
                    logger.info(f"PDF extracted: {os.path.basename(file_path)} ({len(reader.pages)} pages, {len(content)} chars)")
                    return content
                else:
                    logger.warning(f"PDF has no extractable text: {os.path.basename(file_path)}")
                    return None
            else:
                # Text-based files (.txt, .md, .csv, .json, .xml, .html, etc.)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read {os.path.basename(file_path)}: {e}")
            return None
    
    def process_file(self, file_path: str, content: str = None) -> dict:
        """Process a single file (PDF or text): chunk, embed, add to FAISS."""
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        stat = os.path.getsize(file_path)
        file_hash = self._hash_file(file_path)
        
        if content is None:
            content = self._read_file_content(file_path)
            if content is None:
                return {"error": f"Cannot read file: unsupported format or read error"}
        
        # Chunk
        chunks = self.chunker.chunk_text(content)
        if not chunks:
            return {"error": "No content to process"}
        
        # Embed
        embeddings = self.embedder.encode_parallel(chunks)
        
        # Store in SQLite
        doc_id = self.vector_store.upsert_document(file_path, file_hash, os.path.getmtime(file_path), len(chunks))
        
        # Store chunks
        chunk_data = [(c, self._hash_text(c)) for c in chunks]
        self.vector_store.save_chunks(doc_id, chunk_data, embeddings)
        
        # Add to FAISS
        start_idx = self.clusterer.faiss_index.ntotal
        faiss_ids = self.clusterer.faiss_index.add(embeddings)
        self.vector_store.update_faiss_indices(doc_id, list(range(start_idx, start_idx + len(chunks))))
        self.clusterer.faiss_index.save()
        
        return {
            "file_path": file_path,
            "chunks_processed": len(chunks),
            "doc_id": doc_id
        }
    
    def delete_file(self, file_path: str):
        """Remove file from vector store and FAISS, clean up empty cluster folder."""
        filename = os.path.basename(file_path)
        self.vector_store.delete_document(file_path)
        self.log_event("delete", file=filename, message=f"Deleted {filename}")
        # Clean up empty cluster folder if applicable
        parent_dir = os.path.dirname(file_path)
        self.vector_store._cleanup_empty_folder(parent_dir)
        # Note: FAISS removal is expensive, rebuild periodically instead
    
    def cluster_all(self, force_regenerate: bool = False) -> dict:
        """Cluster all documents using Top-K chunk-pair similarity. Virtual only - no file moves."""
        file_paths = self.vector_store.get_all_file_paths()
        
        if not file_paths:
            return {}
        
        # Get all embeddings
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for file_path, emb_blob, content, chunk_idx in all_chunks:
            if file_path not in embeddings_dict:
                embeddings_dict[file_path] = []
            embeddings_dict[file_path].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        # Cluster using Top-K chunk-pair average similarity
        clusters = self.clusterer.cluster_documents(file_paths, embeddings_dict)
        
        # Generate folder names using LLM for small clusters, fallback for large
        named_clusters = {}
        used_names = {}
        for name, data in clusters.items():
            file_list = [fp for fp, _ in data["members"]]
            
            if len(data["members"]) <= 10:
                # Use LLM for descriptive naming — include file content samples
                samples = []
                for fp in file_list[:5]:
                    chunks = self.vector_store.get_document_embeddings(fp)
                    if chunks:
                        samples.append(chunks[0][2][:800])  # First chunk content
                
                folder_name = self.llm_service.generate_cluster_name(file_list, samples)
            else:
                # Fallback for large clusters
                folder_name = self.llm_service._fallback_name([os.path.basename(f) for f in file_list])
            
            # Deduplicate names
            if folder_name in used_names:
                used_names[folder_name] += 1
                folder_name = f"{folder_name}_{used_names[folder_name]}"
            else:
                used_names[folder_name] = 1
            
            named_clusters[folder_name] = data
        
        # Save to DB
        self.vector_store.save_clusters(named_clusters)
        
        # ─── Physical OS sync: move files to cluster subfolders ───
        moved_files = {}
        for cluster_name, data in named_clusters.items():
            for fp, _ in data["members"]:
                new_path = self.vector_store.sync_file_to_disk(fp, cluster_name)
                if new_path != fp:
                    moved_files[os.path.basename(fp)] = cluster_name
        
        if moved_files:
            logger.info(f"OS sync: physically moved {len(moved_files)} files to cluster folders")
        
        # Save snapshot for Time Travel
        try:
            self.vector_store.save_cluster_snapshot(named_clusters)
        except Exception as e:
            logger.warning(f"Failed to save cluster snapshot: {e}")
        
        self.log_event("cluster", message=f"Clustered {len(file_paths)} files into {len(named_clusters)} groups")
        
        return named_clusters
    
    def recluster_all(self) -> dict:
        """Re-clustering with physical file moves to match cluster structure."""
        clusters = self.cluster_all()
        return {"clusters": list(clusters.keys()), "moved": {}}
    
    def scan_and_organize_directory(self, directory_path: str) -> dict:
        """Scan a local directory, process all files, and organize them IN-PLACE.
        Creates semantic cluster subfolders inside the original directory."""
        if not os.path.isdir(directory_path):
            return {"error": f"Directory not found: {directory_path}"}
        
        directory_path = os.path.abspath(directory_path)
        processed = 0
        errors = []
        
        # Collect all files (including in subdirectories)
        all_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories and SEFS cache
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in files:
                if fname.startswith('.'):
                    continue
                all_files.append(os.path.join(root, fname))
        
        if not all_files:
            return {"error": "No files found in directory"}
        
        logger.info(f"Scanning directory: {directory_path} ({len(all_files)} files)")
        
        # Process each file and track source directory
        original_paths = set()
        for file_path in all_files:
            try:
                result = self.process_file(file_path)
                if "error" not in result:
                    # Track the source directory for in-place sync
                    self.vector_store.set_source_dir(file_path, directory_path)
                    original_paths.add(os.path.normpath(file_path))
                    processed += 1
                else:
                    errors.append(f"{os.path.basename(file_path)}: {result['error']}")
            except Exception as e:
                errors.append(f"{os.path.basename(file_path)}: {e}")
        
        if processed == 0:
            return {"error": "No files could be processed", "details": errors}
        
        # Cluster all files — this also physically moves them via sync_file_to_disk
        clusters = self.cluster_all()
        
        # Count how many files actually moved (compare current DB paths vs originals)
        moved_count = 0
        current_paths = self.vector_store.get_all_file_paths()
        for p in current_paths:
            if os.path.normpath(p) not in original_paths:
                # Path changed, meaning the file was moved
                if os.path.normpath(os.path.dirname(os.path.dirname(p))) == os.path.normpath(directory_path):
                    moved_count += 1
        
        # Clean up empty original subdirectories
        for root, dirs, files in os.walk(directory_path, topdown=False):
            if root == directory_path:
                continue
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    logger.info(f"Cleaned up empty folder: {root}")
            except Exception:
                pass
        
        logger.info(f"In-place organization complete: {processed} files → {len(clusters)} clusters, {moved_count} files moved")
        
        return {
            "status": "organized",
            "directory": directory_path,
            "files_processed": processed,
            "clusters": list(clusters.keys()),
            "files_moved": moved_count,
            "errors": errors if errors else None
        }
    
    async def get_file_summary(self, file_path: str) -> str:
        """Get or generate summary for a file."""
        file_hash = self._hash_file(file_path)
        
        try:
            with self.lock_manager.lock():
                if not os.path.exists(file_path):
                    # Check if keys are rotated or file moved
                    return "[File not found - moved]"
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                summary = await self.llm_service.generate_file_summary(content, file_hash)
                self.log_event("summary", file=os.path.basename(file_path), message=f"Generated AI summary")
                return summary
        except Exception:
            return "[Cannot read file]"
    
    def find_similar(self, file_path: str, limit: int = 5) -> list[tuple[str, float]]:
        """Find similar files using Top-K chunk-pair similarity."""
        rows = self.vector_store.get_document_embeddings(file_path)
        if not rows:
            return []
        
        target_vecs = [np.frombuffer(emb, dtype=np.float32) for _, emb, _ in rows]
        
        # Get all other files' embeddings
        all_chunks = self.vector_store.get_all_document_chunks()
        other_files = {}
        for fp, emb_blob, content, chunk_idx in all_chunks:
            if fp == file_path:
                continue
            if fp not in other_files:
                other_files[fp] = []
            other_files[fp].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        # Compute Top-K chunk-pair similarity against each file
        similarities = []
        for fp, vecs in other_files.items():
            sim = self.clusterer._compute_file_similarity(target_vecs, vecs)
            similarities.append((fp, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def get_file_chunks(self, file_path: str) -> list[dict]:
        """Get chunks for a file."""
        rows = self.vector_store.get_document_embeddings(file_path)
        return [
            {"index": idx, "content": content[:200] + "..." if len(content) > 200 else content}
            for idx, emb, content in rows
        ]
    
    def cleanup_expired_caches(self):
        self.vector_store.cleanup_expired_summaries()
    
    # ─── Standout Feature Methods ──────────────────────────────────────

    def find_duplicates(self) -> list[dict]:
        """Find near-duplicate files (similarity > threshold)."""
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for file_path, emb_blob, content, chunk_idx in all_chunks:
            if file_path not in embeddings_dict:
                embeddings_dict[file_path] = []
            embeddings_dict[file_path].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        file_list = list(embeddings_dict.keys())
        duplicates = []
        threshold = settings.DUPLICATE_THRESHOLD
        
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                sim = self.clusterer._compute_file_similarity(
                    embeddings_dict[file_list[i]], embeddings_dict[file_list[j]]
                )
                if sim >= threshold:
                    duplicates.append({
                        "file_a": os.path.basename(file_list[i]),
                        "file_b": os.path.basename(file_list[j]),
                        "similarity": round(sim, 4)
                    })
        
        return duplicates
    
    def get_cross_cluster_edges(self) -> list[dict]:
        """Find semantic edges between files in different clusters."""
        clusters = self.vector_store.get_clusters()
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for file_path, emb_blob, content, chunk_idx in all_chunks:
            if file_path not in embeddings_dict:
                embeddings_dict[file_path] = []
            embeddings_dict[file_path].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        # Build file-to-cluster mapping
        file_to_cluster = {}
        for cname, cdata in clusters.items():
            for fp, _ in cdata.get("members", []):
                file_to_cluster[fp] = cname
        
        edges = []
        file_list = list(embeddings_dict.keys())
        min_sim = settings.CROSS_CLUSTER_MIN
        max_sim = settings.CROSS_CLUSTER_MAX
        
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                c_i = file_to_cluster.get(file_list[i], "")
                c_j = file_to_cluster.get(file_list[j], "")
                if c_i == c_j:
                    continue  # Same cluster, skip
                sim = self.clusterer._compute_file_similarity(
                    embeddings_dict[file_list[i]], embeddings_dict[file_list[j]]
                )
                if min_sim <= sim <= max_sim:
                    edges.append({
                        "source": os.path.basename(file_list[i]),
                        "target": os.path.basename(file_list[j]),
                        "source_cluster": c_i,
                        "target_cluster": c_j,
                        "similarity": round(sim, 4)
                    })
        
        return edges
    
    def predict_placement(self, file_path: str, content: str) -> list[dict]:
        """Predict which cluster a file would be placed in."""
        # Get embeddings for the file
        chunks = self.chunker.chunk_text(content)
        if not chunks:
            return []
        
        embeddings = self.embedder.encode_parallel(chunks)
        file_vecs = [emb for emb in embeddings]
        
        # Get current clusters and their member embeddings
        clusters = self.vector_store.get_clusters()
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for fp, emb_blob, c, ci in all_chunks:
            if fp not in embeddings_dict:
                embeddings_dict[fp] = []
            embeddings_dict[fp].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        # Compute similarity against each cluster's centroid (average of member embeddings)
        predictions = []
        for cname, cdata in clusters.items():
            cluster_vecs = []
            for fp, _ in cdata.get("members", []):
                cluster_vecs.extend(embeddings_dict.get(fp, []))
            if not cluster_vecs:
                continue
            sim = self.clusterer._compute_file_similarity(file_vecs, cluster_vecs)
            predictions.append({
                "folder": cname,
                "confidence": round(sim, 4),
                "member_count": cdata.get("member_count", 0)
            })
        
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:5]
    
    def get_entropy_data(self) -> dict:
        """Get file stability/entropy data for heatmap."""
        move_counts = self.vector_store.get_file_move_counts()
        now = time.time()
        entropy = {}
        
        for filename, data in move_counts.items():
            moves = data["moves"]
            last_move = data.get("last_move", 0) or 0
            recency = max(0, 1.0 - (now - last_move) / 86400)  # Decay over 24h
            
            # Entropy score: 0 = stable (green), 1 = volatile (red)
            if moves <= 1:
                score = 0.1 * recency
            elif moves <= 3:
                score = 0.3 + 0.2 * recency
            else:
                score = min(1.0, 0.5 + 0.1 * moves * recency)
            
            entropy[filename] = {
                "score": round(score, 3),
                "moves": moves,
                "last_move": last_move,
                "stability": "stable" if score < 0.3 else "shifting" if score < 0.6 else "volatile"
            }
        
        return entropy
    
    def evaluate_move(self, filename: str, target_folder: str) -> dict:
        """Evaluate the semantic impact of moving a file to a different cluster."""
        # Find file and its current cluster
        file_paths = self.vector_store.get_all_file_paths()
        file_path = None
        for fp in file_paths:
            if os.path.basename(fp) == filename:
                file_path = fp
                break
        
        if not file_path:
            return {"error": "File not found"}
        
        # Get file embeddings
        rows = self.vector_store.get_document_embeddings(file_path)
        if not rows:
            return {"error": "No embeddings for file"}
        file_vecs = [np.frombuffer(emb, dtype=np.float32) for _, emb, _ in rows]
        
        # Get current cluster info
        current_cluster = self.vector_store.get_cluster_for_file(file_path)
        clusters = self.vector_store.get_clusters()
        
        # Get target cluster embeddings
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for fp, emb_blob, c, ci in all_chunks:
            if fp not in embeddings_dict:
                embeddings_dict[fp] = []
            embeddings_dict[fp].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        target_vecs = []
        if target_folder in clusters:
            for fp, _ in clusters[target_folder].get("members", []):
                target_vecs.extend(embeddings_dict.get(fp, []))
        
        current_vecs = []
        current_name = current_cluster or "Unknown"
        if current_name in clusters:
            for fp, _ in clusters[current_name].get("members", []):
                if fp != file_path:
                    current_vecs.extend(embeddings_dict.get(fp, []))
        
        # Compute similarities
        current_sim = self.clusterer._compute_file_similarity(file_vecs, current_vecs) if current_vecs else 0
        target_sim = self.clusterer._compute_file_similarity(file_vecs, target_vecs) if target_vecs else 0
        
        impact = target_sim - current_sim
        
        return {
            "file": filename,
            "current_folder": current_name,
            "target_folder": target_folder,
            "current_coherence": round(current_sim, 4),
            "target_coherence": round(target_sim, 4),
            "impact": round(impact, 4),
            "recommendation": "beneficial" if impact > 0.05 else "neutral" if impact > -0.05 else "detrimental"
        }
    
    async def get_folder_intelligence(self, folder_name: str) -> dict:
        """Get or generate folder intelligence summary."""
        # Check cache
        cached = self.vector_store.get_folder_summary(folder_name)
        if cached and (time.time() - cached.get("created_at", 0) < 3600):
            return cached
        
        # Get folder files
        clusters = self.vector_store.get_clusters()
        if folder_name not in clusters:
            return {"summary": "Unknown folder", "coherence_score": 0}
        
        members = clusters[folder_name].get("members", [])
        file_names = [os.path.basename(fp) for fp, _ in members]
        
        # Get sample content
        samples = []
        for fp, _ in members[:3]:
            rows = self.vector_store.get_document_embeddings(fp)
            if rows:
                samples.append(rows[0][2][:500])
        
        # Compute coherence score
        all_chunks = self.vector_store.get_all_document_chunks()
        embeddings_dict = {}
        for fp, emb_blob, content, chunk_idx in all_chunks:
            if fp not in embeddings_dict:
                embeddings_dict[fp] = []
            embeddings_dict[fp].append(np.frombuffer(emb_blob, dtype=np.float32))
        
        member_paths = [fp for fp, _ in members]
        coherence = 0.0
        pairs = 0
        for i in range(len(member_paths)):
            for j in range(i + 1, len(member_paths)):
                if member_paths[i] in embeddings_dict and member_paths[j] in embeddings_dict:
                    sim = self.clusterer._compute_file_similarity(
                        embeddings_dict[member_paths[i]], embeddings_dict[member_paths[j]]
                    )
                    coherence += sim
                    pairs += 1
        coherence = coherence / pairs if pairs > 0 else 0.5
        
        # Generate LLM summary
        llm_result = await self.llm_service.generate_folder_summary(folder_name, file_names, samples)
        
        result = {
            "summary": llm_result.get("description", ""),
            "suggested_name": llm_result.get("suggested_name", folder_name),
            "coherence_score": round(coherence, 4),
            "file_count": len(members)
        }
        
        # Cache
        self.vector_store.save_folder_summary(folder_name, result["summary"], coherence)
        
        return result
    
    async def execute_nl_command(self, command: str) -> dict:
        """Execute a natural language command."""
        # Build context
        clusters = self.vector_store.get_clusters()
        files = [os.path.basename(fp) for fp in self.vector_store.get_all_file_paths()]
        folders = list(clusters.keys())
        
        context = {"files": files, "folders": folders, "clusters": {n: d.get("member_count", 0) for n, d in clusters.items()}}
        
        result = await self.nl_service.execute(command, context)
        
        # If the action is "search", execute it
        if result.get("action") == "search":
            query = result.get("params", {}).get("query", command)
            search_results = self.semantic_search(query, limit=5)
            result["search_results"] = search_results
        
        # If the action is "move", actually perform the move
        elif result.get("action") == "move":
            params = result.get("params", {})
            target = params.get("target_folder", "")
            moved_count = 0
            
            if target:
                all_paths = self.vector_store.get_all_file_paths()
                
                # Handle single file move
                fname = params.get("filename", "")
                filenames = params.get("filenames", [])
                # Also support source_folder for bulk moves
                source_folder = params.get("source_folder", "")
                
                files_to_move = []
                
                if fname:
                    filenames = [fname]
                
                if filenames:
                    # Move specific files by name
                    for fn in filenames:
                        for fp in all_paths:
                            if os.path.basename(fp) == fn:
                                files_to_move.append(fp)
                                break
                
                elif source_folder:
                    # Move all files from a source folder/cluster
                    clusters_now = self.vector_store.get_clusters()
                    if source_folder in clusters_now:
                        files_to_move = [fp for fp, _ in clusters_now[source_folder].get("members", [])]
                
                # Execute the moves
                for fp in files_to_move:
                    old_cluster_result = self.vector_store.get_cluster_for_file(fp)
                    old_cluster = old_cluster_result[0] if old_cluster_result else "Unknown"
                    
                    success = self.vector_store.reassign_file_to_cluster(fp, target)
                    if success:
                        moved_count += 1
                        self.vector_store.save_file_move(fp, old_cluster, target)
                        logger.info(f"NL Move: {os.path.basename(fp)} -> {target}")
                
                # Evaluate final coherence
                if moved_count > 0:
                    result["moved_files"] = moved_count
                    result["explanation"] = f"{result.get('explanation', '')} ✅ Successfully moved {moved_count} file(s) to '{target}'."
                else:
                    result["moved_files"] = 0
                    result["explanation"] = f"{result.get('explanation', '')} ⚠️ Could not find files to move."
        
        # If the action is "organize", handle bulk reorganization
        elif result.get("action") == "organize":
            params = result.get("params", {})
            moves = params.get("moves", [])
            moved_count = 0
            all_paths = self.vector_store.get_all_file_paths()
            
            for move_spec in moves:
                fn = move_spec.get("filename", "")
                tgt = move_spec.get("target_folder", "")
                if fn and tgt:
                    for fp in all_paths:
                        if os.path.basename(fp) == fn:
                            old_result = self.vector_store.get_cluster_for_file(fp)
                            old_name = old_result[0] if old_result else "Unknown"
                            if self.vector_store.reassign_file_to_cluster(fp, tgt):
                                moved_count += 1
                                self.vector_store.save_file_move(fp, old_name, tgt)
                            break
            
            if moved_count > 0:
                result["moved_files"] = moved_count
                result["explanation"] = f"{result.get('explanation', '')} ✅ Reorganized {moved_count} file(s)."
        
        # If the action is "delete", remove files from DB and disk
        elif result.get("action") == "delete":
            params = result.get("params", {})
            fname = params.get("filename", "")
            filenames = params.get("filenames", [])
            if fname:
                filenames = [fname]
            
            deleted_count = 0
            all_paths = self.vector_store.get_all_file_paths()
            
            for fn in filenames:
                for fp in all_paths:
                    if os.path.basename(fp) == fn:
                        try:
                            # Remove from vector store / DB
                            self.delete_file(fp)
                            # Remove physical file from disk
                            if os.path.exists(fp):
                                os.remove(fp)
                                logger.info(f"NL Delete: physically removed {fp}")
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"NL Delete failed for {fn}: {e}")
                        break
            
            result["deleted_files"] = deleted_count
            if deleted_count > 0:
                result["explanation"] = f"{result.get('explanation', '')} 🗑️ Deleted {deleted_count} file(s)."
            else:
                result["explanation"] = f"{result.get('explanation', '')} ⚠️ Could not find files to delete."
        
        self.log_event("nl_command", message=f"NL: {command[:50]}")
        return result
    
    def generate_export_report(self) -> str:
        """Generate a Markdown report of the full semantic organization."""
        clusters = self.vector_store.get_clusters()
        entropy = self.get_entropy_data()
        duplicates = self.find_duplicates()
        
        lines = ["# SEFS Semantic Organization Report"]
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Files:** {self.vector_store.get_document_count()}")
        lines.append(f"**Total Clusters:** {len(clusters)}")
        lines.append(f"**Total Chunks:** {self.vector_store.get_chunk_count()}")
        
        lines.append("\n## Semantic Folders")
        for name, data in clusters.items():
            members = data.get("members", [])
            lines.append(f"\n### 📁 {name}")
            lines.append(f"**Files:** {len(members)}")
            for fp, score in members:
                fname = os.path.basename(fp)
                ent = entropy.get(fname, {})
                stability = ent.get("stability", "stable")
                icon = "🟢" if stability == "stable" else "🟡" if stability == "shifting" else "🔴"
                lines.append(f"- {icon} `{fname}` (score: {round(score, 3)}, {stability})")
        
        if duplicates:
            lines.append("\n## ⚠️ Near-Duplicate Files")
            for dup in duplicates:
                lines.append(f"- `{dup['file_a']}` ↔ `{dup['file_b']}` ({round(dup['similarity'] * 100, 1)}% similar)")
        
        return "\n".join(lines)
    
    def clear_all(self):
        """Clear all data: DB, FAISS index, and uploaded files."""
        # Clear DB
        self.vector_store.clear_all_data()
        
        # Clear LLM name cache
        self.llm_service._cluster_name_cache.clear()
        logger.info("Cleared LLM cluster name cache")
        
        # Reset FAISS index
        self.clusterer.faiss_index.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.clusterer.faiss_index.dimension))
        self.clusterer.faiss_index.save()
        
        # Clear upload storage
        upload_dir = settings.UPLOAD_STORAGE
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                fp = os.path.join(upload_dir, f)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp)
                except Exception as e:
                    logger.error(f"Failed to delete {fp}: {e}")
        
        self.log_event("clear", message="All data cleared")
        logger.info("Cleared all data: DB, FAISS, and upload storage")
        return {"status": "cleared"}
    
    @staticmethod
    def _hash_file(file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()


# Global instances - initialized lazily
vector_store = None
chunking_service = ChunkingService()
embedding_service = None
llm_service = None
clustering_service = None
semantic_service = None


def initialize_services():
    """Initialize services lazily on first use"""
    global vector_store, embedding_service, llm_service, clustering_service, semantic_service
    
    if semantic_service is not None:
        return semantic_service
    
    logger.info("Initializing semantic services...")
    
    if vector_store is None:
        vector_store = VectorStore(settings.VECTOR_DB_PATH)
    
    if embedding_service is None:
        embedding_service = EmbeddingService()
    
    if llm_service is None:
        llm_service = LLMService()
        
    if clustering_service is None:
        clustering_service = ClusteringService()
        
    if semantic_service is None:
        semantic_service = SemanticService()
    
    logger.info("Semantic services initialized")
    return semantic_service


def get_semantic_service():
    """Get or initialize semantic service"""
    global semantic_service
    if semantic_service is None:
        initialize_services()
    return semantic_service


# Don't initialize on import - do it lazily
# initialize_services()
import os
import time
import shutil
import logging
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import settings

logger = logging.getLogger(__name__)


class SemanticHandler(FileSystemEventHandler):
    def __init__(self, update_callback=None):
        self.update_callback = update_callback
        self.processing_files = set()
        self._recluster_lock = threading.Lock()
        self._recluster_timer = None
        self._pending_files = []
        self._last_recluster = 0
        self._semantic_service = None
        # Paths to suppress — SEFS-triggered moves won't be re-processed
        self._suppressed_paths = set()
        self._suppress_lock = threading.Lock()

    def suppress_path(self, path: str, duration: float = 5.0):
        """Temporarily suppress events for a path (used during SEFS-triggered moves)."""
        normalized = os.path.normpath(path)
        with self._suppress_lock:
            self._suppressed_paths.add(normalized)
        # Auto-clear after duration
        def _clear():
            time.sleep(duration)
            with self._suppress_lock:
                self._suppressed_paths.discard(normalized)
        threading.Thread(target=_clear, daemon=True).start()
    
    def _is_suppressed(self, path: str) -> bool:
        normalized = os.path.normpath(path)
        with self._suppress_lock:
            return normalized in self._suppressed_paths

    def get_service(self):
        if self._semantic_service is None:
            from semantic_engine import get_semantic_service
            self._semantic_service = get_semantic_service()
        return self._semantic_service
    
    def on_created(self, event):
        if not event.is_directory and not self._is_suppressed(event.src_path):
            self.process_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and not self._is_suppressed(event.src_path):
            self.process_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and not self._is_suppressed(event.src_path):
            file_path = event.src_path
            self.get_service().delete_file(file_path)
            self._schedule_recluster()
            if self.update_callback:
                self.update_callback({"type": "delete", "file": os.path.basename(file_path)})
    
    def on_moved(self, event):
        if not event.is_directory and not self._is_suppressed(event.src_path) and not self._is_suppressed(event.dest_path):
            logger.info(f"Detected move: {event.src_path} -> {event.dest_path}")
            self.get_service().delete_file(event.src_path)
            self.process_file(event.dest_path)
    
    def process_file(self, file_path):
        if not os.path.exists(file_path):
            return
        
        if file_path in self.processing_files:
            return
        
        filename = os.path.basename(file_path)
        if filename.startswith('.'):
            return
        
        self.processing_files.add(file_path)
        threading.Thread(target=self._async_process, args=(file_path,)).start()
    
    def _async_process(self, file_path):
        try:
            time.sleep(1.5)
            if not os.path.exists(file_path):
                return
            
            result = self.get_service().process_file(file_path)
            if "error" in result:
                logger.error(f"Failed to process {file_path}: {result['error']}")
                return
            
            logger.info(f"Processed {os.path.basename(file_path)}: {result['chunks_processed']} chunks")
            
            self._pending_files.append(file_path)
            self._schedule_recluster()
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        finally:
            if file_path in self.processing_files:
                self.processing_files.remove(file_path)
    
    def _schedule_recluster(self):
        """Debounced re-clustering."""
        with self._recluster_lock:
            now = time.time()
            if now - self._last_recluster < 5.0:  # 5 second debounce
                if self._recluster_timer is not None:
                    self._recluster_timer.cancel()
                self._recluster_timer = threading.Timer(5.0, self._do_recluster)
                self._recluster_timer.start()
            else:
                self._do_recluster()
    
    def _do_recluster(self):
        """Re-cluster and move files to appropriate folders."""
        try:
            self._last_recluster = time.time()
            pending = self._pending_files.copy()
            self._pending_files.clear()
            
            if not pending:
                return
            
            logger.info(f"Re-clustering {len(pending)} new/modified files...")
            
            result = self.get_service().recluster_all()
            
            for filename, folder in result.get("moved", {}).items():
                if self.update_callback:
                    self.update_callback({
                        "type": "move",
                        "file": filename,
                        "folder": folder
                    })
            
            logger.info(f"Re-clustering complete: {len(result.get('clusters', []))} folders")
            
        except Exception as e:
            logger.error(f"Re-clustering failed: {e}")


class FileWatcher:
    def __init__(self, update_callback=None):
        self.observer = Observer()
        self.handler = SemanticHandler(update_callback)
        self.root = settings.MONITORED_ROOT
    
    def start(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.observer.schedule(self.handler, self.root, recursive=True)
        self.observer.start()
        
        # Register handler with semantic_engine for SEFS→OS suppression
        try:
            from semantic_engine import set_watcher_handler
            set_watcher_handler(self.handler)
        except ImportError:
            pass
        
        logger.info(f"Watcher started on {self.root} (Recursive Mode, Bidirectional Sync)")
    
    def stop(self):
        if self.handler._recluster_timer:
            self.handler._recluster_timer.cancel()
        self.observer.stop()
        self.observer.join()
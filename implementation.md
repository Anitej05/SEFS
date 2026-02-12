# SEFS Implementation Details

This document provides a deep dive into the specific implementation strategies and algorithms used in the **Semantic File System (SEFS)**.

---

## 📅 The Life of a File in SEFS

When a file enters the system (via upload or the `monitored_root`), it undergoes several stages:

### 1. Ingestion & Pre-processing
*   **Watcher Detection**: The `FileWatcher` (using Python's `watchdog` library) detects the file and triggers a `SemanticHandler`.
*   **Debouncing**: A 1.5-second sleep is applied to ensure the OS has finished writing the file before processing starts.
*   **Hashing**: The entire file is hashed (SHA-256) to check for previous versions. Only modified files are processed.

### 2. Semantic Chunking
*   **Size**: Files are split into **512-token chunks** with a **50-token overlap** (tunable in `config.py`).
*   **Strategy**: The `ChunkingService` uses space-based splits to avoid breaking words, ensuring semantic integrity for the embedder.
*   **Storage**: Each chunk's text and its hash are stored in the `chunks` table in SQLite.

### 3. Vectorization
*   **Model**: `all-MiniLM-L6-v2` converts each chunk into a 384-dimensional vector.
*   **Hardware**: Inference is prioritized on **CUDA GPUs** but falls back seamlessly to CPU if necessary.
*   **FAISS Integration**: Embeddings are appended to a `FlatL2` index for exact search or an `HNSW` index for fast approximate search.

### 4. Semantic Clustering
The system uses a unique **Top-K Chunk Similarity** approach to cluster files:

1.  **Pairwise Matrix**: Calculate similarity between every pair of documents ($D_i, D_j$).
2.  **Top-K Average**: Instead of averaging all chunks, the system computes cosine similarity for *all* pairs of chunks ($chunk_{i,n}, chunk_{j,m}$) and averages only the **top 5 strongest matches**. This allows files with partially related content (e.g., a shared module) to be grouped effectively.
3.  **Union-Find**: A Union-Find (Disjoint Set Union) algorithm groups files into connected components where similarity exceeds the threshold (default 0.7).
4.  **Rebalancing**: Periodically, the system re-evaluates cluster membership to drift low-score files into more appropriate groups.

### 5. AI Reasoning (LLM)
*   **Folder Naming**: `Llama 3.3` receives the file paths and a few-shot prompt. It returns a concise, snake_case name that describes the shared theme of the cluster.
*   **Summarization**: Uses a **Map-Reduce** chain:
    1.  **Map**: Summarize independent 3000-character chunks.
    2.  **Reduce**: Combine chunk summaries into a single 2-3 sentence overview.
*   **Caching**: Summaries are cached in SQLite for 24 hours (TTL).

---

## 📡 Real-time Synchronization

SEFS uses a "Lazy-Broadcast" pattern:
1.  **Backend Events**: `SemanticService` triggers a log event.
2.  **WebSocket Manager**: Broadcasts the event to all active clients.
3.  **Frontend Update**: The React `Dashboard.jsx` receives the event and updates its local state using a `useCallback` hook, causing specific UI components (like the file list or graph) to re-render without a full page refresh.

---

## 💾 Database Schema

### `documents` table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key |
| `file_path` | TEXT | Physical path |
| `file_hash` | TEXT | Content hash |
| `chunk_count`| INTEGER | Chunks generated |

### `chunks` table
| Column | Type | Description |
| :--- | :--- | :--- |
| `faiss_idx` | INTEGER | Index in the FAISS binary |
| `content` | TEXT | The raw snippet |
| `embedding` | BLOB | Pickled numpy array |

---

## 🚀 Optimized Search

Semantic search is not a simple FAISS lookup. It includes **Query Expansion**:
1.  User query: *"database setup"*
2.  LLM expansion: *"SQL schemas, connection strings, migrations, database initialization scripts"*
3.  Vector Search: Embedding the expanded query and searching FAISS for the Top-50 chunks.
4.  Aggregation: Ranking files based on the frequency and strength of their chunks appearing in the search results.

# SEFS Tech Stack Documentation

This document provides a detailed overview of the technologies, frameworks, and libraries used in the **Semantic File System (SEFS)** project.

## 🏗️ Architecture Overview

SEFS is a distributed system consisting of a React-based interactive frontend and a Python-powered semantic backend. It uses vector embeddings to organize files semantically rather than just by name or date.

---

## 🎨 Frontend (Client-Side)

The frontend is built for performance and interactivity, focusing on visualizing the semantic relationships between files.

| Category | Technology | Version | Description |
| :--- | :--- | :--- | :--- |
| **Framework** | [React](https://react.dev/) | `^19.2.0` | Core UI library for building the dashboard. |
| **Build Tool** | [Vite](https://vitejs.dev/) | `^7.3.1` | Next-generation frontend tooling for fast development. |
| **Visualization** | [React Force Graph](https://github.com/vasturiano/react-force-graph) | `^1.48.2` | Renders the interactive 2D node-link diagram for file clusters. |
| **Styling** | Vanilla CSS | - | custom modern glassmorphic design system. |
| **Animation** | [Framer Motion](https://www.framer.com/motion/) | `^12.34.0` | Production-ready motion library for React. |
| **Icons** | [Lucide React](https://lucide.dev/) | `^0.563.0` | Clean and consistent SVG icon set. |
| **HTTP Client** | [Axios](https://axios-http.com/) | `^1.13.5` | Promise-based HTTP client for API communication. |

---

## ⚙️ Backend (Server-Side)

The backend handles file indexing, vector storage, and orchestration of AI models.

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Language** | [Python 3](https://www.python.org/) | Primary language for backend logic and AI integration. |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance web framework for building APIs. |
| **Web Server** | [Uvicorn](https://www.uvicorn.org/) | Lightning-fast ASGI server implementation. |
| **File Watching** | [Watchdog](https://github.com/gorakhargosh/python-watchdog) | Monitors the file system for real-time semantic updates. |
| **Schema/Validation**| [Pydantic](https://docs.pydantic.dev/) | Data validation and settings management using python type hints. |
| **Asynchronous I/O** | `websockets`, `aiohttp` | Used for real-time communication and async API calls. |

---

## 🧠 AI & Machine Learning

The "Semantic" core of the system relies on state-of-the-art embedding and language models.

### 🔍 Embeddings & Vector Search
- **Model**: `all-MiniLM-L6-v2` (via [Sentence Transformers](https://www.sbert.net/))
    - Generates 384-dimensional dense vectors representing file content.
- **Vector Engine**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
    - Used for high-speed similarity search across millions of file chunks.
- **Hardware Acceleration**: [CUDA](https://developer.nvidia.com/cuda-zone) support for GPU-accelerated embedding generation.

### 💬 Large Language Models (LLM)
- **Model**: `Llama 3.3 70B`
- **Inference Engine**: [Cerebras AI](https://www.cerebras.net/)
    - Provides ultra-low latency inference for cluster naming and file summarization.
- **Prompting Strategy**: Map-reduce for long file summaries and few-shot for folder naming.

---

## 💾 Storage & Data

| Type | Technology | Description |
| :--- | :--- | :--- |
| **Metadata DB** | [SQLite](https://sqlite.org/) | Stores file metadata, chunk paths, and semantic links (`vectors.db`). |
| **Vector Store** | FAISS Binary | Persistent storage for the FAISS index (`faiss_index.bin`). |
| **File Storage** | Local Disk | `monitored_root/` for organized files and `upload_storage/` for temporary uploads. |

---

## 🛠️ Infrastructure & DevTools

- **Environment Management**: Python `venv` and Node.js `node_modules`.
- **CORS Handling**: Middleware configured in FastAPI for frontend-backend communication.
- **Auto-Reload**: Enabled for both Vite (HMR) and Uvicorn (`--reload`).
- **TTS (Optional)**: Basic Text-to-Speech service integrated via `tts_service.py`.

## 📈 Performance Characteristics

- **Embedding Speed**: ~1,000 chunks per second on GPU.
- **Search Latency**: Sub-millisecond similarity lookups via FAISS.
- **Summarization**: 2-5 seconds per file, with 24-hour intelligent caching.
- **Clustering**: Connected components-based clustering in under 1 second for hundreds of files.

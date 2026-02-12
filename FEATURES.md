# SEFS Features

## Upload Capabilities

### 1. Single File Upload
- Upload any file type
- Automatic semantic categorization
- Real-time processing feedback

### 2. Multiple Files Upload
- Select multiple files at once
- Batch processing
- Progress tracking

### 3. Folder Upload
- Upload entire folder structures
- Preserves subfolder hierarchy during processing
- Recursive file discovery

### 4. Zip Archive Upload
- Upload .zip files
- Automatic extraction
- Processes all contained files
- Supports nested folders within zip

## File Processing

### Chunking
- Splits files into 512-token chunks
- 50-token overlap for context preservation
- Semantic boundary detection

### Embedding
- MiniLM-L6-v2 model (384 dimensions)
- GPU-accelerated when available
- Parallel batch processing

### Change Detection
- Hash-based chunk comparison
- Only re-embeds modified chunks
- Efficient incremental updates

## Clustering

### Vector-Based Similarity
- FAISS index for fast search
- Cosine similarity between document centroids
- Connected components algorithm

### Dynamic Clustering
- Threshold-based grouping (default: 0.7)
- Automatic cluster formation
- Rebalancing for outliers

### Folder Naming
- LLM-generated names (Llama 3.3 70B)
- Samples 3 representative files
- Fallback to extension-based naming
- Cached indefinitely

## File Summaries

### On-Demand Generation
- Lazy loading (only when clicked)
- Map-reduce for large files
- 24-hour cache TTL

### Multi-Chunk Summarization
1. Split into manageable chunks
2. Summarize each chunk
3. Combine summaries
4. Generate final summary

## User Interface

### Browse View
- Force-directed graph visualization
- Interactive node exploration
- Zoom and pan controls
- Empty state with upload prompt

### Layers View
- Hierarchical folder display
- Expandable file lists
- Quick actions per file
- Folder management

### Knowledge Base
- Searchable file index
- Filter by name or folder
- Quick file preview
- Metadata display

### System Metrics
- Real-time statistics
- Performance monitoring
- Engine status indicators
- Activity logs

## Real-Time Features

### WebSocket Updates
- Live file movement notifications
- Instant UI updates
- Connection status monitoring

### Auto-Categorization
- Triggered on file upload
- Triggered on file modification
- Debounced re-clustering (5s)

### Activity Logging
- File operations tracking
- Timestamp recording
- Operation type classification

## API Endpoints

### File Operations
- `POST /upload` - Single file/zip upload
- `POST /upload-bulk` - Multiple files
- `GET /files` - File structure
- `GET /file/{folder}/{filename}` - File content
- `POST /delete-file` - Delete file
- `POST /move` - Move file

### Clustering
- `POST /recluster` - Force re-cluster
- `GET /clusters` - Cluster info
- `GET /similar/{filename}` - Find similar files

### Summaries
- `GET /file-summary/{filename}` - Get/generate summary
- `GET /file-chunks/{filename}` - View chunks

### System
- `GET /stats` - System metrics
- `GET /system-info` - Configuration
- `WS /ws` - WebSocket connection

## Performance Optimizations

### Caching
- Folder names: Permanent
- File summaries: 24 hours
- Embeddings: Permanent (until file changes)

### Batching
- Multiple file uploads processed in parallel
- Debounced re-clustering
- Batch embedding generation

### Indexing
- FAISS for O(log n) similarity search
- SQLite indexes on key fields
- Efficient chunk lookup

## Error Handling

### Upload Errors
- Invalid file type detection
- Size limit enforcement
- Graceful failure messages

### Processing Errors
- Chunk extraction failures
- Embedding generation errors
- LLM API timeouts

### Recovery
- Automatic retry logic
- Fallback mechanisms
- Error logging and tracking

## Configuration

### Adjustable Parameters
- Chunk size and overlap
- Similarity threshold
- Clustering parameters
- Cache TTLs
- GPU usage

### Model Selection
- Embedding model choice
- LLM model selection
- API provider configuration

## Security

### File Validation
- Extension checking
- Content type verification
- Size limits

### Path Safety
- Sanitized file paths
- Directory traversal prevention
- Monitored root enforcement

## Extensibility

### Plugin Architecture
- Custom embedding models
- Alternative LLM providers
- Custom clustering algorithms

### API Integration
- RESTful endpoints
- WebSocket support
- Batch operations

## Future Enhancements

### Planned Features
- Multi-language support
- Custom folder naming rules
- Advanced search filters
- File versioning
- Collaborative features
- Cloud storage integration

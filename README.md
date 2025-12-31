# Photo & Video Search with CLIP

A powerful semantic search system using OpenAI's CLIP model and Qdrant vector database. Search through large collections of images and videos using natural language queries with semantic understanding.

## Features

- 🔍 **Semantic Image Search**: Search images using natural language descriptions
- 🎬 **Semantic Video Search**: Search videos with frame-level understanding and clustering
- 🚀 **GPU Acceleration**: Optimized for CUDA-enabled GPUs with batch processing
- 🎯 **High Accuracy**: Powered by OpenAI's CLIP ViT-Large-Patch14 model
- 📊 **Vector Database**: Efficient similarity search with Qdrant
- 🎨 **Interactive UI**: Beautiful Streamlit web interface for both images and videos
- 📁 **Recursive Scanning**: Automatically processes all content in nested folders
- ⚡ **Batch Processing**: Process thousands of items efficiently
- 🎞️ **Smart Frame Clustering**: Videos are sampled, clustered, and mean-pooled for optimal search

## Architecture

### Image Search
```
┌─────────────────┐
│  Image Folder   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLIP Encoder   │ ──► Image Embeddings (768D vectors)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qdrant Vector DB│ ──► Store & Index
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Query      │ ──► Text Embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cosine Search   │ ──► Top-K Results
└─────────────────┘
```

### Video Search
```
┌─────────────────┐
│  Video Folder   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Sampling  │ ──► Extract frames @ configurable FPS
│  (5-15 FPS)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLIP Encoder   │ ──► Frame Embeddings (768D vectors)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HDBSCAN Cluster │ ──► Group similar frames
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mean Pooling   │ ──► One embedding per cluster
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qdrant Vector DB│ ──► Store with metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Query      │ ──► Search across video segments
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cosine Search   │ ──► Top-K Video Segments
└─────────────────┘
```

## Prerequisites

- Python 3.12
- CUDA 12.1+ (for GPU support)
- Docker (for Qdrant)
- 4GB+ GPU memory recommended
- OpenCV for video processing

## Installation

### Option 1: Docker (Recommended)

The easiest way to run the application with all dependencies.

1. **Build and start services**:
```bash
docker-compose up -d --build
```

2. **Access the application**:
- Streamlit UI: http://localhost:8501
- Qdrant API: http://localhost:6333
- Qdrant Dashboard: http://localhost:6333/dashboard

3. **Index your images**:
```bash
# Place images in ./images folder first
docker-compose exec app uv run python3 image_to_embedding.py
```

4. **Stop services**:
```bash
docker-compose down
```

**Docker Architecture:**
```
┌─────────────────────────────────────────┐
│  Docker Network: image-search-network   │
│                                         │
│  ┌──────────────┐    ┌──────────────┐ │
│  │   Qdrant     │◄───┤  Streamlit   │ │
│  │   :6333      │    │    App       │ │
│  │              │    │   :8501      │ │
│  └──────┬───────┘    └──────────────┘ │
│         │                              │
│         ▼                              │
│  qdrant_data/                          │
│  (persistent storage)                  │
└─────────────────────────────────────────┘
```

**Data Persistence:**
- Vector DB data: `./qdrant_data/` (automatically created and persisted)
- Images: `./images/` (mounted as read-only)
- Search results: `./search_results/`

### Option 2: Local Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd photo-doc-data-embeddings
```

2. **Install dependencies with uv**:
```bash
uv sync
```

Or install manually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow qdrant-client numpy streamlit
```

3. **Start Qdrant vector database**:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

## Usage

### Docker Deployment

#### Start the Application
```bash
docker-compose up -d
```

#### Index Images
```bash
# Place your images in ./images folder
docker-compose exec app uv run python3 image_to_embedding.py
```

#### Search via Web UI
Open http://localhost:8501 in your browser

#### Search via CLI
```bash
docker-compose exec app uv run python3 search_images.py "your query"
```

#### View Logs
```bash
docker-compose logs -f app
docker-compose logs -f qdrant
```

#### Stop Services
```bash
docker-compose down
```

### Local Deployment

### 1. Index Your Images

First, process your images and create embeddings:

```python
python3 image_to_embedding.py
```

**Configuration** (edit in `image_to_embedding.py`):
```python
FOLDER_PATH = "./images"          # Your images folder
COLLECTION_NAME = "image_embeddings"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
```

The script will:
- Recursively scan all images in the folder
- Generate embeddings using CLIP
- Store vectors in Qdrant with metadata
- Process in batches for efficiency

**Supported image formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp`, `.svg`, `.ico`

**Supported video formats**: `.mp4`, `.avi`, `.mov`, `.mkv`

### 2. Search via Web UI (Recommended)

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Features:
- Enter natural language queries
- Adjust number of results (1-200)
- View images in a responsive grid
- See similarity scores and metadata
- Configure Qdrant connection

### 3. Search via Command Line

#### Search Images
```bash
python3 search_images.py "aadhaar card"
python3 search_images.py "passport photo"
python3 search_images.py "person smiling with glasses"
```

Results are copied to `./search_results/` folder with ranking and scores.

#### Search Videos
```bash
python3 search_videos.py "person walking"
python3 search_videos.py "car driving on highway"
python3 search_videos.py "people talking indoors"
```

Results show matching video segments with frame information.

## Video Search Implementation

### How It Works

1. **Frame Sampling**: Videos are sampled at a configurable rate (5-15 FPS)
   - Controlled by `VIDEO_SAMPLE_RATE` environment variable
   - Extracts representative frames from the entire video

2. **Embedding Generation**: Each frame is processed through CLIP
   - Generates 768-dimensional embeddings
   - Batch processing for efficiency

3. **Clustering**: Similar frames are grouped using HDBSCAN
   - Identifies semantic scenes/segments in the video
   - Filters out noise and transitional frames
   - Configurable cluster parameters

4. **Mean Pooling**: Each cluster is represented by a single embedding
   - Averages all frame embeddings in a cluster
   - Normalized for cosine similarity search
   - Preserves semantic information

5. **Indexing**: Pooled embeddings stored in Qdrant with metadata
   - Video path, cluster info, frame indices
   - Enables precise segment retrieval

### Video Search Configuration

Environment variables for video processing:

```bash
# Frame sampling rate (frames per second)
VIDEO_SAMPLE_RATE=10  # Default: 10 FPS

# Clustering parameters
MIN_CLUSTER_SIZE=5    # Minimum frames to form a cluster
MIN_SAMPLES=3         # HDBSCAN min_samples parameter

# Database settings
VIDEO_COLLECTION_NAME=video_embeddings  # Qdrant collection name
VECTOR_DIMENSIONS=768                   # CLIP embedding size
```

### Index Videos

```bash
# Set video folder path
export VIDEO_FOLDER_PATH="./videos"

# Run video indexing
python3 video_to_embedding.py
```

Or use the Streamlit UI to index videos interactively.

## API Reference

### ImageEmbeddingProcessor

Main class for image processing and search.

#### Initialization
```python
processor = ImageEmbeddingProcessor(
    model_name="openai/clip-vit-large-patch14",
    batch_size=64  # Adjust based on GPU memory
)
```

#### Methods

**`image_to_embedding(image_path: str) -> np.ndarray`**
- Converts a single image to embedding vector
- Returns: 768-dimensional normalized numpy array

**`text_to_embedding(text: str) -> np.ndarray`**
- Converts text to embedding vector
- Returns: 768-dimensional normalized numpy array

**`batch_image_to_embeddings(image_paths: List[str]) -> np.ndarray`**
- Process multiple images in batch
- More efficient than individual processing
- Returns: Array of embedding vectors

**`process_folder_to_qdrant(folder_path, collection_name, qdrant_host, qdrant_port)`**
- Index all images in folder to Qdrant
- Creates/recreates collection
- Processes in batches with progress tracking

**`search_by_text(query_text, collection_name, qdrant_host, qdrant_port, limit)`**
- Search for similar images using text query
- Returns: List of results with scores and metadata

### VideoEmbeddingProcessor

Main class for video processing and search.

#### Initialization
```python
processor = VideoEmbeddingProcessor(
    model_name="openai/clip-vit-large-patch14"
)
```

#### Methods

**`process_videos_to_qdrant(folder_path, collection_name, qdrant_host, qdrant_port)`**
- Index all videos in folder to Qdrant
- Samples frames, generates embeddings, clusters, and stores
- Automatic scene detection and segmentation

**`search_videos_by_text(query_text, collection_name, qdrant_host, qdrant_port, limit)`**
- Search for video segments using text query
- Returns: List of matching segments with metadata
  - Video path and name
  - Cluster ID and frame indices
  - Similarity score
  - Frame count information

**`get_collection_stats(collection_name, qdrant_host, qdrant_port)`**
- Get statistics about indexed videos
- Returns: Total embeddings, dimensions, distance metric

## Configuration

### GPU Settings

The script automatically detects and uses GPU if available. To force CPU:

```python
self.device = "cpu"  # In ImageEmbeddingProcessor.__init__
```

### Batch Size

Adjust based on your GPU memory:
- 4GB GPU: `batch_size=32`
- 6GB GPU: `batch_size=64`
- 8GB+ GPU: `batch_size=128`

### Qdrant Configuration

Edit connection settings:
```python
QDRANT_HOST = "localhost"  # Or remote host
QDRANT_PORT = 6333
COLLECTION_NAME = "image_embeddings"
```

## Performance Tips

1. **GPU Memory**: Reduce batch size if you get OOM errors
2. **Indexing Speed**: Use GPU for 10x faster processing
3. **Search Speed**: Qdrant is optimized for sub-millisecond searches
4. **Storage**: ~3KB per image for embeddings

### Benchmarks

On RTX 3050 (4GB):
- Indexing: ~10-15 images/second
- Search: <100ms for 50K images
- Embedding dimension: 768

```
photo-doc-data-embeddings/
├── Dockerfile              # App container definition
├── docker-compose.yml      # Multi-container orchestration
├── .dockerignore          # Docker build exclusions
├── image_to_embedding.py  # Image processing & indexing
├── video_to_embedding.py  # Video processing & indexing
├── search_images.py       # CLI image search tool
├── search_videos.py       # CLI video search tool
├── app.py                 # Streamlit web interface (images & videos)
├── pyproject.toml         # Dependencies
├── README.md              # Documentation
├── video_embeddings/      # Video processing module
│   ├── __init__.py       # Module exports
│   ├── ingest.py         # Video frame sampling
│   ├── embedding.py      # Frame embedding generation
│   ├── cluster.py        # HDBSCAN clustering
│   ├── mean_pool.py      # Cluster pooling
│   ├── vector_db.py      # Qdrant operations
│   └── orchestrator.py   # Video indexing pipeline
├── qdrant_data/           # Vector DB storage (Docker)
├── images/                # Your image folder (create this)
├── videos/                # Your video folder (create this)
├── search_results/        # Search output folder
└── .venv/                 # Virtual environment (local)
```

## Docker Configuration

### Services

**Qdrant (Vector Database)**
- Image: `qdrant/qdrant:latest`
- Ports: 6333 (API), 6334 (gRPC)
- Volume: `./qdrant_data:/qdrant/storage` (persistent)
- Network: `image-search-network`

**App (Streamlit + CLIP)**
- Build: Custom Dockerfile with Python 3.12
- Port: 8501
- Environment:
  - `QDRANT_HOST=qdrant`
  - `QDRANT_PORT=6333`
- Volumes:
  - `./images:/app/images:ro` (read-only)
  - `./search_results:/app/search_results`

### Docker Commands

**View running containers:**
```bash
docker-compose ps
```

**Access container shell:**
```bash
docker-compose exec app bash
docker-compose exec qdrant sh
```

**View resource usage:**
```bash
docker stats
```

**Clean up everything:**
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

### Backup and Restore

**Backup Qdrant data:**
```bash
tar -czf qdrant_backup_$(date +%Y%m%d).tar.gz qdrant_data/
```

**Restore from backup:**
```bash
docker-compose down
tar -xzf qdrant_backup_20231224.tar.gz
docker-compose up -d
```

### GPU Support in Docker

To enable NVIDIA GPU support:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Add to `docker-compose.yml`:
```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Project Structure

## Troubleshooting

### Docker Issues

**Container fails to start:**
```bash
# Check logs
docker-compose logs -f

# Rebuild without cache
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Qdrant not ready:**
```bash
# Check health status
docker-compose ps

# Wait for healthy status
docker-compose up -d
curl http://localhost:6333/health
```

**App can't connect to Qdrant:**
```bash
# Test connection from app container
docker-compose exec app curl http://qdrant:6333/health

# Verify network
docker network inspect photo-doc-data-embeddings_image-search-network
```

**Permission issues with qdrant_data:**
```bash
sudo chown -R $USER:$USER qdrant_data/
```

**Port conflicts:**
```yaml
# Edit docker-compose.yml
ports:
  - "8502:8501"  # Change host port
  - "6334:6333"  # Change host port
```

### Local Installation Issues

### CUDA Errors

If you get cuBLAS errors:
```python
# Use CPU instead
self.device = "cpu"
```

Or reinstall PyTorch with correct CUDA version:
```bash
uv remove torch torchvision
uv add torch torchvision --index https://download.pytorch.org/whl/cu121
```

### Out of Memory

Reduce batch size:
```python
processor = ImageEmbeddingProcessor(batch_size=16)
```

### Qdrant Connection Failed

Ensure Qdrant is running:
```bash
docker ps | grep qdrant
```

Restart if needed:
```bash
docker restart <qdrant-container-id>
```

### No Results Found

- Check collection name matches
- Verify images were indexed successfully
## Example Queries

### Image Search
- `"indian aadhaar card"`
- `"passport photograph with blue background"`
- `"person wearing glasses"`
- `"document with signature"`
- `"group photo outdoors"`
- `"landscape with mountains"`
- `"indoor office setting"`

### Video Search
- `"person walking outdoors"`
- `"car driving on highway"`
- `"people talking in meeting"`
- `"sunset over ocean"`
- `"cooking in kitchen"`
- `"children playing in park"`
- `"city traffic at night"`

## Technology Stack

- **CLIP**: OpenAI's vision-language model
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Qdrant**: Vector similarity search engine
- **Streamlit**: Web UI framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **OpenCV**: Video processing
- **HDBSCAN**: Density-based clustering
- **scikit-learn**: Machine learning utilitiesork
- **NumPy**: Numerical computing
- **Pillow**: Image processing

## Acknowledgments

- OpenAI for the CLIP model
- Qdrant team for the vector database
- Hugging Face for model hosting

## Support

For issues and questions, please open a GitHub issue.

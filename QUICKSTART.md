# Photo & Video Search - Quick Start Guide

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Start services**:
```bash
docker-compose up -d
```

2. **Access the app**:
- Open http://localhost:8501 in your browser

3. **Index your content**:
- Place images in `./images/` folder
- Place videos in `./videos/` folder
- Use the web UI to index images and videos

### Option 2: Local Installation

1. **Install dependencies**:
```bash
uv sync
```

2. **Start Qdrant**:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

3. **Index images**:
```bash
uv run python3 image_to_embedding.py
```

4. **Index videos**:
```bash
# Create videos folder and add videos
mkdir -p videos
uv run python3 video_to_embedding.py
```

5. **Start the app**:
```bash
uv run streamlit run app.py
```

## 📋 Testing

Run the test script to verify everything is working:

```bash
# Quick test (skips model loading)
uv run python3 test_video_search.py

# Full test (includes model loading)
uv run python3 test_video_search.py --full
```

## 🎬 Video Search Usage

### Environment Variables

Create a `.env` file or set these variables:

```bash
VIDEO_SAMPLE_RATE=10        # Frames per second to sample
MIN_CLUSTER_SIZE=5          # Minimum frames per cluster
MIN_SAMPLES=3               # HDBSCAN parameter
VIDEO_COLLECTION_NAME=video_embeddings
VIDEO_FOLDER_PATH=./videos
```

### CLI Usage

**Index videos**:
```bash
python3 video_to_embedding.py
```

**Search videos**:
```bash
python3 search_videos.py "person walking"
python3 search_videos.py "car on highway"
python3 search_videos.py "sunset over water"
```

### Web UI Usage

1. Open http://localhost:8501
2. Select "Videos" or "Both" in search mode
3. Enter video folder path and click "🎬 Index Videos"
4. Enter search query and click "🔍 Search"

## 🖼️ Image Search (Original Feature)

### CLI Usage

**Index images**:
```bash
python3 image_to_embedding.py
```

**Search images**:
```bash
python3 search_images.py "aadhaar card"
python3 search_images.py "passport photo"
```

Results are saved to `./search_results/` folder.

## 🔧 Configuration

### Video Processing Parameters

- **VIDEO_SAMPLE_RATE**: Controls frame sampling density
  - Lower = fewer frames, faster processing, less detail
  - Higher = more frames, slower processing, more detail
  - Recommended: 5-15 FPS

- **MIN_CLUSTER_SIZE**: Minimum frames to form a cluster
  - Lower = more granular segments
  - Higher = fewer, larger segments
  - Recommended: 3-10

- **MIN_SAMPLES**: HDBSCAN density parameter
  - Lower = more clusters
  - Higher = fewer clusters
  - Recommended: 2-5

### GPU Settings

The system automatically detects and uses GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 📊 How Video Search Works

1. **Ingestion**: Videos are opened and frames are extracted at the configured sample rate
2. **Embedding**: Each frame is processed through CLIP to generate 768-dimensional vectors
3. **Clustering**: HDBSCAN groups similar frames into semantic segments
4. **Pooling**: Each cluster is represented by a mean-pooled embedding
5. **Indexing**: Pooled embeddings are stored in Qdrant with metadata
6. **Search**: Text queries are embedded and matched against video segments

This approach allows:
- Multiple embeddings per video (one per scene/segment)
- Efficient search across large video collections
- Scene-level precision in results

## 🐛 Troubleshooting

### Qdrant not accessible
```bash
docker ps | grep qdrant
# If not running:
docker-compose up -d qdrant
```

### CUDA out of memory
Reduce batch size or sample rate:
```bash
export VIDEO_SAMPLE_RATE=5  # Sample fewer frames
```

### OpenCV not installed
```bash
uv add opencv-python
```

### Model download slow
Models are cached after first download in `~/.cache/huggingface/`

## 📚 Additional Resources

- Full documentation: See README.md
- Video embeddings module: ./video_embeddings/
- Example queries: See README.md
- Docker configuration: docker-compose.yml

## 🎯 Example Workflow

1. Add videos to `./videos/` folder
2. Run: `python3 video_to_embedding.py`
3. Wait for indexing to complete
4. Run: `python3 search_videos.py "your search query"`
5. Or use Streamlit UI for interactive search

## 💡 Tips

- Start with a few test videos to verify the pipeline
- Adjust VIDEO_SAMPLE_RATE based on video content (action = higher, static = lower)
- Use the web UI for interactive parameter tuning
- Group similar videos in the same folder for easier management
- Check Qdrant dashboard at http://localhost:6333/dashboard for statistics

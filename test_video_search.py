#!/usr/bin/env python3
"""
Test script to verify the video search implementation
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("✓ Transformers (CLIP)")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL (Pillow)")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from qdrant_client import QdrantClient
        print("✓ Qdrant Client")
    except ImportError as e:
        print(f"✗ Qdrant Client import failed: {e}")
        return False
    
    try:
        import hdbscan
        print(f"✓ HDBSCAN {hdbscan.__version__}")
    except ImportError as e:
        print(f"✗ HDBSCAN import failed: {e}")
        return False
    
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    return True

def test_video_embeddings_module():
    """Test the video embeddings module."""
    print("\n" + "="*60)
    print("Testing Video Embeddings Module...")
    print("="*60)
    
    try:
        from video_embeddings import index_videos, VideoVectorDB, VideoFrameEmbedder
        print("✓ video_embeddings module imports")
    except ImportError as e:
        print(f"✗ video_embeddings import failed: {e}")
        return False
    
    try:
        from video_embeddings.ingest import ingest, sample_video
        print("✓ video_embeddings.ingest")
    except ImportError as e:
        print(f"✗ video_embeddings.ingest import failed: {e}")
        return False
    
    try:
        from video_embeddings.embedding import create_embeddings, VideoFrameEmbedder
        print("✓ video_embeddings.embedding")
    except ImportError as e:
        print(f"✗ video_embeddings.embedding import failed: {e}")
        return False
    
    try:
        from video_embeddings.cluster import cluster_embeddings
        print("✓ video_embeddings.cluster")
    except ImportError as e:
        print(f"✗ video_embeddings.cluster import failed: {e}")
        return False
    
    try:
        from video_embeddings.mean_pool import mean_pool_clusters
        print("✓ video_embeddings.mean_pool")
    except ImportError as e:
        print(f"✗ video_embeddings.mean_pool import failed: {e}")
        return False
    
    try:
        from video_embeddings.vector_db import VideoVectorDB
        print("✓ video_embeddings.vector_db")
    except ImportError as e:
        print(f"✗ video_embeddings.vector_db import failed: {e}")
        return False
    
    try:
        from video_embeddings.orchestrator import index_videos, process_single_video
        print("✓ video_embeddings.orchestrator")
    except ImportError as e:
        print(f"✗ video_embeddings.orchestrator import failed: {e}")
        return False
    
    return True

def test_main_modules():
    """Test main application modules."""
    print("\n" + "="*60)
    print("Testing Main Modules...")
    print("="*60)
    
    try:
        from image_to_embedding import ImageEmbeddingProcessor
        print("✓ image_to_embedding module")
    except ImportError as e:
        print(f"✗ image_to_embedding import failed: {e}")
        return False
    
    try:
        from video_to_embedding import VideoEmbeddingProcessor
        print("✓ video_to_embedding module")
    except ImportError as e:
        print(f"✗ video_to_embedding import failed: {e}")
        return False
    
    return True

def test_qdrant_connection():
    """Test connection to Qdrant."""
    print("\n" + "="*60)
    print("Testing Qdrant Connection...")
    print("="*60)
    
    try:
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        print(f"Connecting to Qdrant at {host}:{port}")
        client = QdrantClient(host=host, port=port)
        
        # Try to get collections
        collections = client.get_collections()
        print(f"✓ Connected to Qdrant successfully")
        print(f"  Collections: {[c.name for c in collections.collections]}")
        
        return True
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        print("  Make sure Qdrant is running:")
        print("    docker-compose up -d qdrant")
        return False

def test_model_loading():
    """Test loading the CLIP model."""
    print("\n" + "="*60)
    print("Testing CLIP Model Loading...")
    print("="*60)
    
    try:
        from video_to_embedding import VideoEmbeddingProcessor
        
        print("Loading CLIP model (this may take a minute)...")
        processor = VideoEmbeddingProcessor()
        print("✓ CLIP model loaded successfully")
        print(f"  Device: {processor.device}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def check_folders():
    """Check if required folders exist."""
    print("\n" + "="*60)
    print("Checking Folders...")
    print("="*60)
    
    folders = {
        "images": "./images",
        "videos": "./videos",
        "search_results": "./search_results"
    }
    
    all_exist = True
    for name, path in folders.items():
        folder_path = Path(path)
        if folder_path.exists():
            print(f"✓ {name}: {path} (exists)")
        else:
            print(f"✗ {name}: {path} (missing - will be created)")
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {path}")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VIDEO SEARCH IMPLEMENTATION TEST")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Install missing dependencies:")
        print("  uv sync")
        return
    
    # Test video embeddings module
    if not test_video_embeddings_module():
        all_passed = False
        print("\n⚠ Video embeddings module has issues")
        return
    
    # Test main modules
    if not test_main_modules():
        all_passed = False
        print("\n⚠ Main modules have issues")
        return
    
    # Check folders
    check_folders()
    
    # Test Qdrant connection
    if not test_qdrant_connection():
        all_passed = False
        print("\n⚠ Qdrant is not available")
    
    # Test model loading (optional, takes time)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        if not test_model_loading():
            all_passed = False
            print("\n⚠ Model loading failed")
    else:
        print("\n" + "="*60)
        print("Skipping Model Loading Test")
        print("  Run with --full flag to test model loading")
        print("="*60)
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Start the application: streamlit run app.py")
        print("  2. Or use CLI: python3 video_to_embedding.py")
        print("  3. Search videos: python3 search_videos.py 'your query'")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding")
    print()

if __name__ == "__main__":
    main()

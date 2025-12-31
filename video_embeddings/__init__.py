"""
Video Embeddings Module

This module provides functionality to index and search videos using CLIP embeddings.
Videos are sampled into frames, which are then embedded, clustered, and mean-pooled
to create multiple representative embeddings per video.
"""

from .orchestrator import index_videos
from .vector_db import VideoVectorDB
from .embedding import VideoFrameEmbedder

__all__ = ['index_videos', 'VideoVectorDB', 'VideoFrameEmbedder']

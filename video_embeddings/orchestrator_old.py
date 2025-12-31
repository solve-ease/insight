from .ingest import ingest
from .embedding import VideoFrameEmbedder
from .cluster import cluster_embeddings
from .mean_pool import mean_pool_clusters
from .vector_db import VideoVectorDB
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

def process_frames_in_batches(frames: List[Image.Image], embedder: VideoFrameEmbedder) -> np.ndarray:
    """
    Process frames in small batches to avoid OOM.
    
    Args:
        frames: List of PIL Image frames
        embedder: VideoFrameEmbedder instance
    
    Returns:
        numpy array of embeddings
    """
    if not frames:
        return np.array([])
    
    all_embeddings = []
    batch_size = embedder.batch_size
    
    logger.info(f"Processing {len(frames)} frames in batches of {batch_size}")
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        embeddings = embedder.batch_frames_to_embeddings(batch_frames)
        
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    return np.array([])

def process_single_video(video_path, frames, embedder, vector_db):
    """Process a single video: embed, cluster, pool, and store."""
    try:
        logger.info(f"Processing video: {video_path}")
        
        if not frames or len(frames) == 0:
            logger.warning(f"No frames for video {video_path}")
            return
        
        logger.info(f"Video {video_path.name} has {len(frames)} frames")
        
        # Create embeddings for all frames using batch processing
        embeddings = process_frames_in_batches(frames, embedder)
        
        if len(embeddings) == 0:
            logger.warning(f"No embeddings generated for {video_path}")
            return
        
        logger.info(f"Generated {len(embeddings)} embeddings for {video_path}")
        
        # Cluster the embeddings
        cluster_labels = cluster_embeddings(embeddings)
        
        if len(cluster_labels) == 0:
            logger.warning(f"No clusters found for {video_path}")
            return
        
        # Mean pool clusters
        pooled_embeddings, cluster_info = mean_pool_clusters(embeddings, cluster_labels)
        
        if not pooled_embeddings:
            logger.warning(f"No pooled embeddings for {video_path}")
            return
        
        logger.info(f"Created {len(pooled_embeddings)} pooled embeddings for {video_path}")
        
        # Prepare metadata for each pooled embedding
        metadata_list = []
        for i, (embedding, info) in enumerate(zip(pooled_embeddings, cluster_info)):
            metadata = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'cluster_id': info['cluster_id'],
                'num_frames_in_cluster': info['num_frames'],
                'frame_indices': info['frame_indices'],
                'total_frames': len(frames),
                'embedding_index': i
            }
            metadata_list.append(metadata)
        
        logger.info(f"Processing {len(sampled_videos)} videos")
        
        # Process each video
        for idx, (video_path, frames) in enumerate(sampled_videos, 1):
            logger.info(f"[Video {idx}/{len(sampled_videos)}] {video_path.name}")
            process_single_video(video_path, frames, embedder, vector_db)
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        raise

def index_videos(folder_path: str, qdrant_host: str = "localhost", 
                qdrant_port: int = 6333, collection_name: str = "video_embeddings"):
    """
    Index all videos in a folder.
    
    Args:
        folder_path: Path to folder containing videos
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection
    """
    try:
        logger.info(f"Starting video indexing from folder: {folder_path}")
        
        # Initialize components
        embedder = VideoFrameEmbedder()
        vector_db = VideoVectorDB(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
        
        # Ensure collection exists
        vector_db.check_collection()
        
        # Ingest videos and sample frames
        sampled_videos = ingest(folder_path)
        
        if not sampled_videos:
            logger.warning("No videos found to process")
            return
        
        logger.info(f"Processing {len(sampled_videos)} videos")
        
        # Process each video
        for video_path, frames in sampled_videos:
            asyncio.run(process_single_video(video_path, frames, embedder, vector_db))
        
        logger.info(f"Successfully indexed {len(sampled_videos)} videos")
        
        # Get collection stats
        collection_info = vector_db.vector_db.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' now has {collection_info.points_count} embeddings")
        
    except Exception as e:
        logger.error(f"Error indexing videos from folder {folder_path}: {e}")
        raise
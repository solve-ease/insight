import torch
import os
from pathlib import Path
import logging

# Set environment variables for CUDA
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from video_embeddings import index_videos, VideoVectorDB
from video_embeddings.embedding import VideoFrameEmbedder
from image_to_embedding import ImageEmbeddingProcessor
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoEmbeddingProcessor:
    """
    Unified processor for both images and videos.
    Handles video frame sampling, embedding generation, clustering, and search.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", batch_size: Optional[int] = None, shared_model=None):
        """
        Initialize the video embedding processor.
        
        Args:
            model_name: CLIP model name
            batch_size: Batch size for processing
            shared_model: Optional ImageEmbeddingProcessor to share the CLIP model with (saves GPU memory)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Use environment variable or default based on device
        if batch_size is None:
            batch_size = int(os.getenv("VIDEO_BATCH_SIZE", "4" if self.device == "cuda" else "8"))
        
        logger.info(f"Initializing with batch_size: {batch_size}")
        
        # Check if we can share the model from ImageEmbeddingProcessor
        if shared_model is not None and hasattr(shared_model, 'model') and hasattr(shared_model, 'processor'):
            logger.info("Using shared CLIP model from ImageEmbeddingProcessor (saves GPU memory)")
            # Create a lightweight embedder that uses the shared model
            self.embedder = VideoFrameEmbedder(
                model_name=model_name, 
                batch_size=batch_size,
                shared_clip_model=shared_model.model,
                shared_clip_processor=shared_model.processor
            )
        else:
            # Initialize a new embedder with its own model
            logger.info("Initializing new CLIP model for video processing")
            self.embedder = VideoFrameEmbedder(model_name=model_name, batch_size=batch_size)
        
        logger.info("Video embedding processor initialized")
    
    def process_videos_to_qdrant(
        self,
        folder_path: str,
        collection_name: str = "video_embeddings",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Process all videos in a folder and index them in Qdrant.
        
        Args:
            folder_path: Path to folder containing videos
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        logger.info(f"Starting video processing for folder: {folder_path}")
        
        try:
            index_videos(
                folder_path=folder_path,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                collection_name=collection_name,
                embedder=self.embedder
            )
            logger.info("Video processing completed successfully")
        except Exception as e:
            logger.error(f"Error processing videos: {e}")
            raise
    
    def search_videos_by_text(
        self,
        query_text: str,
        collection_name: str = "video_embeddings",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        limit: int = 10
    ):
        """
        Search for videos using text query.
        
        Args:
            query_text: Text query to search for
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            limit: Number of results to return
        
        Returns:
            List of search results with video information
        """
        try:
            # Generate text embedding using the video embedder
            text_embedding = self.embedder.processor(
                text=[query_text], 
                return_tensors="pt", 
                padding=True
            )
            text_embedding = {k: v.to(self.embedder.device) for k, v in text_embedding.items()}
            
            with torch.no_grad():
                text_features = self.embedder.model.get_text_features(**text_embedding)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            query_embedding = text_features.cpu().numpy()[0]
            
            # Search in vector database
            vector_db = VideoVectorDB(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
            search_results = vector_db.search_videos(query_embedding, limit=limit)
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "video_path": result.payload["video_path"],
                    "video_name": result.payload["video_name"],
                    "cluster_id": result.payload["cluster_id"],
                    "num_frames_in_cluster": result.payload["num_frames_in_cluster"],
                    "frame_indices": result.payload.get("frame_indices", []),
                    "total_frames": result.payload.get("total_frames", 0),
                    "score": result.score,
                    "embedding_index": result.payload.get("embedding_index", 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            raise
    
    def get_collection_stats(
        self,
        collection_name: str = "video_embeddings",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Get statistics about the video collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            vector_db = VideoVectorDB(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
            collection_info = vector_db.vector_db.get_collection(collection_name)
            
            return {
                "collection_name": collection_name,
                "total_embeddings": collection_info.points_count,
                "vector_dimension": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise


if __name__ == "__main__":
    # Configuration
    VIDEO_FOLDER_PATH = os.getenv("VIDEO_FOLDER_PATH", "./videos")
    COLLECTION_NAME = os.getenv("VIDEO_COLLECTION_NAME", "video_embeddings")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Initialize processor
    processor = VideoEmbeddingProcessor()
    
    # Process videos
    if Path(VIDEO_FOLDER_PATH).exists():
        print(f"\n{'='*60}")
        print(f"Processing videos from: {VIDEO_FOLDER_PATH}")
        print(f"{'='*60}\n")
        
        processor.process_videos_to_qdrant(
            folder_path=VIDEO_FOLDER_PATH,
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT
        )
        
        print("\n✓ Video processing complete!")
        
        # Show stats
        stats = processor.get_collection_stats(
            collection_name=COLLECTION_NAME,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT
        )
        
        print(f"\nCollection Statistics:")
        print(f"  Name: {stats['collection_name']}")
        print(f"  Total Embeddings: {stats['total_embeddings']}")
        print(f"  Vector Dimension: {stats['vector_dimension']}")
        print(f"  Distance Metric: {stats['distance_metric']}")
    else:
        print(f"Error: Video folder not found: {VIDEO_FOLDER_PATH}")
        print(f"Please create the folder or set VIDEO_FOLDER_PATH environment variable")

import hdbscan
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Cluster video frame embeddings using HDBSCAN.
    
    Args:
        embeddings: Array of embeddings (n_frames, embedding_dim)
    
    Returns:
        Array of cluster labels
    """
    try:
        min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "2"))
        min_samples = int(os.getenv("MIN_SAMPLES", "3"))
        
        logger.info(f"Clustering {len(embeddings)} embeddings")
        
        # Normalize embeddings for cosine similarity (euclidean on normalized = cosine)
        # CLIP embeddings are already normalized, but ensure it
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
            metric="euclidean"  # On normalized vectors, euclidean distance = cosine distance
        )
        
        cluster_labels = clusterer.fit_predict(normalized_embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return cluster_labels
        
    except Exception as e:
        logger.error(f"Error clustering embeddings: {e}")
        return np.array([])
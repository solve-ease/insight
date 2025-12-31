import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

def mean_pool_clusters(embeddings: np.ndarray, cluster_labels: np.ndarray) -> List[np.ndarray]:
    """
    Mean pool embeddings within each cluster.
    
    Args:
        embeddings: Array of embeddings (n_frames, embedding_dim)
        cluster_labels: Array of cluster labels for each frame
    
    Returns:
        List of mean-pooled embeddings, one per cluster
    """
    try:
        unique_labels = np.unique(cluster_labels)
        
        pooled_embeddings = []
        cluster_info = []
        
        for label in unique_labels:
            # Get all embeddings for this cluster (including noise points with label -1)
            cluster_mask = cluster_labels == label
            cluster_embeddings = embeddings[cluster_mask]
            
            if label == -1:
                # Noise points: treat each frame individually (no pooling)
                # This preserves rare/unique events
                for idx, frame_embedding in enumerate(cluster_embeddings):
                    frame_idx = np.where(cluster_mask)[0][idx]
                    # Normalize
                    normalized_embedding = frame_embedding / np.linalg.norm(frame_embedding)
                    
                    pooled_embeddings.append(normalized_embedding)
                    cluster_info.append({
                        'cluster_id': -1,
                        'num_frames': 1,
                        'frame_indices': [int(frame_idx)],
                        'is_noise': True
                    })
                
                logger.info(f"Noise points: {len(cluster_embeddings)} frames added individually (rare events)")
            else:
                # Regular clusters: mean pool
                mean_embedding = np.mean(cluster_embeddings, axis=0)
                # Normalize
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                
                pooled_embeddings.append(mean_embedding)
                cluster_info.append({
                    'cluster_id': int(label),
                    'num_frames': int(np.sum(cluster_mask)),
                    'frame_indices': np.where(cluster_mask)[0].tolist(),
                    'is_noise': False
                })
                
                logger.info(f"Cluster {label}: {np.sum(cluster_mask)} frames pooled")
        
        return pooled_embeddings, cluster_info
        
    except Exception as e:
        logger.error(f"Error in mean pooling: {e}")
        return [], []

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import os
import hdbscan
import cv2

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", video_sample_rate = 1, batch_size: int = 64):
        """Initialize the CLIP model and processor."""
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Load model with explicit device (using float32 to avoid cuBLAS issues)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()
        self.batch_size = batch_size
        self.video_sample_rate = video_sample_rate
    
    def image_to_embedding(self, image_path: str) -> np.ndarray:
        """
        Convert an image to embedding using CLIP model.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy array: Image embedding vector
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
        
    
    def video_to_embedding(self, video_path: str) -> list[np.ndarray]:

        # sample the video
        sampled_video = self.sample_video(Path(video_path))

        video_embedings = []

        for i in sampled_video:
            video_embedings.append(
                self.image_to_embedding(i)
            )
        
        cluster_labels = self.cluster_embeddings(embeddings=np.ndarray(video_embedings))

        clusters: dict[int , list] = {}
        labels = []

        for i in cluster_labels:
            label = int(cluster_labels[i])

            if label not in labels:
                labels.append(label)
            
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append(video_embedings[i])
        
        # now we have clustered emebdings in this dict with a labels list with all the labels in it

        final_embeddings = []

        for i in labels:
            if i == -1:
                final_embeddings.extend(clusters[i])
            
            else:
                mean_embedding = np.mean(clusters[i])
                normalized_mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

                final_embeddings.append(normalized_mean_embedding)
        
        return final_embeddings

        
    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
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
    
    def sample_video(self, path: Path):
        try:
            frames = []

            path_str = str(path)
            logger.info(f"Sampling video at path: {path_str}")

            vid = cv2.VideoCapture(path_str)
            if not vid.isOpened():
                logger.error(f"Failed to open video file: {path_str}")
                return []

            frame_rate = vid.get(cv2.CAP_PROP_FPS)

            rate = int(frame_rate/self.video_sample_rate)
            
            # Ensure rate is at least 1
            if rate < 1:
                rate = 1

            # here we can also add temporal data also in the frames, in the future for better embeddings

            frame_count = 0
            while True:
                ret, frame = vid.read()
                if not ret:
                    break

                current_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

                if current_frame % rate == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    logger.info(f"Sampled frame {frame_count} at position {current_frame} from video {path_str}")
                    frames.append(frame)
                    frame_count += 1
            
            vid.release()
            logger.info(f"Sampled {len(frames)} frames from {path_str}")
            return frames
        
        except Exception as e:
            logger.error(f"Error sampling video {path_str}: {e}")
            return []    
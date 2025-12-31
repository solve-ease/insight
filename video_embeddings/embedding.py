import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

class VideoFrameEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", batch_size: Optional[int] = None, 
                 shared_clip_model=None, shared_clip_processor=None):
        """
        Initialize the CLIP model for video frame embeddings.
        
        Args:
            model_name: CLIP model name
            batch_size: Batch size for processing
            shared_clip_model: Optional pre-loaded CLIP model to share (saves GPU memory)
            shared_clip_processor: Optional pre-loaded CLIP processor to share
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Use environment variable or default
        if batch_size is None:
            batch_size = int(os.getenv("VIDEO_BATCH_SIZE", "4" if self.device == "cuda" else "8"))
        
        # Use shared model if provided, otherwise load new one
        if shared_clip_model is not None and shared_clip_processor is not None:
            logger.info("Using shared CLIP model and processor (memory efficient)")
            self.model = shared_clip_model
            self.processor = shared_clip_processor
        else:
            logger.info("Loading new CLIP model")
            self.model = CLIPModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        
        self.model.eval()
        self.batch_size = batch_size
        logger.info(f"Batch size set to: {batch_size}")
    
    def batch_frames_to_embeddings(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Convert a SINGLE batch of frames to embeddings.
        This should only process up to batch_size frames at once.
        
        Args:
            frames: List of PIL Image objects (should be <= batch_size)
        
        Returns:
            numpy array: Frame embedding vectors
        """
        if not frames:
            return np.array([])
        
        # Ensure we don't exceed batch size
        if len(frames) > self.batch_size:
            logger.warning(f"Batch size {len(frames)} exceeds configured batch_size {self.batch_size}, processing first {self.batch_size} only")
            frames = frames[:self.batch_size]
        
        try:
            # Process batch
            inputs = self.processor(images=frames, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings = image_features.cpu().numpy()
            
            # Clear GPU cache after each batch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return embeddings
                
        except Exception as e:
            logger.error(f"Error processing batch of {len(frames)} frames: {e}")
            # Try to free memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise

        return np.array([])
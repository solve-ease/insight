import torch
import os

# Set environment variables to use system CUDA libraries
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import uuid

class ImageEmbeddingProcessor:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", batch_size: int = 64):
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
    
    def batch_image_to_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Convert multiple images to embeddings in batch for better GPU utilization.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            numpy array: Image embedding vectors
        """
        images = []
        valid_indices = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        
        if not images:
            return np.array([])
        
        # Process batch
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding using CLIP model.
        
        Args:
            text: Text string to embed
        
        Returns:
            numpy array: Text embedding vector
        """
        # Tokenize text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize the embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def search_by_text(
        self,
        query_text: str,
        collection_name: str = "image_embeddings",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        limit: int = 10
    ):
        """
        Search for similar images using text query.
        
        Args:
            query_text: Text query to search for
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            limit: Number of results to return
        
        Returns:
            List of search results with image paths and scores
        """
        # Initialize Qdrant client
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Generate text embedding
        text_embedding = self.text_to_embedding(query_text)
        
        # Search in Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=text_embedding.tolist(),
            limit=limit
        ).points
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "image_path": result.payload["image_path"],
                "filename": result.payload["filename"],
                "score": result.score,
                "index": result.payload["index"]
            })
        
        return results
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """
        Recursively get all image files from a folder and its subfolders.
        
        Args:
            folder_path: Path to the folder containing images
        
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.svg', '.ico'}
        image_files = []
        
        # Recursively walk through all subdirectories
        for file in Path(folder_path).rglob('*'):
            if file.is_file() and file.suffix.lower() in image_extensions:
                # Verify it's actually an image by trying to open it
                try:
                    with Image.open(file) as img:
                        img.verify()
                    image_files.append(str(file))
                except Exception:
                    # Skip files that can't be opened as images
                    print(f"Skipping invalid image file: {file.name}")
        
        return sorted(image_files)
    
    def process_folder_to_qdrant(
        self,
        folder_path: str,
        collection_name: str = "image_embeddings",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Process all images in a folder and add them to Qdrant collection.
        
        Args:
            folder_path: Path to folder containing images
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        # Initialize Qdrant client
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Get all image files
        image_files = self.get_image_files(folder_path)
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Generate embedding for first image to get dimension
        sample_embedding = self.image_to_embedding(image_files[0])
        embedding_dim = len(sample_embedding)
        
        # Create or recreate collection
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name} with dimension {embedding_dim}")
        
        # Process and upload images in batches
        points = []
        total_processed = 0
        
        for batch_start in range(0, len(image_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_files))
            batch_paths = image_files[batch_start:batch_end]
            
            try:
                print(f"Processing batch {batch_start//self.batch_size + 1}/{(len(image_files) + self.batch_size - 1)//self.batch_size} ({batch_start+1}-{batch_end}/{len(image_files)})")
                
                # Generate embeddings for batch
                embeddings = self.batch_image_to_embeddings(batch_paths)
                
                # Create points for valid embeddings
                for idx, (image_path, embedding) in enumerate(zip(batch_paths, embeddings)):
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "image_path": image_path,
                            "filename": os.path.basename(image_path),
                            "index": batch_start + idx
                        }
                    )
                    points.append(point)
                
                total_processed += len(embeddings)
                
                # Clear GPU cache after each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Upload in batches of 100
                if len(points) >= 1000:
                    client.upsert(collection_name=collection_name, points=points)
                    print(f"Uploaded {len(points)} embeddings to Qdrant (Total: {total_processed})")
                    points = []
                    
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Try processing individually as fallback
                for image_path in batch_paths:
                    try:
                        embedding = self.image_to_embedding(image_path)
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding.tolist(),
                            payload={
                                "image_path": image_path,
                                "filename": os.path.basename(image_path),
                                "index": batch_start + batch_paths.index(image_path)
                            }
                        )
                        points.append(point)
                        total_processed += 1
                    except Exception as e2:
                        print(f"Error processing {image_path}: {e2}")
        
        # Upload remaining points
        if points:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Uploaded final batch of {len(points)} embeddings")
        
        # Print collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"\nCollection '{collection_name}' statistics:")
        print(f"Total vectors: {collection_info.points_count}")
        print(f"Vector dimension: {embedding_dim}")


if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "./images"  # Replace with your images folder path
    COLLECTION_NAME = "image_embeddings"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    
    # Initialize processor
    processor = ImageEmbeddingProcessor()
    
    # Process folder and upload to Qdrant
    processor.process_folder_to_qdrant(
        folder_path=FOLDER_PATH,
        collection_name=COLLECTION_NAME,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT
    )
    
    print("\n✓ Processing complete!")

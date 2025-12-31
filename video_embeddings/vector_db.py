from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import logging
import uuid
from typing import List

logger = logging.getLogger(__name__)

class VideoVectorDB:
    def __init__(self, host: str = None, port: int = None, collection_name: str = None):
        self.qdrant_host = host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.vector_dim = int(os.getenv("VECTOR_DIMENSIONS", "768"))
        self.collection_name = collection_name or os.getenv("VIDEO_COLLECTION_NAME", "video_embeddings")
        
        try:
            logger.info(f"Establishing connection with Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            self.vector_db = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        except Exception as e:
            logger.error(f"Error connecting to Qdrant Vector DB: {e}")
            raise
    
    def create_collection(self):
        """Create a new collection for video embeddings."""
        try:
            logger.info(f"Creating collection: {self.collection_name}")
            self.vector_db.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info("Collection created successfully")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def check_collection(self):
        """Check if collection exists, create if not."""
        try:
            logger.info(f"Checking if collection {self.collection_name} exists")
            
            if not self.vector_db.collection_exists(self.collection_name):
                logger.info("Collection doesn't exist, creating...")
                self.create_collection()
            else:
                logger.info("Collection exists")
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            raise
    
    def write_embeddings(self, embeddings: List, metadata_list: List[dict]):
        """
        Write multiple embeddings to the vector database.
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries, one per embedding
        """
        try:
            logger.info(f"Writing {len(embeddings)} embeddings to vector db")
            
            self.check_collection()
            
            points = []
            for embedding, metadata in zip(embeddings, metadata_list):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    payload=metadata
                )
                points.append(point)
            
            self.vector_db.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully wrote {len(points)} points to vector db")
            
        except Exception as e:
            logger.error(f"Error writing embeddings to vector db: {e}")
            raise
    
    def search_videos(self, query_embedding, limit: int = 10):
        """
        Search for similar videos using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
        
        Returns:
            List of search results
        """
        try:
            logger.info(f"Searching for similar videos (limit={limit})")
            
            results = self.vector_db.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                limit=limit
            ).points
            
            logger.info(f"Found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            raise
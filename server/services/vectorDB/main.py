import os
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from server.services.embedings.main import EmbeddingProcessor
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class vectorDB():
    def __init__(self, host = "localhost" , port = 6333, collection_name = "Insight_Media_Search"):
        # host = os.getenv("QDRANT_HOST")
        self.client = QdrantClient(host = host, port=port)

        self.collection_name = collection_name

        if not self.client.collection_exists(collection_name=collection_name):
            logger.warning("Collection not found, Creting a new Collection")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=int(os.getenv("EMBEDDING_SIZE","768")),
                    distance=Distance.COSINE,
                )
            )
        
    def add_point(self, embedding: list, path=""):
        try:
            logger.info(f"Adding {len(embedding)} points for path: {path}")

            points = []
            point_ids = []

            # Create a unique ID for each embedding/cluster
            for i in embedding:
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                points.append(PointStruct(
                    id = point_id,
                    vector=i.tolist(),
                    payload={
                        "path": path
                    }
                ))

            try:
                # Upload all points at once for better performance
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"Successfully uploaded {len(points)} points to vector database")
            except Exception as e:
                logger.error(f"Error while uploading point to vector database: {e}")
                traceback.print_exc() 
                return -1
            
            # Return all point IDs as a list
            return point_ids
        
        except Exception as e:
            logger.error(f"Error occured in add_point function: {e}")
            traceback.print_exc()
            return -1
    
    def delete_points(self, points):
        logger.info(f"Deleting {len(points)} points")

        if len(points) == 0:
            return

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=points)
            )
        
        except Exception as e:
            logger.error(f"Error occured while deleting points: {e}")

db = vectorDB()
embedProc = EmbeddingProcessor()

def delete_points(points):
    logger.info({"deleting points"})

    db.delete_points(points)        

def vector_db_add(media_path):
    logger.info("adding  media files to vector database")

    if not isinstance(media_path,Path):
        path = Path(media_path)
    
    else:
        path: Path = media_path
    
    if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        embedding = embedProc.video_to_embedding(path)
    
    else:
        embedding = [embedProc.image_to_embedding(path),]

    return db.add_point(embedding, path=str(media_path))
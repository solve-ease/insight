import os
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList

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
        
    def add_point(self, embedding, path=""):
        try:
            logger.info("Adding a point")

            point_id = uuid.uuid4()

            point = PointStruct(
                id = point_id,
                vector=embedding.tolist(),
                payload={
                    "path": path
                }
            )

            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=[point],
                    batch_size=1,
                )
            except Exception as e:
                logger.error(f"Error while uploading point to vector database: {e}") 
                return -1
            
            return point_id
        
        except Exception as e:
            logger.error(f"Error occured in add_point function: {e}")
    
    def delete_point(self, points):
        logger.info(f"Deleting {len(points)} points")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=points)
            )
        
        except Exception as e:
            logger.error(f"Error occured while deleting points: {e}")


def delete_points(points):
    logger.info({"deleting points"})


def vector_db_add(media):
    logger.info("adding  media files to vector database")

    return uuid.uuid4()
from fastapi import APIRouter
import numpy as np
from server.services.vectorDB.main import vectorDB
import torch
from server.services.embedings.main import EmbeddingProcessor

vecDB = vectorDB()
embedProc = EmbeddingProcessor()

router = APIRouter()

@router.get("/search")
async def search_endpoint(query: str):
    text_embedding = embedProc.text_to_embedding(query)

    result = vecDB.query_points(text_embedding=text_embedding)

    return {"query": query, "results": result}
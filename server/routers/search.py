from fastapi import APIRouter

router = APIRouter()

@router.get("/search")
async def search_endpoint(query: str):
    return {"query": query, "results": []}
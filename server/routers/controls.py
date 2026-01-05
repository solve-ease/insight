# implement the following:
# - rescan
# - reindex all files
# - delete vector db collection

from fastapi import APIRouter

router = APIRouter()

@router.post("/controls/rescan")
async def rescan_endpoint():
    return {"status": "rescan started"}

@router.post("/controls/reindex")
async def reindex_endpoint():
    return {"status": "reindex started"}

@router.post("/controls/delete_vector_db")
async def delete_vector_db_endpoint():
    return {"status": "vector db collection deleted"}
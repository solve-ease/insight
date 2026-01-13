# implement the following:
# - rescan
# - reindex all files
# - delete vector db collection

from hashlib import sha256
import traceback
from fastapi import APIRouter
from server.services.prefilter.main import prefilter
from server.services.fileDB.main import fileDB
from PIL import Image
import logging
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

fileDatabase = fileDB()

@router.get("/controls/rescan")
async def rescan_endpoint():
    try:
        await fileDatabase.connect()
        rescan_start_time = datetime.now(timezone.utc)
        prefilter_res = await prefilter(rescan_start_time)
        hash = list()

        # logger.info(f"prefilter result: {prefilter_res}")
        # checking the remaining files using hashing
        for idx , i in enumerate(prefilter_res):
            with open(i, "rb") as f:
                hash.append((i,sha256(f.read()).hexdigest()))
        
        files_changed = await fileDatabase.check_file_hash(hash)

        return {"status": "rescan completed", "files":f"{files_changed}"}
    
    except Exception as e:
        logger.error(f"Error occured in  /controls/rescan: {e}")
        traceback.print_exc()
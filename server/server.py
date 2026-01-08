from fastapi import FastAPI
import logging
from .routers.controls import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

@app.get("/")
async def health():
    return {"status": "ok"}
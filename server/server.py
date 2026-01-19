from fastapi import FastAPI
import logging
from .routers.controls import router as controlRouter
from .routers.search import router as searchRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(controlRouter)
app.include_router(searchRouter)

@app.get("/")
async def health():
    return {"status": "ok"}
from fastapi import FastAPI
import logging
from .routers.controls import router as controlRouter
from .routers.search import router as searchRouter
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)


app.include_router(controlRouter)
app.include_router(searchRouter)

@app.get("/")
async def health():
    return {"status": "ok"}
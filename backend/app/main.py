from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import recommend
from fastapi.staticfiles import StaticFiles
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images") 

app = FastAPI(title="BeautyAI Backend")

app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")

# CORS middleware pre routera
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/recommend", tags=["Recommend"])

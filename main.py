import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config.settings import settings
from src.database.base import init_db
from routes import document_routes, search_routes, analysis_routes, analytics_routes

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Quick, non-blocking startup sequence
    print("Initializing Database & Core Systems...")
    init_db()
    yield
    # Shutdown sequence

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise AI Research & Knowledge Assistant API with TensorFlow classification, SQLite metadata tracking, RAG citations, and document comparison.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)

@app.get("/")
def root():
    return {
        "status": "online",
        "app_name": settings.PROJECT_NAME,
        "version": "2.0.0",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

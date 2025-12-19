"""
FunGen Web API - Main Application
FastAPI backend for funscript generation and video processing.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.routers import auth, projects, videos, funscripts, processing, devices
from app.websocket import ConnectionManager
from app.models.database import init_db
from app.services.processing_service import ProcessingService
from app.services.video_service import VideoService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# WebSocket connection manager
ws_manager = ConnectionManager()

# Services (initialized in lifespan)
processing_service: ProcessingService = None
video_service: VideoService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global processing_service, video_service

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")

    # Initialize services
    processing_service = ProcessingService(
        upload_dir=settings.upload_dir,
        output_dir=settings.output_dir
    )
    video_service = VideoService(upload_dir=settings.upload_dir)

    # Store services in app state for access in routes
    app.state.processing_service = processing_service
    app.state.video_service = video_service
    app.state.ws_manager = ws_manager

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for FunGen - AI-Powered Funscript Generator",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(videos.router, prefix="/api/videos", tags=["Videos"])
app.include_router(funscripts.router, prefix="/api/funscripts", tags=["Funscripts"])
app.include_router(processing.router, prefix="/api/processing", tags=["Processing"])
app.include_router(devices.router, prefix="/api/devices", tags=["Devices"])

# Mount static files for uploads
os.makedirs(settings.upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "auth": "/api/auth",
            "projects": "/api/projects",
            "videos": "/api/videos",
            "funscripts": "/api/funscripts",
            "processing": "/api/processing",
            "devices": "/api/devices",
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Events sent to client:
    - processing_progress: {job_id, progress, stage, message}
    - processing_complete: {job_id, result}
    - processing_error: {job_id, error}
    - playback_sync: {position_ms, is_playing}
    - device_status: {device_id, status, info}
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            await ws_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

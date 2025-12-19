"""
Service layer for FunGen Web API.
Bridges FastAPI backend with existing processing code.
"""

from .processing_service import ProcessingService
from .video_service import VideoService

__all__ = ["ProcessingService", "VideoService"]

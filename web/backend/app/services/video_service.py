"""
Video Service
Handles video file operations and metadata extraction.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import aiofiles
import uuid

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video file metadata."""
    width: int
    height: int
    fps: float
    duration_ms: int
    frame_count: int
    codec: Optional[str] = None
    bitrate: Optional[int] = None


class VideoService:
    """Service for video file operations."""

    ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

    def __init__(self, upload_dir: str, thumbnail_dir: Optional[str] = None):
        self.upload_dir = Path(upload_dir)
        self.thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir else self.upload_dir / "thumbnails"

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Validate video file.

        Returns:
            (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower()

        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        if file_size > self.MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {self.MAX_FILE_SIZE / (1024**3):.1f}GB"

        return True, ""

    async def save_upload(
        self,
        file_content: bytes,
        original_filename: str
    ) -> tuple[str, str]:
        """
        Save uploaded video file.

        Returns:
            (video_id, file_path)
        """
        video_id = str(uuid.uuid4())
        ext = Path(original_filename).suffix.lower()
        filename = f"{video_id}{ext}"
        file_path = self.upload_dir / filename

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)

        return video_id, str(file_path)

    async def extract_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from video file using OpenCV.
        """
        import cv2

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._extract_metadata_sync,
            video_path
        )

    def _extract_metadata_sync(self, video_path: str) -> VideoMetadata:
        """Synchronous metadata extraction."""
        import cv2

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

            # Try to get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                duration_ms=duration_ms,
                frame_count=frame_count,
                codec=codec if codec.strip() else None,
            )

        finally:
            cap.release()

    async def generate_thumbnail(
        self,
        video_path: str,
        video_id: str,
        time_ms: int = 0,
        width: int = 320
    ) -> str:
        """
        Generate thumbnail from video.

        Returns:
            Path to thumbnail file
        """
        import cv2

        thumbnail_path = self.thumbnail_dir / f"{video_id}_{time_ms}.jpg"

        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._generate_thumbnail_sync,
            video_path,
            str(thumbnail_path),
            time_ms,
            width
        )

        return str(thumbnail_path)

    def _generate_thumbnail_sync(
        self,
        video_path: str,
        output_path: str,
        time_ms: int,
        width: int
    ):
        """Synchronous thumbnail generation."""
        import cv2

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ret, frame = cap.read()

            if not ret:
                # Try first frame if requested time fails
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if not ret:
                raise ValueError("Could not read frame from video")

            # Resize maintaining aspect ratio
            h, w = frame.shape[:2]
            aspect = w / h
            new_width = width
            new_height = int(width / aspect)

            resized = cv2.resize(frame, (new_width, new_height))
            cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        finally:
            cap.release()

    async def delete_video(self, video_path: str, video_id: str):
        """Delete video file and associated thumbnails."""
        # Delete video file
        if os.path.exists(video_path):
            os.remove(video_path)

        # Delete thumbnails
        for thumb in self.thumbnail_dir.glob(f"{video_id}_*.jpg"):
            thumb.unlink()

    def get_stream_path(self, video_path: str) -> str:
        """Get path for video streaming."""
        return video_path

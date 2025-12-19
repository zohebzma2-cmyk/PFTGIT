"""
Videos API Router
Handles video upload and metadata.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
import aiofiles
import cv2

from app.config import settings

router = APIRouter()


class VideoMetadata(BaseModel):
    """Video metadata response."""
    id: str
    filename: str
    path: str
    duration_ms: int
    width: int
    height: int
    fps: float
    frame_count: int
    size_bytes: int


# In-memory storage
videos_db: dict[str, dict] = {}


async def extract_video_metadata(video_path: str) -> dict:
    """Extract metadata from video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
        }
    finally:
        cap.release()


@router.post("/upload", response_model=VideoMetadata)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file."""
    # Validate file type
    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique filename
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{ext}"
    file_path = os.path.join(settings.upload_dir, filename)

    # Save file
    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Extract metadata
    try:
        metadata = await extract_video_metadata(file_path)
    except Exception as e:
        # Clean up file on error
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")

    # Store in database
    video_data = {
        "id": video_id,
        "filename": file.filename,
        "path": file_path,
        "size_bytes": os.path.getsize(file_path),
        **metadata,
    }
    videos_db[video_id] = video_data

    return video_data


@router.get("/{video_id}", response_model=VideoMetadata)
async def get_video(video_id: str):
    """Get video metadata."""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    return videos_db[video_id]


@router.get("/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream video file."""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")

    video = videos_db[video_id]
    return FileResponse(
        video["path"],
        media_type="video/mp4",
        filename=video["filename"],
    )


@router.get("/{video_id}/thumbnail")
async def get_thumbnail(video_id: str, time_ms: int = 0):
    """Get video thumbnail at specified time."""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")

    video = videos_db[video_id]
    thumb_path = os.path.join(settings.upload_dir, f"{video_id}_thumb.jpg")

    # Generate thumbnail if not exists
    if not os.path.exists(thumb_path):
        cap = cv2.VideoCapture(video["path"])
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(thumb_path, frame)
        else:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    return FileResponse(thumb_path, media_type="image/jpeg")


@router.delete("/{video_id}")
async def delete_video(video_id: str):
    """Delete a video."""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")

    video = videos_db[video_id]

    # Delete file
    if os.path.exists(video["path"]):
        os.remove(video["path"])

    # Delete thumbnail if exists
    thumb_path = os.path.join(settings.upload_dir, f"{video_id}_thumb.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)

    del videos_db[video_id]
    return {"message": "Video deleted"}

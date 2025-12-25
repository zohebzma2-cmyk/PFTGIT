"""
Videos API Router
Handles video upload and metadata with persistent database storage.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
import aiofiles
import cv2

from app.config import settings
from app.models.database import get_db
from app.models.video import Video
from app.models.project import Project
from app.auth.dependencies import get_current_active_user
from app.auth.jwt import TokenData

router = APIRouter()


class VideoMetadataResponse(BaseModel):
    """Video metadata response."""
    id: str
    filename: str
    original_filename: str
    duration_ms: Optional[int]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    frame_count: Optional[int]
    file_size: int
    project_id: str
    created_at: str

    class Config:
        from_attributes = True


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


async def get_or_create_default_project(
    db: AsyncSession,
    user_id: str
) -> Project:
    """Get or create a default 'Uploads' project for the user."""
    result = await db.execute(
        select(Project).where(
            Project.owner_id == user_id,
            Project.name == "Uploads"
        )
    )
    project = result.scalar_one_or_none()

    if not project:
        project = Project(
            name="Uploads",
            description="Default project for uploaded videos",
            owner_id=user_id
        )
        db.add(project)
        await db.flush()

    return project


def get_user_upload_dir(user_id: str) -> str:
    """Get user-specific upload directory."""
    user_dir = os.path.join(settings.upload_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


@router.post("/upload", response_model=VideoMetadataResponse)
async def upload_video(
    file: UploadFile = File(...),
    project_id: Optional[str] = Query(None, description="Project ID to add video to"),
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a video file."""
    # Validate file type
    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Get or create project
    if project_id:
        # Verify project belongs to user
        result = await db.execute(
            select(Project).where(
                Project.id == project_id,
                Project.owner_id == current_user.user_id
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
    else:
        # Use default uploads project
        project = await get_or_create_default_project(db, current_user.user_id)

    # Generate unique filename in user's directory
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{ext}"
    user_dir = get_user_upload_dir(current_user.user_id)
    file_path = os.path.join(user_dir, filename)

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
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")

    # Create database record
    video = Video(
        id=video_id,
        filename=filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        mime_type=file.content_type,
        project_id=project.id,
        **metadata
    )

    db.add(video)
    await db.flush()

    return VideoMetadataResponse(
        id=video.id,
        filename=video.filename,
        original_filename=video.original_filename,
        duration_ms=video.duration_ms,
        width=video.width,
        height=video.height,
        fps=video.fps,
        frame_count=video.frame_count,
        file_size=video.file_size,
        project_id=video.project_id,
        created_at=video.created_at.isoformat()
    )


@router.get("/", response_model=List[VideoMetadataResponse])
async def list_videos(
    project_id: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all videos for the current user."""
    query = (
        select(Video)
        .join(Project)
        .where(Project.owner_id == current_user.user_id)
    )

    if project_id:
        query = query.where(Video.project_id == project_id)

    result = await db.execute(query)
    videos = result.scalars().all()

    return [
        VideoMetadataResponse(
            id=v.id,
            filename=v.filename,
            original_filename=v.original_filename,
            duration_ms=v.duration_ms,
            width=v.width,
            height=v.height,
            fps=v.fps,
            frame_count=v.frame_count,
            file_size=v.file_size,
            project_id=v.project_id,
            created_at=v.created_at.isoformat()
        )
        for v in videos
    ]


@router.get("/{video_id}", response_model=VideoMetadataResponse)
async def get_video(
    video_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get video metadata."""
    result = await db.execute(
        select(Video)
        .join(Project)
        .where(
            Video.id == video_id,
            Project.owner_id == current_user.user_id
        )
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return VideoMetadataResponse(
        id=video.id,
        filename=video.filename,
        original_filename=video.original_filename,
        duration_ms=video.duration_ms,
        width=video.width,
        height=video.height,
        fps=video.fps,
        frame_count=video.frame_count,
        file_size=video.file_size,
        project_id=video.project_id,
        created_at=video.created_at.isoformat()
    )


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Stream video file."""
    result = await db.execute(
        select(Video)
        .join(Project)
        .where(
            Video.id == video_id,
            Project.owner_id == current_user.user_id
        )
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video.file_path,
        media_type=video.mime_type or "video/mp4",
        filename=video.original_filename,
    )


@router.get("/{video_id}/thumbnail")
async def get_thumbnail(
    video_id: str,
    time_ms: int = 0,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get video thumbnail at specified time."""
    result = await db.execute(
        select(Video)
        .join(Project)
        .where(
            Video.id == video_id,
            Project.owner_id == current_user.user_id
        )
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Generate thumbnail path
    thumb_filename = f"{video_id}_{time_ms}.jpg"
    thumb_path = os.path.join(settings.thumbnails_dir, current_user.user_id, thumb_filename)
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)

    # Generate thumbnail if not exists
    if not os.path.exists(thumb_path):
        if not os.path.exists(video.file_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        cap = cv2.VideoCapture(video.file_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(thumb_path, frame)
        else:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    return FileResponse(thumb_path, media_type="image/jpeg")


@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a video."""
    result = await db.execute(
        select(Video)
        .join(Project)
        .where(
            Video.id == video_id,
            Project.owner_id == current_user.user_id
        )
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete file
    if os.path.exists(video.file_path):
        os.remove(video.file_path)

    # Delete thumbnail if exists
    thumb_dir = os.path.join(settings.thumbnails_dir, current_user.user_id)
    if os.path.exists(thumb_dir):
        for f in os.listdir(thumb_dir):
            if f.startswith(video_id):
                os.remove(os.path.join(thumb_dir, f))

    # Delete from database
    await db.delete(video)

    return {"message": "Video deleted"}

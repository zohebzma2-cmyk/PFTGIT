"""
Processing API Router
Handles AI-powered funscript generation jobs.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
import uuid
from datetime import datetime
import asyncio

router = APIRouter()


class ProcessingStage(str, Enum):
    """Processing stages."""
    QUEUED = "queued"
    ANALYZING = "analyzing"
    DETECTING = "detecting"
    GENERATING = "generating"
    SMOOTHING = "smoothing"
    COMPLETE = "complete"
    FAILED = "failed"


class ProcessingSettings(BaseModel):
    """Settings for funscript generation."""
    # AI Model settings
    model_name: str = "yolov8"
    confidence_threshold: float = 0.5

    # Detection settings
    detection_fps: int = 10
    roi_enabled: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0

    # Generation settings
    smoothing_factor: float = 0.3
    min_stroke_length: int = 10
    max_stroke_speed: int = 500


class ProcessingJob(BaseModel):
    """Processing job model."""
    id: str
    video_id: str
    funscript_id: Optional[str] = None
    settings: ProcessingSettings
    stage: ProcessingStage
    progress: float
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ProcessingJobCreate(BaseModel):
    """Create processing job request."""
    video_id: str
    settings: Optional[ProcessingSettings] = None


# In-memory job storage
jobs_db: dict[str, dict] = {}


async def run_processing_job(job_id: str):
    """Background task to run processing job."""
    if job_id not in jobs_db:
        return

    job = jobs_db[job_id]
    job["stage"] = ProcessingStage.ANALYZING
    job["started_at"] = datetime.utcnow()
    job["progress"] = 0.0
    job["message"] = "Analyzing video..."

    try:
        # Simulate processing stages
        stages = [
            (ProcessingStage.ANALYZING, "Analyzing video...", 0.2),
            (ProcessingStage.DETECTING, "Detecting motion...", 0.5),
            (ProcessingStage.GENERATING, "Generating funscript...", 0.8),
            (ProcessingStage.SMOOTHING, "Smoothing output...", 0.95),
        ]

        for stage, message, target_progress in stages:
            job["stage"] = stage
            job["message"] = message

            # Simulate progress within stage
            while job["progress"] < target_progress:
                await asyncio.sleep(0.5)
                job["progress"] = min(job["progress"] + 0.05, target_progress)

        # Complete
        job["stage"] = ProcessingStage.COMPLETE
        job["progress"] = 1.0
        job["message"] = "Processing complete!"
        job["completed_at"] = datetime.utcnow()
        job["funscript_id"] = str(uuid.uuid4())  # Would be real funscript ID

    except Exception as e:
        job["stage"] = ProcessingStage.FAILED
        job["error"] = str(e)
        job["message"] = f"Processing failed: {e}"


@router.post("/jobs", response_model=ProcessingJob)
async def create_job(data: ProcessingJobCreate, background_tasks: BackgroundTasks):
    """Create a new processing job."""
    job_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "video_id": data.video_id,
        "funscript_id": None,
        "settings": data.settings.model_dump() if data.settings else ProcessingSettings().model_dump(),
        "stage": ProcessingStage.QUEUED,
        "progress": 0.0,
        "message": "Queued for processing",
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    jobs_db[job_id] = job

    # Start background processing
    background_tasks.add_task(run_processing_job, job_id)

    return job


@router.get("/jobs", response_model=List[ProcessingJob])
async def list_jobs():
    """List all processing jobs."""
    return list(jobs_db.values())


@router.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_job(job_id: str):
    """Get processing job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job["stage"] in [ProcessingStage.COMPLETE, ProcessingStage.FAILED]:
        raise HTTPException(status_code=400, detail="Job already finished")

    job["stage"] = ProcessingStage.FAILED
    job["error"] = "Cancelled by user"
    job["message"] = "Job cancelled"

    return {"message": "Job cancelled"}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs_db[job_id]
    return {"message": "Job deleted"}

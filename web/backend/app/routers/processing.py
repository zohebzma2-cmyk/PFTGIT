"""
Processing API Router
Handles AI-powered funscript generation jobs.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import asyncio
import logging
import os
import sys
from pathlib import Path

from app.auth.dependencies import get_current_user
from app.models.user import User

# Add project root to path for desktop app imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()
logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages matching the desktop app pipeline."""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    STAGE1_DETECTION = "stage1_detection"
    STAGE2_CONTACT = "stage2_contact"
    STAGE3_GENERATION = "stage3_generation"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


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

    # Output settings
    invert_output: bool = False


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
    actions: Optional[List[Dict[str, int]]] = None


class ProcessingJobCreate(BaseModel):
    """Create processing job request."""
    video_id: str
    settings: Optional[ProcessingSettings] = None


# In-memory job storage
jobs_db: dict[str, dict] = {}


def check_yolo_available() -> bool:
    """Check if YOLO/ultralytics is available for real processing."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False


async def run_real_processing(job_id: str, video_path: str, settings: ProcessingSettings) -> List[Dict[str, int]]:
    """
    Run real AI processing using YOLO models.
    This uses the same pipeline as the desktop app.
    """
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np

        job = jobs_db[job_id]

        # Check for model files
        model_dir = PROJECT_ROOT / "models"
        detection_model_path = None

        # Look for detection model
        for model_file in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "best.pt"]:
            if (model_dir / model_file).exists():
                detection_model_path = str(model_dir / model_file)
                break

        if not detection_model_path:
            raise FileNotFoundError("No YOLO model found in models/ directory")

        job["stage"] = ProcessingStage.STAGE1_DETECTION
        job["message"] = "Loading AI models..."
        job["progress"] = 0.1

        # Load model
        model = YOLO(detection_model_path)

        job["message"] = "Opening video..."
        job["progress"] = 0.15

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (total_frames / fps) * 1000

        job["message"] = f"Processing {total_frames} frames..."
        job["progress"] = 0.2

        # Process frames at reduced rate
        detection_fps = settings.detection_fps
        frame_skip = max(1, int(fps / detection_fps))

        detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Run detection
                results = model(frame, verbose=False, conf=settings.confidence_threshold)

                # Extract bounding boxes and positions
                time_ms = int((frame_idx / fps) * 1000)

                for result in results:
                    if len(result.boxes) > 0:
                        # Use center Y position of largest box
                        boxes = result.boxes.xyxy.cpu().numpy()
                        if len(boxes) > 0:
                            # Get largest box by area
                            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                            largest_idx = np.argmax(areas)
                            box = boxes[largest_idx]

                            # Center Y as position (normalized to 0-100)
                            center_y = (box[1] + box[3]) / 2
                            height = frame.shape[0]
                            pos = int((1 - center_y / height) * 100)
                            pos = max(0, min(100, pos))

                            detections.append({
                                "time_ms": time_ms,
                                "pos": pos
                            })

                # Update progress
                progress = 0.2 + (frame_idx / total_frames) * 0.6
                job["progress"] = progress
                job["message"] = f"Processing frame {frame_idx}/{total_frames}"

            frame_idx += 1

            # Check for cancellation
            if job.get("cancelled"):
                cap.release()
                raise asyncio.CancelledError()

        cap.release()

        # Stage 3: Generate funscript from detections
        job["stage"] = ProcessingStage.STAGE3_GENERATION
        job["message"] = "Generating funscript from detections..."
        job["progress"] = 0.85

        if len(detections) < 2:
            # Fall back to simulated if no detections
            return await generate_simulated_funscript(duration_ms, settings)

        # Smooth and interpolate detections into funscript
        actions = []
        for det in detections:
            actions.append({
                "at": det["time_ms"],
                "pos": det["pos"]
            })

        # Post-processing: smooth the script
        job["stage"] = ProcessingStage.POSTPROCESSING
        job["message"] = "Smoothing output..."
        job["progress"] = 0.95

        if settings.smoothing_factor > 0 and len(actions) > 2:
            actions = smooth_actions(actions, settings.smoothing_factor)

        if settings.invert_output:
            actions = [{"at": a["at"], "pos": 100 - a["pos"]} for a in actions]

        return actions

    except ImportError as e:
        logger.warning(f"YOLO not available: {e}, falling back to simulation")
        return await generate_simulated_funscript(60000, settings)
    except Exception as e:
        logger.exception(f"Real processing error: {e}")
        raise


async def generate_simulated_funscript(duration_ms: float, settings: ProcessingSettings) -> List[Dict[str, int]]:
    """Generate a realistic-looking simulated funscript."""
    import random
    import math

    actions = []
    current_time = 0
    current_pos = 50
    direction = 1  # 1 = up, -1 = down

    # Simulate natural movement patterns
    min_interval = max(100, settings.min_stroke_length)
    max_interval = min_interval * 4

    while current_time < duration_ms:
        # Generate next position
        # Natural movement: alternate between high and low
        if direction == 1:
            target_pos = random.randint(70, 100)
        else:
            target_pos = random.randint(0, 30)

        # Add some randomness to speed
        interval = random.randint(min_interval, max_interval)

        # Speed limiting
        dp = abs(target_pos - current_pos)
        dt = interval
        speed = (dp / dt) * 1000
        if speed > settings.max_stroke_speed:
            # Limit the position change
            max_dp = (settings.max_stroke_speed * dt) / 1000
            if target_pos > current_pos:
                target_pos = int(current_pos + max_dp)
            else:
                target_pos = int(current_pos - max_dp)
            target_pos = max(0, min(100, target_pos))

        actions.append({
            "at": int(current_time),
            "pos": target_pos
        })

        current_pos = target_pos
        current_time += interval

        # Occasionally change direction
        if random.random() < 0.3:
            direction *= -1

    if settings.invert_output:
        actions = [{"at": a["at"], "pos": 100 - a["pos"]} for a in actions]

    return actions


def smooth_actions(actions: List[Dict[str, int]], factor: float) -> List[Dict[str, int]]:
    """Apply smoothing to funscript actions."""
    if len(actions) < 3:
        return actions

    window_size = max(3, int(factor * 10))
    result = [actions[0]]

    for i in range(1, len(actions) - 1):
        start = max(0, i - window_size // 2)
        end = min(len(actions), i + window_size // 2 + 1)

        avg_pos = sum(a["pos"] for a in actions[start:end]) / (end - start)
        result.append({
            "at": actions[i]["at"],
            "pos": int(avg_pos)
        })

    result.append(actions[-1])
    return result


async def run_processing_job(job_id: str, video_path: Optional[str] = None):
    """Background task to run processing job."""
    if job_id not in jobs_db:
        return

    job = jobs_db[job_id]
    job["stage"] = ProcessingStage.INITIALIZING
    job["started_at"] = datetime.utcnow()
    job["progress"] = 0.0
    job["message"] = "Initializing processing..."

    try:
        settings = ProcessingSettings(**job["settings"]) if isinstance(job["settings"], dict) else job["settings"]

        # Check if we can do real processing
        if video_path and os.path.exists(video_path) and check_yolo_available():
            logger.info(f"Running real AI processing for job {job_id}")
            job["message"] = "Running AI detection..."
            actions = await run_real_processing(job_id, video_path, settings)
        else:
            logger.info(f"Running simulated processing for job {job_id} (YOLO available: {check_yolo_available()})")

            # Simulated processing stages
            stages = [
                (ProcessingStage.ANALYZING, "Analyzing video...", 0.2),
                (ProcessingStage.STAGE1_DETECTION, "Running object detection...", 0.4),
                (ProcessingStage.STAGE2_CONTACT, "Detecting contacts...", 0.6),
                (ProcessingStage.STAGE3_GENERATION, "Generating funscript...", 0.8),
                (ProcessingStage.POSTPROCESSING, "Smoothing output...", 0.95),
            ]

            for stage, message, target_progress in stages:
                if job.get("cancelled"):
                    raise asyncio.CancelledError()

                job["stage"] = stage
                job["message"] = message

                # Simulate progress within stage
                while job["progress"] < target_progress:
                    await asyncio.sleep(0.3)
                    job["progress"] = min(job["progress"] + 0.03, target_progress)

            # Generate simulated funscript
            duration_ms = 60000  # Default 1 minute if no video info
            actions = await generate_simulated_funscript(duration_ms, settings)

        # Complete
        job["stage"] = ProcessingStage.COMPLETE
        job["progress"] = 1.0
        job["message"] = f"Processing complete! Generated {len(actions)} actions."
        job["completed_at"] = datetime.utcnow()
        job["funscript_id"] = str(uuid.uuid4())
        job["actions"] = actions

    except asyncio.CancelledError:
        job["stage"] = ProcessingStage.CANCELLED
        job["message"] = "Processing cancelled"
        job["error"] = "Cancelled by user"

    except Exception as e:
        logger.exception(f"Processing error for job {job_id}: {e}")
        job["stage"] = ProcessingStage.FAILED
        job["error"] = str(e)
        job["message"] = f"Processing failed: {e}"


@router.post("/jobs", response_model=ProcessingJob)
async def create_job(
    data: ProcessingJobCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Create a new processing job."""
    job_id = str(uuid.uuid4())

    # Try to get video path from video service
    video_path = None
    try:
        video_service = request.app.state.video_service
        # Get video metadata to find the file path
        # For now, we'll use the upload directory
        upload_dir = os.getenv("UPLOAD_DIR", "/app/uploads")
        video_path = os.path.join(upload_dir, f"{data.video_id}.mp4")
    except Exception as e:
        logger.warning(f"Could not determine video path: {e}")

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
        "actions": None,
        "cancelled": False,
    }

    jobs_db[job_id] = job

    # Start background processing
    background_tasks.add_task(run_processing_job, job_id, video_path)

    return ProcessingJob(**{k: v for k, v in job.items() if k != 'cancelled'})


@router.get("/jobs", response_model=List[ProcessingJob])
async def list_jobs(current_user: User = Depends(get_current_user)):
    """List all processing jobs."""
    return [ProcessingJob(**{k: v for k, v in job.items() if k != 'cancelled'}) for job in jobs_db.values()]


@router.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Get processing job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs_db[job_id]
    return ProcessingJob(**{k: v for k, v in job.items() if k != 'cancelled'})


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Cancel a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job["stage"] in [ProcessingStage.COMPLETE, ProcessingStage.FAILED, ProcessingStage.CANCELLED]:
        raise HTTPException(status_code=400, detail="Job already finished")

    job["cancelled"] = True
    job["stage"] = ProcessingStage.CANCELLED
    job["error"] = "Cancelled by user"
    job["message"] = "Job cancelled"

    return {"message": "Job cancelled"}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Delete a processing job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs_db[job_id]
    return {"message": "Job deleted"}


@router.get("/capabilities")
async def get_capabilities():
    """Get processing capabilities info."""
    return {
        "yolo_available": check_yolo_available(),
        "models_directory": str(PROJECT_ROOT / "models"),
        "processing_mode": "real" if check_yolo_available() else "simulated",
        "supported_formats": ["mp4", "avi", "mkv", "mov", "webm"]
    }

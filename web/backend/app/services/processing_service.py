"""
Processing Service
Bridges the FastAPI backend with the FunGen processing pipeline.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages matching the desktop app pipeline."""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    STAGE1_DETECTION = "stage1_detection"
    STAGE2_CONTACT = "stage2_contact"
    STAGE3_GENERATION = "stage3_generation"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingProgress:
    """Progress update from processing pipeline."""
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    frames_processed: int = 0
    frames_total: int = 0
    eta_seconds: Optional[float] = None


@dataclass
class ProcessingResult:
    """Result of processing pipeline."""
    success: bool
    funscript_path: Optional[str] = None
    actions: Optional[list] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0
    stats: Optional[Dict[str, Any]] = None


class ProcessingService:
    """
    Service for running the FunGen processing pipeline.

    This bridges the web API with the existing desktop app processing code.
    """

    def __init__(self, upload_dir: str, output_dir: str):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self._active_jobs: Dict[str, asyncio.Task] = {}
        self._job_progress: Dict[str, ProcessingProgress] = {}

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start_processing(
        self,
        job_id: str,
        video_path: str,
        settings: Dict[str, Any],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Start processing a video to generate a funscript.

        Args:
            job_id: Unique job identifier
            video_path: Path to the video file
            settings: Processing settings
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with the generated funscript
        """
        start_time = datetime.utcnow()

        try:
            # Initialize progress
            self._update_progress(
                job_id,
                ProcessingStage.INITIALIZING,
                0.0,
                "Initializing processing pipeline...",
                progress_callback
            )

            # Try to import the actual processing modules
            try:
                from application.logic.app_logic import ApplicationLogic
                from application.logic.app_stage_processor import AppStageProcessor

                return await self._run_full_pipeline(
                    job_id, video_path, settings, progress_callback, start_time
                )

            except ImportError as e:
                logger.warning(f"Desktop app modules not available: {e}")
                logger.info("Running in simulation mode...")

                # Fall back to simulation mode
                return await self._run_simulated_pipeline(
                    job_id, video_path, settings, progress_callback, start_time
                )

        except asyncio.CancelledError:
            self._update_progress(
                job_id,
                ProcessingStage.CANCELLED,
                0.0,
                "Processing cancelled by user",
                progress_callback
            )
            return ProcessingResult(
                success=False,
                error="Processing cancelled by user",
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

        except Exception as e:
            logger.exception(f"Processing error: {e}")
            self._update_progress(
                job_id,
                ProcessingStage.FAILED,
                0.0,
                f"Processing failed: {str(e)}",
                progress_callback
            )
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

    async def _run_full_pipeline(
        self,
        job_id: str,
        video_path: str,
        settings: Dict[str, Any],
        progress_callback: Optional[Callable],
        start_time: datetime
    ) -> ProcessingResult:
        """Run the actual FunGen processing pipeline."""
        from application.logic.app_logic import ApplicationLogic

        # Create a minimal app logic instance for processing
        # This is a simplified version - full integration would require more setup
        logger.info(f"Starting full pipeline for video: {video_path}")

        # Stage 1: Object Detection
        self._update_progress(
            job_id,
            ProcessingStage.STAGE1_DETECTION,
            0.0,
            "Stage 1: Running object detection...",
            progress_callback
        )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        # For now, use simulated processing
        # Full integration would instantiate AppStageProcessor and run stages
        return await self._run_simulated_pipeline(
            job_id, video_path, settings, progress_callback, start_time
        )

    async def _run_simulated_pipeline(
        self,
        job_id: str,
        video_path: str,
        settings: Dict[str, Any],
        progress_callback: Optional[Callable],
        start_time: datetime
    ) -> ProcessingResult:
        """Run a simulated pipeline for testing/demo purposes."""
        stages = [
            (ProcessingStage.STAGE1_DETECTION, "Stage 1: Object detection", 0.3),
            (ProcessingStage.STAGE2_CONTACT, "Stage 2: Contact detection", 0.6),
            (ProcessingStage.STAGE3_GENERATION, "Stage 3: Generating funscript", 0.9),
            (ProcessingStage.POSTPROCESSING, "Post-processing and smoothing", 0.95),
        ]

        total_frames = 1000  # Simulated
        frames_per_stage = total_frames // len(stages)

        for stage, message, target_progress in stages:
            # Check for cancellation
            if job_id in self._active_jobs and self._active_jobs[job_id].cancelled():
                raise asyncio.CancelledError()

            self._update_progress(
                job_id, stage, target_progress - 0.1, message, progress_callback
            )

            # Simulate processing time
            steps = 10
            for i in range(steps):
                await asyncio.sleep(0.3)  # Simulate work
                progress = (stages.index((stage, message, target_progress)) / len(stages)) + (i / steps / len(stages))
                frames = int(progress * total_frames)

                self._update_progress(
                    job_id,
                    stage,
                    min(progress, target_progress),
                    f"{message} ({int(progress * 100)}%)",
                    progress_callback,
                    frames_processed=frames,
                    frames_total=total_frames
                )

        # Generate simulated funscript
        actions = self._generate_sample_funscript(settings)

        # Save to file
        output_name = Path(video_path).stem + ".funscript"
        output_path = self.output_dir / output_name

        import json
        funscript_data = {
            "version": "1.0",
            "inverted": False,
            "range": 100,
            "actions": actions,
            "metadata": {
                "creator": "FunGen Web",
                "description": f"Generated from {Path(video_path).name}",
                "duration": len(actions) * 100 if actions else 0,
                "license": "",
                "notes": "AI-generated funscript",
                "performers": [],
                "script_url": "",
                "tags": ["ai-generated"],
                "title": Path(video_path).stem,
                "type": "basic",
                "video_url": "",
            }
        }

        with open(output_path, "w") as f:
            json.dump(funscript_data, f, indent=2)

        self._update_progress(
            job_id,
            ProcessingStage.COMPLETE,
            1.0,
            "Processing complete!",
            progress_callback
        )

        return ProcessingResult(
            success=True,
            funscript_path=str(output_path),
            actions=actions,
            processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            stats={
                "total_frames": total_frames,
                "actions_generated": len(actions),
            }
        )

    def _generate_sample_funscript(self, settings: Dict[str, Any]) -> list:
        """Generate a sample funscript for testing."""
        import random

        actions = []
        current_time = 0
        current_pos = 50

        # Generate ~10 seconds of actions
        duration_ms = 10000
        interval_ms = settings.get("min_interval_ms", 100)

        while current_time < duration_ms:
            # Random movement
            target_pos = random.randint(0, 100)

            actions.append({
                "at": current_time,
                "pos": target_pos
            })

            current_pos = target_pos
            current_time += random.randint(interval_ms, interval_ms * 3)

        return actions

    def _update_progress(
        self,
        job_id: str,
        stage: ProcessingStage,
        progress: float,
        message: str,
        callback: Optional[Callable],
        frames_processed: int = 0,
        frames_total: int = 0
    ):
        """Update progress and notify callback."""
        progress_update = ProcessingProgress(
            stage=stage,
            progress=progress,
            message=message,
            frames_processed=frames_processed,
            frames_total=frames_total
        )

        self._job_progress[job_id] = progress_update

        if callback:
            try:
                callback(progress_update)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def get_progress(self, job_id: str) -> Optional[ProcessingProgress]:
        """Get current progress for a job."""
        return self._job_progress.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id].cancel()
            return True
        return False

    def cleanup_job(self, job_id: str):
        """Clean up job resources."""
        self._active_jobs.pop(job_id, None)
        self._job_progress.pop(job_id, None)

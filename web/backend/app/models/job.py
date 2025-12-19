"""
Processing job model for tracking AI generation tasks.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base


class ProcessingJob(Base):
    """Processing job model for AI funscript generation."""

    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Status
    status = Column(String(50), default="queued")  # queued, running, completed, failed, cancelled
    stage = Column(String(50), nullable=True)  # stage1, stage2, stage3
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # Settings used for processing
    settings = Column(JSON, default=dict)

    # Processing stats
    frames_processed = Column(Integer, default=0)
    frames_total = Column(Integer, nullable=True)

    # Relationships
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    project = relationship("Project", back_populates="jobs")

    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="jobs")

    # Result
    funscript_id = Column(String(36), ForeignKey("funscripts.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ProcessingJob {self.id} - {self.status}>"

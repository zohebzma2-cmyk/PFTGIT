"""
Video model for uploaded video files.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base


class Video(Base):
    """Video file model."""

    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # File info
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    mime_type = Column(String(100), nullable=True)

    # Video metadata
    duration_ms = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)

    # Thumbnail
    thumbnail_path = Column(Text, nullable=True)

    # Project relationship
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    project = relationship("Project", back_populates="videos")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    funscripts = relationship("Funscript", back_populates="video")
    jobs = relationship("ProcessingJob", back_populates="video")

    def __repr__(self):
        return f"<Video {self.original_filename}>"

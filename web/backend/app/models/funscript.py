"""
Funscript model for storing generated funscripts.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base


class Funscript(Base):
    """Funscript model."""

    __tablename__ = "funscripts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)

    # File info
    file_path = Column(Text, nullable=True)

    # Funscript data
    actions = Column(JSON, default=list)  # List of {at, pos} points
    script_metadata = Column(JSON, default=dict)  # Funscript metadata

    # Settings
    inverted = Column(Boolean, default=False)
    range_min = Column(Integer, default=0)
    range_max = Column(Integer, default=100)

    # Statistics
    point_count = Column(Integer, default=0)
    duration_ms = Column(Integer, nullable=True)

    # Relationships
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    project = relationship("Project", back_populates="funscripts")

    video_id = Column(String(36), ForeignKey("videos.id"), nullable=True)
    video = relationship("Video", back_populates="funscripts")

    # Source (manual, ai_generated, imported)
    source = Column(String(50), default="manual")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Funscript {self.name}>"

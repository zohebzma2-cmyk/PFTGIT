"""
Application configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    app_name: str = "FunGen API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Server settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))

    # File storage - use PERSISTENT_STORAGE_PATH for Render disk
    # On Render, set PERSISTENT_STORAGE_PATH=/var/data
    storage_base: str = os.getenv("PERSISTENT_STORAGE_PATH", ".")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Database - supports both SQLite and PostgreSQL
    # On Render, set DATABASE_URL to the PostgreSQL connection string
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./fungen.db"
    )

    # Processing
    max_concurrent_jobs: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Ignore extra environment variables to avoid parsing errors
        extra = "ignore"

    @property
    def upload_dir(self) -> str:
        """Directory for uploaded videos."""
        return os.path.join(self.storage_base, "uploads")

    @property
    def output_dir(self) -> str:
        """Directory for generated outputs."""
        return os.path.join(self.storage_base, "output")

    @property
    def thumbnails_dir(self) -> str:
        """Directory for video thumbnails."""
        return os.path.join(self.storage_base, "thumbnails")


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.thumbnails_dir, exist_ok=True)

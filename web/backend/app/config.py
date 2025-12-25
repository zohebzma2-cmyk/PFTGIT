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

    # File storage
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Database - supports both SQLite and PostgreSQL
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


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

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
    debug: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # File storage
    upload_dir: str = "./uploads"
    output_dir: str = "./output"
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Database
    database_url: str = "sqlite+aiosqlite:///./fungen.db"

    # Processing
    max_concurrent_jobs: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

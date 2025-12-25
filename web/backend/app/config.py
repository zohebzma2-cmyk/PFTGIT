"""
Application configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


def get_cors_origins() -> list[str]:
    """Get CORS origins from environment or defaults."""
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins:
        return [origin.strip() for origin in env_origins.split(",")]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    app_name: str = "FunGen API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Server settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))

    # CORS settings - override in production with CORS_ORIGINS env var
    cors_origins: list[str] = get_cors_origins()

    # File storage
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Database - supports both SQLite and PostgreSQL
    # For production, set DATABASE_URL to PostgreSQL connection string
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./fungen.db"
    )

    # Processing
    max_concurrent_jobs: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

"""
Projects API Router
Handles project CRUD operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid
import os

from app.config import settings

router = APIRouter()


class ProjectCreate(BaseModel):
    """Project creation request."""
    name: str
    description: Optional[str] = ""


class ProjectUpdate(BaseModel):
    """Project update request."""
    name: Optional[str] = None
    description: Optional[str] = None


class Project(BaseModel):
    """Project response model."""
    id: str
    name: str
    description: str
    video_path: Optional[str] = None
    funscript_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# In-memory storage (replace with database)
projects_db: dict[str, dict] = {}


@router.get("/", response_model=List[Project])
async def list_projects():
    """List all projects."""
    return list(projects_db.values())


@router.post("/", response_model=Project)
async def create_project(project: ProjectCreate):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    now = datetime.utcnow()

    new_project = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "video_path": None,
        "funscript_path": None,
        "created_at": now,
        "updated_at": now,
    }

    projects_db[project_id] = new_project
    return new_project


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a project by ID."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects_db[project_id]


@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, project: ProjectUpdate):
    """Update a project."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")

    existing = projects_db[project_id]
    if project.name is not None:
        existing["name"] = project.name
    if project.description is not None:
        existing["description"] = project.description
    existing["updated_at"] = datetime.utcnow()

    return existing


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")

    del projects_db[project_id]
    return {"message": "Project deleted"}

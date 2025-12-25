"""
Projects API Router
Handles project CRUD operations with persistent database storage.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.models.database import get_db
from app.models.project import Project
from app.auth.dependencies import get_current_active_user
from app.auth.jwt import TokenData

router = APIRouter()


class ProjectCreate(BaseModel):
    """Project creation request."""
    name: str
    description: Optional[str] = ""


class ProjectUpdate(BaseModel):
    """Project update request."""
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    """Project response model."""
    id: str
    name: str
    description: Optional[str]
    is_public: bool
    video_count: int
    funscript_count: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all projects for the current user."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.videos), selectinload(Project.funscripts))
        .where(Project.owner_id == current_user.user_id)
        .order_by(Project.updated_at.desc())
    )
    projects = result.scalars().all()

    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            is_public=p.is_public,
            video_count=len(p.videos),
            funscript_count=len(p.funscripts),
            created_at=p.created_at.isoformat(),
            updated_at=p.updated_at.isoformat()
        )
        for p in projects
    ]


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    data: ProjectCreate,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new project."""
    project = Project(
        name=data.name,
        description=data.description or "",
        owner_id=current_user.user_id
    )

    db.add(project)
    await db.flush()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        is_public=project.is_public,
        video_count=0,
        funscript_count=0,
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat()
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a project by ID."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.videos), selectinload(Project.funscripts))
        .where(
            Project.id == project_id,
            Project.owner_id == current_user.user_id
        )
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        is_public=project.is_public,
        video_count=len(project.videos),
        funscript_count=len(project.funscripts),
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat()
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdate,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a project."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.videos), selectinload(Project.funscripts))
        .where(
            Project.id == project_id,
            Project.owner_id == current_user.user_id
        )
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description

    project.updated_at = datetime.utcnow()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        is_public=project.is_public,
        video_count=len(project.videos),
        funscript_count=len(project.funscripts),
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat()
    )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a project and all its contents."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.videos))
        .where(
            Project.id == project_id,
            Project.owner_id == current_user.user_id
        )
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Note: Videos and funscripts will be cascade deleted
    # File cleanup should be handled separately or via background task
    await db.delete(project)

    return {"message": "Project deleted"}

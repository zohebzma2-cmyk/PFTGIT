"""
Funscripts API Router
Handles funscript CRUD and export operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
import json
import aiofiles

from app.config import settings

router = APIRouter()


class FunscriptPoint(BaseModel):
    """A single point in a funscript."""
    at: int  # Time in milliseconds
    pos: int  # Position 0-100


class FunscriptMetadata(BaseModel):
    """Funscript metadata."""
    creator: str = "FunGen Web"
    description: str = ""
    duration: int = 0
    license: str = ""
    notes: str = ""
    performers: List[str] = []
    script_url: str = ""
    tags: List[str] = []
    title: str = ""
    type: str = "basic"
    video_url: str = ""
    version: str = "1.0"


class Funscript(BaseModel):
    """Complete funscript model."""
    id: str
    name: str
    video_id: Optional[str] = None
    actions: List[FunscriptPoint]
    metadata: FunscriptMetadata
    inverted: bool = False
    range: int = 100


class FunscriptCreate(BaseModel):
    """Funscript creation request."""
    name: str
    video_id: Optional[str] = None
    actions: List[FunscriptPoint] = []
    metadata: Optional[FunscriptMetadata] = None


class FunscriptUpdate(BaseModel):
    """Funscript update request."""
    name: Optional[str] = None
    actions: Optional[List[FunscriptPoint]] = None
    metadata: Optional[FunscriptMetadata] = None
    inverted: Optional[bool] = None
    range: Optional[int] = None


# In-memory storage
funscripts_db: dict[str, dict] = {}


@router.get("/", response_model=List[Funscript])
async def list_funscripts():
    """List all funscripts."""
    return list(funscripts_db.values())


@router.post("/", response_model=Funscript)
async def create_funscript(data: FunscriptCreate):
    """Create a new funscript."""
    funscript_id = str(uuid.uuid4())

    funscript = {
        "id": funscript_id,
        "name": data.name,
        "video_id": data.video_id,
        "actions": [a.model_dump() for a in data.actions],
        "metadata": data.metadata.model_dump() if data.metadata else FunscriptMetadata().model_dump(),
        "inverted": False,
        "range": 100,
    }

    funscripts_db[funscript_id] = funscript
    return funscript


@router.get("/{funscript_id}", response_model=Funscript)
async def get_funscript(funscript_id: str):
    """Get a funscript by ID."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")
    return funscripts_db[funscript_id]


@router.put("/{funscript_id}", response_model=Funscript)
async def update_funscript(funscript_id: str, data: FunscriptUpdate):
    """Update a funscript."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")

    funscript = funscripts_db[funscript_id]

    if data.name is not None:
        funscript["name"] = data.name
    if data.actions is not None:
        funscript["actions"] = [a.model_dump() for a in data.actions]
    if data.metadata is not None:
        funscript["metadata"] = data.metadata.model_dump()
    if data.inverted is not None:
        funscript["inverted"] = data.inverted
    if data.range is not None:
        funscript["range"] = data.range

    return funscript


@router.delete("/{funscript_id}")
async def delete_funscript(funscript_id: str):
    """Delete a funscript."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")

    del funscripts_db[funscript_id]
    return {"message": "Funscript deleted"}


@router.post("/{funscript_id}/points")
async def add_point(funscript_id: str, point: FunscriptPoint):
    """Add a point to a funscript."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")

    funscript = funscripts_db[funscript_id]
    funscript["actions"].append(point.model_dump())
    # Sort by time
    funscript["actions"].sort(key=lambda x: x["at"])

    return {"message": "Point added", "total_points": len(funscript["actions"])}


@router.delete("/{funscript_id}/points/{time_ms}")
async def delete_point(funscript_id: str, time_ms: int):
    """Delete a point at specified time."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")

    funscript = funscripts_db[funscript_id]
    original_count = len(funscript["actions"])
    funscript["actions"] = [a for a in funscript["actions"] if a["at"] != time_ms]

    if len(funscript["actions"]) == original_count:
        raise HTTPException(status_code=404, detail="Point not found at specified time")

    return {"message": "Point deleted", "total_points": len(funscript["actions"])}


@router.get("/{funscript_id}/export")
async def export_funscript(funscript_id: str):
    """Export funscript as .funscript file."""
    if funscript_id not in funscripts_db:
        raise HTTPException(status_code=404, detail="Funscript not found")

    funscript = funscripts_db[funscript_id]

    # Build funscript format
    export_data = {
        "version": "1.0",
        "inverted": funscript["inverted"],
        "range": funscript["range"],
        "actions": funscript["actions"],
        "metadata": funscript["metadata"],
    }

    # Save to file
    export_path = os.path.join(settings.output_dir, f"{funscript['name']}.funscript")
    async with aiofiles.open(export_path, "w") as f:
        await f.write(json.dumps(export_data, indent=2))

    return FileResponse(
        export_path,
        media_type="application/json",
        filename=f"{funscript['name']}.funscript",
    )


@router.post("/import")
async def import_funscript(file: UploadFile = File(...)):
    """Import a funscript file."""
    if not file.filename.endswith(".funscript"):
        raise HTTPException(status_code=400, detail="File must have .funscript extension")

    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid funscript file: {e}")

    # Create funscript from imported data
    funscript_id = str(uuid.uuid4())
    name = os.path.splitext(file.filename)[0]

    funscript = {
        "id": funscript_id,
        "name": name,
        "video_id": None,
        "actions": data.get("actions", []),
        "metadata": data.get("metadata", FunscriptMetadata().model_dump()),
        "inverted": data.get("inverted", False),
        "range": data.get("range", 100),
    }

    funscripts_db[funscript_id] = funscript
    return funscript

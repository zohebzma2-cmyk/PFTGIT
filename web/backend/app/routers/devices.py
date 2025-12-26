"""
Devices API Router
Handles device connections (Handy, etc.).
"""

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

router = APIRouter()


class DeviceType(str, Enum):
    """Supported device types."""
    HANDY = "handy"
    BUTTPLUG = "buttplug"
    SERIAL = "serial"


class DeviceStatus(str, Enum):
    """Device connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ERROR = "error"


class Device(BaseModel):
    """Device model."""
    id: str
    type: DeviceType
    name: str
    status: DeviceStatus
    connection_key: Optional[str] = None
    firmware_version: Optional[str] = None
    last_error: Optional[str] = None


class HandyConnectRequest(BaseModel):
    """Handy connection request."""
    connection_key: str


class DeviceSyncRequest(BaseModel):
    """Device sync request."""
    funscript_id: str
    video_position_ms: int = 0
    server_time_offset_ms: int = 0


# In-memory device storage
devices_db: dict[str, dict] = {}


@router.get("/", response_model=List[Device])
async def list_devices():
    """List all connected devices."""
    return list(devices_db.values())


@router.post("/handy/connect", response_model=Device)
async def connect_handy(data: HandyConnectRequest):
    """Connect to a Handy device."""
    device_id = f"handy_{data.connection_key[:8]}"

    # In production, this would actually connect to the Handy API
    device = {
        "id": device_id,
        "type": DeviceType.HANDY,
        "name": "The Handy",
        "status": DeviceStatus.CONNECTED,
        "connection_key": data.connection_key,
        "firmware_version": "3.2.0",
        "last_error": None,
    }

    devices_db[device_id] = device
    return device


@router.post("/{device_id}/disconnect")
async def disconnect_device(device_id: str):
    """Disconnect a device."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    del devices_db[device_id]
    return {"message": "Device disconnected"}


@router.post("/{device_id}/sync")
async def sync_device(device_id: str, data: DeviceSyncRequest):
    """Start video sync on a device."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    device = devices_db[device_id]
    device["status"] = DeviceStatus.SYNCING

    # In production, this would upload the funscript and start sync
    return {
        "message": "Sync started",
        "device_id": device_id,
        "funscript_id": data.funscript_id,
        "position_ms": data.video_position_ms,
    }


@router.post("/{device_id}/pause")
async def pause_device(device_id: str):
    """Pause device playback."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    device = devices_db[device_id]
    device["status"] = DeviceStatus.CONNECTED

    return {"message": "Playback paused"}


@router.post("/{device_id}/resume")
async def resume_device(device_id: str, position_ms: int = 0):
    """Resume device playback."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    device = devices_db[device_id]
    device["status"] = DeviceStatus.SYNCING

    return {"message": "Playback resumed", "position_ms": position_ms}


@router.post("/{device_id}/set-position")
async def set_position(device_id: str, position: int):
    """Manually set device position (0-100)."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    if position < 0 or position > 100:
        raise HTTPException(status_code=400, detail="Position must be 0-100")

    # In production, this would send position to device
    return {"message": f"Position set to {position}"}


@router.get("/{device_id}/status", response_model=Device)
async def get_device_status(device_id: str):
    """Get device status."""
    if device_id not in devices_db:
        raise HTTPException(status_code=404, detail="Device not found")

    return devices_db[device_id]

"""
WebSocket connection manager for real-time updates.
"""

from fastapi import WebSocket
from typing import Dict, Set, Any
import logging
import json

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from all subscriptions
        for topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def handle_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type")

        if msg_type == "subscribe":
            topic = data.get("topic")
            if topic:
                if topic not in self.subscriptions:
                    self.subscriptions[topic] = set()
                self.subscriptions[topic].add(websocket)
                await websocket.send_json({"type": "subscribed", "topic": topic})

        elif msg_type == "unsubscribe":
            topic = data.get("topic")
            if topic and topic in self.subscriptions:
                self.subscriptions[topic].discard(websocket)
                await websocket.send_json({"type": "unsubscribed", "topic": topic})

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_to_topic(self, topic: str, message: dict):
        """Broadcast message to clients subscribed to a topic."""
        if topic not in self.subscriptions:
            return

        disconnected = set()
        for connection in self.subscriptions[topic]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def send_processing_progress(self, job_id: str, progress: float, stage: str, message: str = ""):
        """Send processing progress update."""
        await self.broadcast_to_topic(f"job:{job_id}", {
            "type": "processing_progress",
            "job_id": job_id,
            "progress": progress,
            "stage": stage,
            "message": message,
        })

    async def send_processing_complete(self, job_id: str, result: Any):
        """Send processing complete notification."""
        await self.broadcast_to_topic(f"job:{job_id}", {
            "type": "processing_complete",
            "job_id": job_id,
            "result": result,
        })

    async def send_processing_error(self, job_id: str, error: str):
        """Send processing error notification."""
        await self.broadcast_to_topic(f"job:{job_id}", {
            "type": "processing_error",
            "job_id": job_id,
            "error": error,
        })

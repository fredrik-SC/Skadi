"""WebSocket handler for real-time alert broadcasting.

Manages SocketIO connections and broadcasts detection events
to all connected browser clients.
"""

from __future__ import annotations

import logging
from typing import Any

from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)


def register_events(sio: SocketIO) -> None:
    """Register SocketIO event handlers.

    Args:
        sio: The SocketIO instance to register events on.
    """

    @sio.on("connect")
    def handle_connect():
        logger.info("Browser client connected")
        emit("status", {"message": "Connected to Skaði"})

    @sio.on("disconnect")
    def handle_disconnect():
        logger.info("Browser client disconnected")


class AlertBroadcaster:
    """Broadcasts detection alerts to all connected WebSocket clients.

    Used by the scan thread to push new detections to browsers
    in real time.

    Args:
        sio: The SocketIO instance to broadcast on.
    """

    def __init__(self, sio: SocketIO) -> None:
        self._sio = sio

    def broadcast_detections(self, detections: list[dict[str, Any]]) -> None:
        """Broadcast new detection events to all connected clients.

        Args:
            detections: List of detection dicts (from DetectionLog.query()).
        """
        if not detections:
            return
        self._sio.emit("new_detections", {"detections": detections})
        logger.debug("Broadcast %d detection(s) to clients", len(detections))

    def broadcast_status(self, status: dict[str, Any]) -> None:
        """Broadcast scanner status update to all connected clients.

        Args:
            status: Status dict with scanning state, sweep count, etc.
        """
        self._sio.emit("scan_status", status)

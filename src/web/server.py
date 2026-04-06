"""Flask web server for Skaði browser-based GUI.

Serves the real-time alert dashboard and detection history API.
All assets served locally for offline operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from src.detectionlog.database import DetectionLog

logger = logging.getLogger(__name__)

# Module-level SocketIO instance (shared with websocket.py)
socketio = SocketIO()


def create_app(
    detection_log_path: Path,
    web_config: dict[str, Any] | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        detection_log_path: Path to the detections SQLite database.
        web_config: Optional web configuration dict from default.yaml.

    Returns:
        Configured Flask application with SocketIO attached.
    """
    cfg = web_config or {}

    # Flask app with correct template and static directories
    web_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )
    app.config["SECRET_KEY"] = "skadi-local-only"

    # Store the detection log path for route handlers
    app.config["DETECTION_LOG_PATH"] = str(detection_log_path)

    # Initialise SocketIO with the app
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Register routes
    _register_routes(app)

    # Register SocketIO events
    from src.web.websocket import register_events
    register_events(socketio)

    logger.info("Web server configured (templates: %s)", web_dir / "templates")
    return app


def _register_routes(app: Flask) -> None:
    """Register HTTP routes on the Flask app."""

    @app.route("/")
    def index():
        """Serve the main dashboard page."""
        return render_template("index.html")

    @app.route("/api/detections")
    def api_detections():
        """JSON API for detection history.

        Query parameters:
            since: ISO 8601 timestamp filter
            freq_min: Minimum frequency in Hz
            freq_max: Maximum frequency in Hz
            threat_level: Filter by threat level
            limit: Maximum results (default 200)
        """
        log_path = Path(app.config["DETECTION_LOG_PATH"])
        log = DetectionLog(log_path)
        try:
            rows = log.query(
                since=request.args.get("since"),
                freq_min_hz=_float_or_none(request.args.get("freq_min")),
                freq_max_hz=_float_or_none(request.args.get("freq_max")),
                threat_level=request.args.get("threat_level"),
                limit=int(request.args.get("limit", 200)),
            )
        finally:
            log.close()
        return jsonify(rows)

    @app.route("/api/status")
    def api_status():
        """Scanner status endpoint."""
        # Status is updated by the scan thread via app.config
        return jsonify({
            "scanning": app.config.get("SCANNER_ACTIVE", False),
            "last_sweep_time": app.config.get("LAST_SWEEP_TIME"),
            "total_detections": app.config.get("TOTAL_DETECTIONS", 0),
            "sweep_count": app.config.get("SWEEP_COUNT", 0),
        })


def _float_or_none(value: str | None) -> float | None:
    """Convert a string to float, returning None if invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

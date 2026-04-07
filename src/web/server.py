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
        return jsonify({
            "scanning": app.config.get("SCANNER_ACTIVE", False),
            "last_sweep_time": app.config.get("LAST_SWEEP_TIME"),
            "total_detections": app.config.get("TOTAL_DETECTIONS", 0),
            "sweep_count": app.config.get("SWEEP_COUNT", 0),
            "scan_requested": app.config.get("SCAN_REQUESTED", False),
            "error": app.config.get("LAST_ERROR"),
        })

    @app.route("/api/scan/start", methods=["POST"])
    def api_scan_start():
        """Request a scan to start."""
        app.config["SCAN_REQUESTED"] = True
        data = request.get_json(silent=True) or {}
        if "freq_start" in data:
            app.config["SCAN_FREQ_START"] = float(data["freq_start"])
        if "freq_stop" in data:
            app.config["SCAN_FREQ_STOP"] = float(data["freq_stop"])
        return jsonify({"status": "scan_requested"})

    @app.route("/api/scan/stop", methods=["POST"])
    def api_scan_stop():
        """Request scanning to stop."""
        app.config["SCAN_REQUESTED"] = False
        return jsonify({"status": "stop_requested"})

    @app.route("/api/config/detection", methods=["GET", "POST"])
    def api_config_detection():
        """Get or update detection thresholds."""
        from src.config import PROJECT_ROOT, load_config
        config_path = PROJECT_ROOT / "config" / "default.yaml"

        if request.method == "GET":
            cfg = load_config(config_path)
            return jsonify(cfg.get("detection", {}))

        data = request.get_json(silent=True) or {}
        cfg = load_config(config_path)
        detection = cfg.get("detection", {})
        if "threshold_db" in data:
            detection["threshold_db"] = float(data["threshold_db"])
        if "min_bandwidth_hz" in data:
            detection["min_bandwidth_hz"] = float(data["min_bandwidth_hz"])
        cfg["detection"] = detection

        import yaml
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return jsonify({"status": "updated", "detection": detection})

    @app.route("/api/config/exclusions", methods=["GET", "POST", "DELETE"])
    def api_config_exclusions():
        """Manage exclusion list."""
        from src.config import PROJECT_ROOT
        excl_path = PROJECT_ROOT / "config" / "exclusions.yaml"

        import yaml
        if request.method == "GET":
            with excl_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return jsonify(data.get("exclusions") or [])

        if request.method == "POST":
            entry = request.get_json(silent=True) or {}
            with excl_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            exclusions = data.get("exclusions") or []
            if not isinstance(exclusions, list):
                exclusions = []
            exclusions.append(entry)
            data["exclusions"] = exclusions
            with excl_path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            return jsonify({"status": "added", "count": len(exclusions)})

        if request.method == "DELETE":
            idx = request.args.get("index")
            with excl_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            exclusions = data.get("exclusions") or []
            if idx is not None and 0 <= int(idx) < len(exclusions):
                exclusions.pop(int(idx))
            data["exclusions"] = exclusions
            with excl_path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            return jsonify({"status": "deleted", "count": len(exclusions)})

    @app.route("/api/config/threats", methods=["GET", "POST"])
    def api_config_threats():
        """Get or update threat rules."""
        from src.config import PROJECT_ROOT
        threat_path = PROJECT_ROOT / "config" / "threat_levels.yaml"

        import yaml
        if request.method == "GET":
            with threat_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return jsonify(data)

        data = request.get_json(silent=True) or {}
        with threat_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return jsonify({"status": "updated"})

    @app.route("/api/detection/<int:detection_id>")
    def api_detection_detail(detection_id):
        """Get full details for a single detection."""
        log_path = Path(app.config["DETECTION_LOG_PATH"])
        import sqlite3
        conn = sqlite3.connect(str(log_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(dict(row))


def _float_or_none(value: str | None) -> float | None:
    """Convert a string to float, returning None if invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

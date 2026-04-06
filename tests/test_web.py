"""Tests for the web server and WebSocket."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.models import DetectedSignal
from src.detectionlog.database import DetectionLog
from src.web.server import create_app, socketio


def _make_signal(freq_hz: float = 100e6, timestamp: float = 1712400000.0) -> DetectedSignal:
    return DetectedSignal(
        centre_freq_hz=freq_hz,
        bandwidth_hz=200_000,
        peak_power_dbm=-70.0,
        mean_power_dbm=-75.0,
        snr_db=20.0,
        timestamp=timestamp,
        scan_step_freq_hz=freq_hz,
    )


@pytest.fixture
def app_with_data(tmp_path):
    """Create a Flask app with some test detections."""
    db_path = tmp_path / "test.db"
    log = DetectionLog(db_path)
    log.log_signal(
        _make_signal(88e6),
        modulation="FM",
        signal_type="FM Broadcast Radio",
        confidence_score=0.95,
        threat_level="INFORMATIONAL",
    )
    log.log_signal(
        _make_signal(440e6),
        modulation="FSK",
        signal_type="Digital Mobile Radio (DMR)",
        confidence_score=0.85,
        threat_level="LOW",
    )
    log.log_signal(
        _make_signal(10e6),
        modulation="PSK",
        signal_type="STANAG 4285",
        confidence_score=0.90,
        threat_level="HIGH",
    )
    log.close()

    app = create_app(db_path)
    app.config["TESTING"] = True
    return app


class TestWebServer:
    """Tests for Flask routes."""

    def test_index_returns_html(self, app_with_data):
        """GET / returns the dashboard HTML."""
        client = app_with_data.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"SKADI" in resp.data

    def test_api_detections_returns_json(self, app_with_data):
        """GET /api/detections returns JSON array."""
        client = app_with_data.test_client()
        resp = client.get("/api/detections")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_api_detections_has_fields(self, app_with_data):
        """Returned detections have all expected fields."""
        client = app_with_data.test_client()
        resp = client.get("/api/detections")
        data = json.loads(resp.data)
        record = data[0]
        for field in ["frequency_hz", "modulation", "signal_type",
                       "confidence_score", "threat_level", "timestamp_utc"]:
            assert field in record

    def test_api_detections_threat_filter(self, app_with_data):
        """Threat level filter works."""
        client = app_with_data.test_client()
        resp = client.get("/api/detections?threat_level=HIGH")
        data = json.loads(resp.data)
        assert len(data) == 1
        assert data[0]["signal_type"] == "STANAG 4285"

    def test_api_detections_freq_filter(self, app_with_data):
        """Frequency filter works."""
        client = app_with_data.test_client()
        resp = client.get("/api/detections?freq_min=80000000&freq_max=100000000")
        data = json.loads(resp.data)
        assert len(data) == 1
        assert data[0]["signal_type"] == "FM Broadcast Radio"

    def test_api_status(self, app_with_data):
        """GET /api/status returns status dict."""
        client = app_with_data.test_client()
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "scanning" in data
        assert "sweep_count" in data


class TestWebSocket:
    """Tests for WebSocket functionality."""

    def test_socketio_connect(self, app_with_data):
        """Client can connect via SocketIO."""
        client = socketio.test_client(app_with_data)
        assert client.is_connected()
        # Should receive a status message on connect
        received = client.get_received()
        assert any(msg["name"] == "status" for msg in received)
        client.disconnect()

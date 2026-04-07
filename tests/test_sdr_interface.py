"""Tests for SDR interface module.

Uses mocked SoapySDR.Device so tests run without hardware connected.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sdr import SDRConnectionError, SDRStreamError
from src.sdr.interface import SDRInterface


@pytest.fixture
def sdr_config() -> dict:
    """Minimal SDR config for testing."""
    return {
        "driver": "sdrplay",
        "mode": "ST",
        "sample_rate": 2048000,
        "bandwidth": 0,
        "agc": True,
        "gain_reduction": 0,
    }


@pytest.fixture
def mock_device():
    """Create a mock SoapySDR device with standard responses."""
    device = MagicMock()
    device.getHardwareKey.return_value = "RSPduo"
    device.getHardwareInfo.return_value = {
        "sdrplay_api_api_version": "3.150000",
        "sdrplay_api_hw_version": "3",
    }
    device.getSampleRate.return_value = 2048000.0
    device.getFrequency.return_value = 100e6
    device.listAntennas.return_value = (
        "Tuner 1 50 ohm", "Tuner 1 Hi-Z", "Tuner 2 50 ohm"
    )
    device.listGains.return_value = ("IFGR", "RFGR")

    # Mock gain range objects
    ifgr_range = MagicMock()
    ifgr_range.minimum.return_value = 20.0
    ifgr_range.maximum.return_value = 59.0
    rfgr_range = MagicMock()
    rfgr_range.minimum.return_value = 0.0
    rfgr_range.maximum.return_value = 9.0
    device.getGainRange.side_effect = lambda d, c, name: (
        ifgr_range if name == "IFGR" else rfgr_range
    )

    device.listSampleRates.return_value = [2048000.0, 4096000.0]

    freq_range = MagicMock()
    freq_range.minimum.return_value = 1000.0
    freq_range.maximum.return_value = 2000000000.0
    device.getFrequencyRange.return_value = [freq_range]

    # Mock stream operations
    stream_result = MagicMock()
    stream_result.ret = 1024
    device.readStream.return_value = stream_result
    device.setupStream.return_value = MagicMock()

    return device


class TestSDRInterface:
    """Tests for SDRInterface class."""

    @patch("src.sdr.interface.SoapySDR")
    def test_connect_success(self, mock_soapy, sdr_config, mock_device):
        """Device opens and configures successfully."""
        mock_soapy.Device.return_value = mock_device
        mock_soapy.SOAPY_SDR_RX = 0
        mock_soapy.SOAPY_SDR_CF32 = "CF32"

        sdr = SDRInterface(sdr_config)
        sdr.connect()

        assert sdr.connected
        mock_device.setSampleRate.assert_called_once()
        mock_device.setGainMode.assert_called_once()
        mock_device.setAntenna.assert_called_once()

        sdr.disconnect()

    @patch("src.sdr.interface.SoapySDR")
    def test_connect_device_not_found(self, mock_soapy, sdr_config):
        """SDRConnectionError raised when device cannot be opened."""
        mock_soapy.Device.side_effect = RuntimeError("no match")

        sdr = SDRInterface(sdr_config)
        with pytest.raises(SDRConnectionError, match="Failed to open"):
            sdr.connect()

        assert not sdr.connected

    @patch("src.sdr.interface.SoapySDR")
    def test_disconnect_idempotent(self, mock_soapy, sdr_config, mock_device):
        """Calling disconnect() multiple times does not raise."""
        mock_soapy.Device.return_value = mock_device

        sdr = SDRInterface(sdr_config)
        sdr.connect()
        sdr.disconnect()
        sdr.disconnect()  # Second call should be fine

        assert not sdr.connected

    @patch("src.sdr.interface.SoapySDR")
    def test_context_manager(self, mock_soapy, sdr_config, mock_device):
        """Context manager connects on entry and disconnects on exit."""
        mock_soapy.Device.return_value = mock_device

        with SDRInterface(sdr_config) as sdr:
            assert sdr.connected

        assert not sdr.connected

    @patch("src.sdr.interface.SoapySDR")
    def test_tune(self, mock_soapy, sdr_config, mock_device):
        """Tune sets frequency and returns actual frequency."""
        mock_soapy.Device.return_value = mock_device
        mock_soapy.SOAPY_SDR_RX = 0

        sdr = SDRInterface(sdr_config)
        sdr.connect()
        actual = sdr.tune(100e6)

        mock_device.setFrequency.assert_called_once()
        assert actual == 100e6
        sdr.disconnect()

    def test_tune_not_connected(self, sdr_config):
        """Tune raises SDRConnectionError when not connected."""
        sdr = SDRInterface(sdr_config)
        with pytest.raises(SDRConnectionError, match="not connected"):
            sdr.tune(100e6)

    @patch("src.sdr.interface.SoapySDR")
    def test_capture_accumulates_chunks(self, mock_soapy, sdr_config, mock_device):
        """Capture correctly accumulates multiple read chunks."""
        mock_soapy.Device.return_value = mock_device
        mock_soapy.SOAPY_SDR_RX = 0
        mock_soapy.SOAPY_SDR_CF32 = "CF32"

        # Simulate readStream returning 512 samples at a time
        result = MagicMock()
        result.ret = 512
        mock_device.readStream.return_value = result

        sdr = SDRInterface(sdr_config)
        sdr.connect()
        sdr.tune(100e6)
        samples = sdr.capture(1024)

        assert len(samples) == 1024
        assert samples.dtype == np.complex64
        # readStream called twice (2 x 512 = 1024)
        assert mock_device.readStream.call_count == 2
        sdr.disconnect()

    @patch("src.sdr.interface.SoapySDR")
    def test_capture_handles_error(self, mock_soapy, sdr_config, mock_device):
        """Capture raises SDRStreamError on negative return code."""
        mock_soapy.Device.return_value = mock_device
        mock_soapy.SOAPY_SDR_RX = 0
        mock_soapy.SOAPY_SDR_CF32 = "CF32"

        result = MagicMock()
        result.ret = -1
        mock_device.readStream.return_value = result

        sdr = SDRInterface(sdr_config)
        sdr.connect()
        sdr.tune(100e6)
        with pytest.raises(SDRStreamError, match="error code"):
            sdr.capture(1024)

        # Stream deactivated on error, but NOT closed (persistent stream)
        mock_device.deactivateStream.assert_called()
        # closeStream happens at disconnect, not per-capture
        sdr.disconnect()

    def test_capture_not_connected(self, sdr_config):
        """Capture raises SDRConnectionError when not connected."""
        sdr = SDRInterface(sdr_config)
        with pytest.raises(SDRConnectionError, match="not connected"):
            sdr.capture(1024)

    @patch("src.sdr.interface.SoapySDR")
    def test_info(self, mock_soapy, sdr_config, mock_device):
        """Info returns device diagnostic dictionary."""
        mock_soapy.Device.return_value = mock_device
        mock_soapy.SOAPY_SDR_RX = 0

        sdr = SDRInterface(sdr_config)
        sdr.connect()
        device_info = sdr.info()

        assert device_info["hardware_key"] == "RSPduo"
        assert "IFGR" in device_info["gains"]
        assert device_info["gains"]["IFGR"]["min"] == 20.0
        assert device_info["gains"]["IFGR"]["max"] == 59.0
        sdr.disconnect()

    def test_sample_rate_property(self, sdr_config):
        """Sample rate property returns the configured value."""
        sdr = SDRInterface(sdr_config)
        assert sdr.sample_rate == 2048000.0


class TestConfigLoader:
    """Tests for the configuration loader."""

    def test_load_default_config(self):
        """Default config file loads successfully with expected keys."""
        from src.config import load_config

        config = load_config()
        assert "sdr" in config
        assert "scan" in config
        assert "detection" in config
        assert config["sdr"]["driver"] == "sdrplay"
        assert config["sdr"]["sample_rate"] == 2048000

    def test_load_missing_config(self):
        """FileNotFoundError raised for missing config file."""
        from src.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_threat_levels(self):
        """Threat levels config loads with expected structure."""
        from src.config import load_config, PROJECT_ROOT

        config = load_config(PROJECT_ROOT / "config" / "threat_levels.yaml")
        assert "default_threat_level" in config
        assert "rules" in config
        assert config["default_threat_level"] == "MEDIUM"

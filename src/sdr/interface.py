"""SoapySDR connection and IQ capture for SDRPlay RSPduo.

Provides the SDRInterface class which manages device connection,
configuration, tuning, and IQ sample capture in single-tuner mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import SoapySDR
except ImportError as e:
    raise ImportError(
        "SoapySDR Python bindings not found. Ensure SoapySDR is installed "
        "and accessible from the current Python environment."
    ) from e

from . import SDRConnectionError, SDRStreamError

logger = logging.getLogger(__name__)

# SoapySDR constants
_RX = SoapySDR.SOAPY_SDR_RX
_CF32 = SoapySDR.SOAPY_SDR_CF32

# Maximum samples per readStream call (SoapySDR typical limit)
_READ_CHUNK_SIZE = 65536


class SDRInterface:
    """SoapySDR interface for SDRPlay RSPduo in single-tuner mode.

    Manages device connection, configuration, stream setup, and IQ capture.
    Supports the context manager protocol for safe resource cleanup.

    Args:
        sdr_config: Dictionary of SDR configuration values from default.yaml.
            Expected keys: driver, mode, sample_rate, bandwidth, agc,
            gain_reduction.

    Example:
        config = load_config()
        with SDRInterface(config["sdr"]) as sdr:
            sdr.tune(100e6)
            samples = sdr.capture(num_samples=2048000)
    """

    def __init__(self, sdr_config: dict[str, Any]) -> None:
        self._driver: str = sdr_config.get("driver", "sdrplay")
        self._mode: str = sdr_config.get("mode", "ST")
        self._sample_rate: float = float(sdr_config.get("sample_rate", 2_048_000))
        self._bandwidth: float = float(sdr_config.get("bandwidth", 0))
        self._agc: bool = sdr_config.get("agc", True)
        self._gain_reduction: float = float(sdr_config.get("gain_reduction", 0))

        self._device: Any | None = None

    @property
    def connected(self) -> bool:
        """Whether the SDR device is currently connected."""
        return self._device is not None

    @property
    def sample_rate(self) -> float:
        """The configured sample rate in Hz."""
        return self._sample_rate

    def connect(self) -> None:
        """Open and configure the SDR device.

        Connects to the SDR via SoapySDR, sets sample rate, gain mode,
        and antenna selection.

        Raises:
            SDRConnectionError: If the device cannot be opened or configured.
        """
        if self._device is not None:
            logger.warning("Device already connected, disconnecting first")
            self.disconnect()

        # Build the device argument string
        device_args = f"driver={self._driver},mode={self._mode}"
        logger.info("Opening SDR device: %s", device_args)

        try:
            self._device = SoapySDR.Device(device_args)
        except RuntimeError as e:
            raise SDRConnectionError(
                f"Failed to open SDR device with args '{device_args}': {e}"
            ) from e

        try:
            self._configure_device()
        except Exception as e:
            # Clean up on configuration failure
            self._device = None
            raise SDRConnectionError(
                f"Failed to configure SDR device: {e}"
            ) from e

        hw_info = dict(self._device.getHardwareInfo())
        logger.info(
            "SDR connected: %s (API: %s, HW ver: %s)",
            self._device.getHardwareKey(),
            hw_info.get("sdrplay_api_api_version", "unknown"),
            hw_info.get("sdrplay_api_hw_version", "unknown"),
        )

    def _configure_device(self) -> None:
        """Apply sample rate, gain, and antenna settings to the device."""
        dev = self._device

        # Sample rate
        dev.setSampleRate(_RX, 0, self._sample_rate)
        actual_rate = dev.getSampleRate(_RX, 0)
        logger.info(
            "Sample rate: requested=%.0f Hz, actual=%.0f Hz",
            self._sample_rate, actual_rate,
        )

        # Bandwidth (0 = auto)
        if self._bandwidth > 0:
            dev.setBandwidth(_RX, 0, self._bandwidth)
            logger.info("Bandwidth set to %.0f Hz", self._bandwidth)

        # Gain / AGC
        if self._agc:
            dev.setGainMode(_RX, 0, True)
            logger.info("AGC enabled")
        elif self._gain_reduction > 0:
            dev.setGainMode(_RX, 0, False)
            dev.setGain(_RX, 0, "IFGR", self._gain_reduction)
            logger.info("Manual gain: IFGR=%.0f dB", self._gain_reduction)

        # Antenna — default to Tuner 1 50 ohm for single-tuner mode
        dev.setAntenna(_RX, 0, "Tuner 1 50 ohm")
        logger.info("Antenna: Tuner 1 50 ohm")

    def disconnect(self) -> None:
        """Close the SDR device and release resources.

        Safe to call multiple times (idempotent).
        """
        if self._device is not None:
            logger.info("Disconnecting SDR device")
            self._device = None

    def tune(self, frequency_hz: float) -> float:
        """Set the centre frequency of the receiver.

        Args:
            frequency_hz: Target centre frequency in Hz.

        Returns:
            The actual tuned frequency in Hz (may differ slightly from requested).

        Raises:
            SDRConnectionError: If the device is not connected.
        """
        if self._device is None:
            raise SDRConnectionError("Device not connected")

        self._device.setFrequency(_RX, 0, frequency_hz)
        actual_freq = self._device.getFrequency(_RX, 0)
        logger.info(
            "Tuned: requested=%.3f MHz, actual=%.3f MHz",
            frequency_hz / 1e6, actual_freq / 1e6,
        )
        return actual_freq

    def capture(self, num_samples: int) -> np.ndarray:
        """Capture IQ samples from the SDR.

        Sets up a receive stream, reads the requested number of samples
        in chunks, then tears down the stream.

        Args:
            num_samples: Number of complex samples to capture.

        Returns:
            numpy.ndarray of dtype complex64 with shape (num_samples,).

        Raises:
            SDRConnectionError: If the device is not connected.
            SDRStreamError: If the stream read fails.
        """
        if self._device is None:
            raise SDRConnectionError("Device not connected")

        # Set up the receive stream
        try:
            stream = self._device.setupStream(_RX, _CF32)
        except RuntimeError as e:
            raise SDRStreamError(f"Failed to set up stream: {e}") from e

        self._device.activateStream(stream)

        buffer = np.zeros(num_samples, dtype=np.complex64)
        samples_read = 0

        try:
            while samples_read < num_samples:
                remaining = num_samples - samples_read
                chunk_size = min(_READ_CHUNK_SIZE, remaining)
                chunk = np.zeros(chunk_size, dtype=np.complex64)

                result = self._device.readStream(
                    stream, [chunk], chunk_size,
                    timeoutUs=1_000_000,
                )

                if result.ret < 0:
                    raise SDRStreamError(
                        f"readStream error code {result.ret}"
                    )
                if result.ret == 0:
                    logger.warning("readStream returned 0 samples, retrying")
                    continue

                buffer[samples_read:samples_read + result.ret] = chunk[:result.ret]
                samples_read += result.ret
        finally:
            self._device.deactivateStream(stream)
            self._device.closeStream(stream)

        logger.info("Captured %d IQ samples", samples_read)
        return buffer

    def info(self) -> dict[str, Any]:
        """Return diagnostic information about the connected device.

        Returns:
            Dictionary with device hardware info, supported antennas,
            gain ranges, sample rate ranges, and frequency ranges.

        Raises:
            SDRConnectionError: If the device is not connected.
        """
        if self._device is None:
            raise SDRConnectionError("Device not connected")

        dev = self._device
        return {
            "hardware_key": dev.getHardwareKey(),
            "hardware_info": dict(dev.getHardwareInfo()),
            "antennas": list(dev.listAntennas(_RX, 0)),
            "gains": {
                name: {
                    "min": dev.getGainRange(_RX, 0, name).minimum(),
                    "max": dev.getGainRange(_RX, 0, name).maximum(),
                }
                for name in dev.listGains(_RX, 0)
            },
            "sample_rates": list(dev.listSampleRates(_RX, 0)),
            "frequency_range": [
                {"min": r.minimum(), "max": r.maximum()}
                for r in dev.getFrequencyRange(_RX, 0)
            ],
        }

    def __enter__(self) -> SDRInterface:
        """Connect to the SDR device on context entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Disconnect from the SDR device on context exit."""
        self.disconnect()

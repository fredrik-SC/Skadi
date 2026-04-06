"""SDR interface layer for Skaði.

Manages hardware connection to SDRPlay RSPduo via SoapySDR.
"""


class SDRError(Exception):
    """Base exception for SDR-related errors."""


class SDRConnectionError(SDRError):
    """Failed to connect to or configure the SDR device."""


class SDRStreamError(SDRError):
    """Error during IQ stream capture."""

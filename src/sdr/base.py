"""Structural interface shared by all SDR sources.

Defines the ``SdrSource`` protocol — the minimal surface the scanner depends
on. Any object that can tune, capture IQ, report its sample rate, and act as a
context manager satisfies it. This lets a live SoapySDR device
(:class:`~src.sdr.interface.SDRInterface`), a recording decorator
(:class:`~src.sdr.recorder.RecordingSDR`), and a file-backed replay source
(:class:`~src.sdr.replay.ReplaySource`) be used interchangeably.

The scanner (:class:`~src.sdr.scanner.SpectrumScanner`) only ever calls
``tune()`` and ``capture()`` and reads ``sample_rate`` from configuration, so
the protocol is deliberately tiny.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SdrSource(Protocol):
    """Minimal IQ source interface used throughout the pipeline.

    Implementations must support the context manager protocol so callers can
    use ``with source as sdr:`` regardless of whether the backing source is
    real hardware or a file on disk.
    """

    @property
    def sample_rate(self) -> float:
        """Effective sample rate of captured IQ in Hz."""
        ...

    @property
    def connected(self) -> bool:
        """Whether the source is ready to capture."""
        ...

    def tune(self, frequency_hz: float) -> float:
        """Set the centre frequency and return the actual tuned frequency."""
        ...

    def capture(self, num_samples: int) -> np.ndarray:
        """Return ``num_samples`` complex64 IQ samples at the tuned frequency."""
        ...

    def __enter__(self) -> "SdrSource":
        """Enter the context, readying the source for capture."""
        ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context, releasing any held resources."""
        ...

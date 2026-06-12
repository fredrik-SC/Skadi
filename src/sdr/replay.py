"""Replay a recorded sweep as a drop-in :class:`~src.sdr.base.SdrSource`.

``ReplaySource`` reads a recording session directory (produced by
:class:`~src.sdr.recorder.SigmfRecorder`) and serves the stored IQ back through
the exact same pipeline, with no hardware in the loop. This makes the whole
detect → fingerprint → classify chain deterministically testable offline and is
the foundation for building a labelled corpus.

The scanner computes its own step frequencies from configuration, so the caller
(``main.py``) restores the manifest's ``scan_config``/``sdr_config`` before
constructing the scanner. ``tune()`` then maps each requested frequency to the
nearest recorded step; a request with no recorded step within tolerance is a
loud error rather than silent empty data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.sdr import SDRError
from src.sdr.recorder import MANIFEST_NAME
from src.sdr.sigmf import read_iq

logger = logging.getLogger(__name__)


class ReplaySource:
    """File-backed SDR source replaying a recorded session.

    Args:
        session_dir: Directory containing ``session.json`` and per-step SigMF
            recordings.
        freq_tolerance_hz: Maximum distance between a requested tune frequency
            and the nearest recorded step centre. Defaults to half the recorded
            step size (so any frequency inside a recorded step matches), with a
            1 kHz floor.

    Raises:
        SDRError: If the session manifest is missing or has no steps.
    """

    def __init__(
        self,
        session_dir: Path,
        freq_tolerance_hz: float | None = None,
    ) -> None:
        self._dir = Path(session_dir)
        manifest_path = self._dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise SDRError(f"No recording manifest found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        self._scan_config: dict[str, Any] = manifest.get("scan_config", {})
        self._sdr_config: dict[str, Any] = manifest.get("sdr_config", {})
        self._sample_rate = float(manifest.get("sample_rate", 0)) or float(
            self._sdr_config.get("sample_rate", 2_048_000)
        )

        steps = manifest.get("steps", [])
        if not steps:
            raise SDRError(f"Recording session {self._dir} contains no steps")

        # Sort steps by centre frequency for nearest-match lookup.
        steps = sorted(steps, key=lambda s: s["center_freq_hz"])
        self._centers = np.array([s["center_freq_hz"] for s in steps], dtype=np.float64)
        self._data_files = [s["data_file"] for s in steps]

        if freq_tolerance_hz is not None:
            self._tolerance_hz = float(freq_tolerance_hz)
        else:
            step_size = float(self._scan_config.get("step_size", 0))
            self._tolerance_hz = max(step_size / 2, 1_000.0)

        self._pending_idx: int | None = None
        self._iq_cache: dict[int, np.ndarray] = {}
        self._warned_length = False

    @property
    def sample_rate(self) -> float:
        """Sample rate of the recorded IQ in Hz."""
        return self._sample_rate

    @property
    def connected(self) -> bool:
        """Always True — a file-backed source is always 'connected'."""
        return True

    @property
    def scan_config(self) -> dict[str, Any]:
        """The scan configuration stored in the session manifest."""
        return self._scan_config

    @property
    def sdr_config(self) -> dict[str, Any]:
        """The SDR configuration stored in the session manifest."""
        return self._sdr_config

    def tune(self, frequency_hz: float) -> float:
        """Select the recorded step nearest to ``frequency_hz``.

        Args:
            frequency_hz: Requested centre frequency in Hz.

        Returns:
            The actual recorded centre frequency of the matched step.

        Raises:
            SDRError: If no recorded step lies within the tolerance.
        """
        idx = int(np.argmin(np.abs(self._centers - frequency_hz)))
        nearest = float(self._centers[idx])
        if abs(nearest - frequency_hz) > self._tolerance_hz:
            raise SDRError(
                f"No recorded step near {frequency_hz / 1e6:.3f} MHz "
                f"(nearest {nearest / 1e6:.3f} MHz, tolerance "
                f"{self._tolerance_hz / 1e3:.1f} kHz)"
            )
        self._pending_idx = idx
        return nearest

    def capture(self, num_samples: int) -> np.ndarray:
        """Return the recorded IQ for the last-tuned step.

        Reconciles the recorded length against the requested ``num_samples``:
        an exact match is returned as-is, a longer recording is truncated, and
        a shorter recording is tiled (periodic extension preserves spectral
        content better than zero-padding).

        Raises:
            SDRError: If called before a successful ``tune()``.
        """
        if self._pending_idx is None:
            raise SDRError("capture() called before tune() on ReplaySource")

        iq = self._load(self._pending_idx)

        if len(iq) == num_samples:
            return iq
        if len(iq) > num_samples:
            return iq[:num_samples]

        # Shorter than requested — tile to length.
        if not self._warned_length:
            logger.warning(
                "Recorded step has %d samples but %d requested; tiling. "
                "Replay scan config likely differs from the recording.",
                len(iq), num_samples,
            )
            self._warned_length = True
        reps = int(np.ceil(num_samples / len(iq)))
        return np.tile(iq, reps)[:num_samples].astype(np.complex64, copy=False)

    def _load(self, idx: int) -> np.ndarray:
        """Load (and cache) the IQ for a step index."""
        if idx not in self._iq_cache:
            self._iq_cache[idx] = read_iq(self._dir / self._data_files[idx])
        return self._iq_cache[idx]

    def __enter__(self) -> "ReplaySource":
        """No-op context entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """No-op context exit."""
        return None

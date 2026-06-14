"""IQ recording: capture raw sweeps to a replayable SigMF session.

Two cooperating pieces:

* :class:`SigmfRecorder` writes one SigMF recording per scan step into a session
  directory, plus a ``session.json`` manifest that captures the exact scan and
  SDR configuration used. Storing the config is what lets a later replay
  reconstruct the identical sweep.
* :class:`RecordingSDR` is a transparent decorator around any
  :class:`~src.sdr.base.SdrSource`. It forwards ``tune``/``capture`` to the real
  source and tees the **raw** captured IQ (before any PSD/DC processing) to the
  recorder, so the corpus stays faithful to what the antenna actually saw.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.sdr.base import SdrSource
from src.sdr.sigmf import write_recording

logger = logging.getLogger(__name__)

MANIFEST_NAME = "session.json"
SESSION_VERSION = 1


class SigmfRecorder:
    """Write per-step SigMF recordings and a session manifest.

    Args:
        session_dir: Directory to write recordings and the manifest into.
            Created if it does not exist.
        scan_config: The scan configuration used for this sweep (persisted so
            replay can regenerate identical step frequencies).
        sdr_config: The SDR configuration used (persisted for provenance).
        label: Optional operator label stored on every step and in the manifest.
    """

    def __init__(
        self,
        session_dir: Path,
        scan_config: dict[str, Any],
        sdr_config: dict[str, Any],
        label: str | None = None,
    ) -> None:
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._scan_config = dict(scan_config)
        self._sdr_config = dict(sdr_config)
        self._label = label
        self._driver = sdr_config.get("driver")
        self._steps: list[dict[str, Any]] = []

    def record_step(
        self,
        center_freq_hz: float,
        sample_rate: float,
        iq: np.ndarray,
    ) -> None:
        """Write one scan step's raw IQ as a SigMF recording.

        Args:
            center_freq_hz: Actual tuned centre frequency in Hz.
            sample_rate: Sample rate of the captured IQ in Hz.
            iq: Raw complex IQ samples for this step.
        """
        index = len(self._steps)
        timestamp = datetime.now(timezone.utc).isoformat()
        base_name = f"step_{index:04d}_{center_freq_hz / 1e6:.3f}"
        base_path = self._dir / base_name

        write_recording(
            base_path,
            iq,
            sample_rate=sample_rate,
            center_freq_hz=center_freq_hz,
            iso_timestamp=timestamp,
            label=self._label,
            driver=self._driver,
        )

        self._steps.append({
            "index": index,
            "center_freq_hz": float(center_freq_hz),
            "num_samples": int(len(iq)),
            "datetime": timestamp,
            "data_file": base_name,
        })

        logger.debug(
            "Recorded step %d: %.3f MHz, %d samples -> %s",
            index, center_freq_hz / 1e6, len(iq), base_name,
        )

    def finalize(self) -> None:
        """Write the ``session.json`` manifest for the recorded sweep."""
        manifest = {
            "skadi_session_version": SESSION_VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
            "label": self._label,
            "sample_rate": float(self._sdr_config.get("sample_rate", 0)),
            "scan_config": self._scan_config,
            "sdr_config": self._sdr_config,
            "steps": self._steps,
        }
        manifest_path = self._dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(
            "Recording session finalised: %d step(s) in %s",
            len(self._steps), self._dir,
        )


class RecordingSDR:
    """Transparent recording decorator around a live :class:`SdrSource`.

    Forwards every call to the wrapped source and tees the raw IQ returned by
    ``capture()`` to a :class:`SigmfRecorder`, tagged with the most recent
    actual-tuned frequency. The wrapped source's context manager is entered and
    exited as usual; the recorder is finalised on exit.

    Args:
        inner: The real SDR source to wrap.
        recorder: The recorder that persists each captured step.
    """

    def __init__(self, inner: SdrSource, recorder: SigmfRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self._last_tune_hz = 0.0

    @property
    def sample_rate(self) -> float:
        """Sample rate of the wrapped source in Hz."""
        return self._inner.sample_rate

    @property
    def connected(self) -> bool:
        """Whether the wrapped source is connected."""
        return self._inner.connected

    def tune(self, frequency_hz: float) -> float:
        """Tune the wrapped source, remembering the actual tuned frequency."""
        self._last_tune_hz = self._inner.tune(frequency_hz)
        return self._last_tune_hz

    def capture(self, num_samples: int) -> np.ndarray:
        """Capture from the wrapped source and record the raw IQ."""
        iq = self._inner.capture(num_samples)
        self._recorder.record_step(self._last_tune_hz, self._inner.sample_rate, iq)
        return iq

    def __enter__(self) -> "RecordingSDR":
        """Enter the wrapped source's context."""
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Finalise the recording, then exit the wrapped source's context."""
        try:
            self._recorder.finalize()
        finally:
            self._inner.__exit__(exc_type, exc_val, exc_tb)

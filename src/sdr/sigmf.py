"""Minimal SigMF-compatible IQ recording reader/writer.

Skaði records raw IQ to disk so captures can be replayed deterministically and
collected into a labelled corpus. We use the `SigMF <https://sigmf.org>`_ layout
— one ``.sigmf-data`` file (raw interleaved IQ) paired with a ``.sigmf-meta``
JSON sidecar — but implement only the small slice we need rather than depending
on the ``sigmf`` package. This keeps the project fully offline with no extra
dependencies, while the files remain readable by any standard SigMF tool.

Data is stored as ``cf32_le``: interleaved little-endian 32-bit float I/Q, which
maps directly to NumPy ``complex64``. We force little-endian on write so the
files are portable regardless of host byte order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIGMF_VERSION = "1.0.0"
DATATYPE = "cf32_le"
DATA_EXT = ".sigmf-data"
META_EXT = ".sigmf-meta"

# Little-endian complex64 — the on-disk representation of cf32_le.
_LE_C8 = np.dtype("<c8")


def _base(path: Path) -> Path:
    """Return the recording base path, stripping any SigMF extension.

    Accepts a base path or either member of the pair, so callers may pass
    ``foo``, ``foo.sigmf-data``, or ``foo.sigmf-meta`` interchangeably.
    """
    path = Path(path)
    if path.name.endswith(DATA_EXT):
        return path.with_name(path.name[: -len(DATA_EXT)])
    if path.name.endswith(META_EXT):
        return path.with_name(path.name[: -len(META_EXT)])
    return path


def data_path_for(base_path: Path) -> Path:
    """Return the ``.sigmf-data`` path for a recording base path."""
    base = _base(base_path)
    return base.with_name(base.name + DATA_EXT)


def meta_path_for(base_path: Path) -> Path:
    """Return the ``.sigmf-meta`` path for a recording base path."""
    base = _base(base_path)
    return base.with_name(base.name + META_EXT)


def write_recording(
    base_path: Path,
    iq: np.ndarray,
    *,
    sample_rate: float,
    center_freq_hz: float,
    iso_timestamp: str,
    label: str | None = None,
    driver: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write a single-capture SigMF recording.

    Args:
        base_path: Recording base path (without extension). The ``.sigmf-data``
            and ``.sigmf-meta`` files are written alongside it.
        iq: Complex IQ samples. Cast to little-endian complex64 on write.
        sample_rate: Sample rate in Hz (stored as ``core:sample_rate``).
        center_freq_hz: Actual tuned centre frequency in Hz
            (stored as ``core:frequency`` on the single capture segment).
        iso_timestamp: ISO 8601 UTC capture time (``core:datetime``).
        label: Optional operator label (stored as ``skadi:label``).
        driver: Optional SDR driver name (stored as ``skadi:driver``).
        extra: Optional extra global metadata merged under ``skadi:*`` keys.
    """
    base = _base(base_path)
    data_file = data_path_for(base)
    meta_file = meta_path_for(base)

    data_file.parent.mkdir(parents=True, exist_ok=True)

    # Force little-endian complex64 so the file is portable cf32_le.
    np.asarray(iq).astype(_LE_C8, copy=False).tofile(data_file)

    global_meta: dict = {
        "core:datatype": DATATYPE,
        "core:sample_rate": float(sample_rate),
        "core:version": SIGMF_VERSION,
        "core:num_channels": 1,
    }
    if label is not None:
        global_meta["skadi:label"] = label
    if driver is not None:
        global_meta["skadi:driver"] = driver
    if extra:
        for key, value in extra.items():
            global_meta[f"skadi:{key}"] = value

    meta = {
        "global": global_meta,
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(center_freq_hz),
                "core:datetime": iso_timestamp,
            }
        ],
        "annotations": [],
    }

    meta_file.write_text(json.dumps(meta, indent=2))


def read_meta(base_path: Path) -> dict:
    """Read and parse the ``.sigmf-meta`` JSON for a recording."""
    return json.loads(meta_path_for(base_path).read_text())


def read_iq(base_path: Path) -> np.ndarray:
    """Read IQ samples from a recording's ``.sigmf-data`` as complex64."""
    raw = np.fromfile(data_path_for(base_path), dtype=_LE_C8)
    return raw.astype(np.complex64, copy=False)

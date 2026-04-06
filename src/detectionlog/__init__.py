"""Detection logging for Skaði.

Provides SQLite-backed persistence for signal detection events.
"""

from src.detectionlog.database import DetectionLog

__all__ = ["DetectionLog"]

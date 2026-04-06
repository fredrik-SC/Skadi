# SEIARA Integration Guide

## Overview

Skadi integrates with SEIARA through a lightweight polling model. SEIARA queries Skadi's SQLite detection log database at regular intervals (recommended: every 60 seconds) for new signal detection events.

```
Skadi (continuous scanning) --> detections.db --> SEIARA (polling)
```

## Detection Log Schema

**Database file:** `data/detections.db` (SQLite)
**Table:** `detections`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `timestamp_utc` | TEXT | Detection time in ISO 8601 UTC (e.g. `2026-04-06T17:16:57.931808+00:00`) |
| `frequency_hz` | REAL | Centre frequency in Hz |
| `bandwidth_hz` | REAL | Estimated signal bandwidth in Hz |
| `modulation` | TEXT | Detected modulation type (AM, FM, NFM, FSK, PSK, OOK, UNKNOWN) |
| `signal_strength_dbm` | REAL | Peak signal strength in dBm |
| `signal_type` | TEXT | Best match from Artemis DB (e.g. "STANAG 4285") or NULL |
| `confidence_score` | REAL | Classification confidence (0.0-1.0) or NULL |
| `alt_match_1` | TEXT | Second-best Artemis match or NULL |
| `alt_match_1_confidence` | REAL | Second match confidence or NULL |
| `alt_match_2` | TEXT | Third-best Artemis match or NULL |
| `alt_match_2_confidence` | REAL | Third match confidence or NULL |
| `known_users` | TEXT | Known operators (from Artemis description) or NULL |
| `threat_level` | TEXT | CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL |
| `acf_value` | REAL | Measured autocorrelation period in ms or NULL |
| `notes` | TEXT | Operator notes or NULL |

**Indexes:**
- `idx_detections_timestamp` on `timestamp_utc`
- `idx_detections_frequency` on `frequency_hz`

## Polling Pattern

### Direct SQLite Query (Recommended)

```python
import sqlite3
from datetime import datetime, timezone

db_path = "/path/to/skadi/data/detections.db"
last_poll = "2026-04-06T17:00:00+00:00"

conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()
cursor.execute(
    "SELECT * FROM detections WHERE timestamp_utc > ? ORDER BY timestamp_utc",
    (last_poll,)
)

for row in cursor.fetchall():
    detection = dict(row)
    # Process: correlate with other intelligence, assess risk, display
    print(f"{detection['frequency_hz']/1e6:.3f} MHz - {detection['signal_type']} "
          f"[{detection['threat_level']}]")

# Update last_poll to the newest detection's timestamp
last_poll = datetime.now(timezone.utc).isoformat()
conn.close()
```

### Using the Skadi Python API

```python
from pathlib import Path
from src.detectionlog.database import DetectionLog

log = DetectionLog(Path("data/detections.db"))

# Query with filters
detections = log.query(
    since="2026-04-06T17:00:00+00:00",
    freq_min_hz=3e6,      # HF band only
    freq_max_hz=30e6,
    threat_level="HIGH",   # Only HIGH threat
    limit=100,
)

for det in detections:
    print(det)

log.close()
```

### Using the REST API

While Skadi's web server is running:

```bash
# All detections (latest 200)
curl http://127.0.0.1:8050/api/detections

# Filtered by threat level
curl "http://127.0.0.1:8050/api/detections?threat_level=CRITICAL"

# Filtered by frequency range
curl "http://127.0.0.1:8050/api/detections?freq_min=3000000&freq_max=30000000"

# Filtered by time
curl "http://127.0.0.1:8050/api/detections?since=2026-04-06T17:00:00%2B00:00"
```

### JSON Export

```bash
# Export all detections to file
python -m src.main --export data/export.json

# Programmatic export
python -c "
from pathlib import Path
from src.detectionlog.export import export_json
export_json(Path('data/detections.db'), Path('data/export.json'))
"
```

## JSON Record Format

Each detection record in the JSON export:

```json
{
  "id": 42,
  "timestamp_utc": "2026-04-06T17:16:57.931808+00:00",
  "frequency_hz": 14250000.0,
  "bandwidth_hz": 2750.0,
  "modulation": "PSK",
  "signal_strength_dbm": -85.3,
  "signal_type": "STANAG 4285",
  "confidence_score": 0.95,
  "alt_match_1": "STANAG 4415",
  "alt_match_1_confidence": 0.72,
  "alt_match_2": "CIS-16",
  "alt_match_2_confidence": 0.65,
  "known_users": "NATO military HF data communication standard",
  "threat_level": "HIGH",
  "acf_value": 106.66,
  "notes": null
}
```

## Example SEIARA Polling Script

See `scripts/seiara_poll_example.py` for a complete polling implementation:

```bash
# Poll every 60 seconds
python scripts/seiara_poll_example.py --db data/detections.db --interval 60

# Show all existing detections, then poll
python scripts/seiara_poll_example.py --all --interval 60
```

## Threat Level Mapping

| Level | Colour | Meaning |
|-------|--------|---------|
| CRITICAL | Red | Direct operational threat (military encrypted, hostile forces) |
| HIGH | Orange | Military/government presence indicator |
| MEDIUM | Yellow | Unknown or unclassified signals (default) |
| LOW | Blue | Commercial/civilian (expected background) |
| INFORMATIONAL | Grey | Benign infrastructure (broadcast, navigation) |

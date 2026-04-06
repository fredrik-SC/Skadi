# Skadi Operations Guide

## Installation

### Prerequisites

- Python 3.10+
- SDRPlay RSPduo with API 3.x drivers installed
- SoapySDR with SoapySDRPlay3 plugin

### Install

```bash
git clone https://github.com/fredrik-SC/Skadi.git
cd Skadi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If SoapySDR Python bindings are installed system-wide:
```bash
echo "/path/to/soapysdr/site-packages" > venv/lib/python3.x/site-packages/soapysdr.pth
```

### Download Artemis Database

```bash
python scripts/download_artemis_db.py
```

This fetches the signal database (~90 MB download, 176 KB SQLite) from AresValley.

## Running Skadi

### Quick Start

```bash
# Scan VHF band with web dashboard
python -m src.main --preset vhf

# Open browser to http://127.0.0.1:8050
```

### Scan Presets

| Preset | Band | Frequency | FFT Res | Dwell | Use Case |
|--------|------|-----------|---------|-------|----------|
| `military_hf` | HF | 2-30 MHz | 125 Hz/bin | 2.0s | Military HF signals (STANAG, Russian FSK) |
| `hf` | HF | 3-30 MHz | 125 Hz/bin | 1.0s | General HF monitoring |
| `vhf` | VHF | 30-174 MHz | 250 Hz/bin | 0.5s | VHF tactical, FM broadcast |
| `airband` | Airband | 108-137 MHz | 250 Hz/bin | 0.5s | Aviation AM communications |
| `uhf` | UHF | 225-512 MHz | 250 Hz/bin | 0.5s | Military UHF, government |

```bash
# Scan HF for military signals
python -m src.main --preset military_hf

# Scan airband
python -m src.main --preset airband

# Custom frequency range
python -m src.main --start 144e6 --stop 148e6

# Single sweep, no web server, export to JSON
python -m src.main --preset vhf --single --no-web --export data/export.json
```

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--preset NAME` | Use a scan preset (hf, vhf, uhf, airband, military_hf) |
| `--start HZ` | Override start frequency |
| `--stop HZ` | Override stop frequency |
| `--single` | Run one sweep and exit |
| `--no-web` | Disable web dashboard (CLI only) |
| `--export PATH` | Export detections to JSON after scanning |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR) |

### Web Dashboard

The dashboard runs at `http://127.0.0.1:8050` and shows:

- **Status bar**: Scanner state, sweep count, total detections
- **Live alerts**: Real-time colour-coded detection cards
- **Detection history**: Sortable, filterable table
- **Scan controls**: Start/Stop with frequency range inputs

Threat level colours:
- CRITICAL = Red (pulsing)
- HIGH = Orange
- MEDIUM = Yellow
- LOW = Blue
- INFORMATIONAL = Grey

## Configuration

All configuration is in `config/` as YAML files:

- `config/default.yaml` — Main configuration (SDR, scan, detection, fingerprint, classification, web)
- `config/threat_levels.yaml` — Signal-to-threat-level mapping rules
- `config/exclusions.yaml` — Known/benign signals to exclude from detection

See `docs/CONFIGURATION.md` for the full parameter reference.

## Detection Log

All detections are stored in `data/detections.db` (SQLite). This is the primary integration point for external systems.

- Query via the REST API: `GET http://127.0.0.1:8050/api/detections`
- Query directly with SQLite tools
- Export to JSON: `python -m src.main --export data/export.json`

See `docs/SEIARA_INTEGRATION.md` for the schema and integration guide.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "SoapySDR not found" | Install SoapySDR and create .pth file in venv |
| "No devices found" | Check RSPduo is connected and SDRPlay API is installed |
| "Port 8050 in use" | Change port in config/default.yaml |
| "No signals detected" | Check antenna connection, try a known busy band (FM 88-108 MHz) |
| Web dashboard empty | Click "Refresh" or start a scan first |

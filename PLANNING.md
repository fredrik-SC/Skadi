# Project Skaði -- Implementation Planning Document

**Version:** 1.0
**Date:** 6 April 2026
**Companion document:** PRD.md

---

## 1. Implementation Phases

### Phase 1: Foundation (Sessions 1-3)

**Objective:** Establish project structure, SDR connectivity, and basic spectrum scanning.

#### Session 1: Project Scaffolding and SDR Connection

- Initialise Python project with virtual environment and dependency management
- Install and verify SoapySDR with SDRPlay RSPduo driver
- Create configuration system (YAML-based)
- Write a minimal script that connects to the RSPduo, tunes to a frequency, and captures raw IQ samples
- Verify IQ data is valid by plotting a basic spectrum (using matplotlib or similar)
- Create the project directory structure (see Section 3)

**Acceptance criteria:** Can connect to RSPduo and capture IQ data from a known frequency. Spectrum plot shows recognisable signal features.

#### Session 2: Spectrum Scanning Engine

- Implement frequency sweep across a configurable VHF/UHF range
- Compute FFT-based power spectral density from IQ samples
- Implement configurable scan step size and dwell time per step
- Build basic signal detection using power threshold above noise floor
- Output detected signals as a list: centre frequency, estimated bandwidth, signal strength (dBm)

**Acceptance criteria:** System scans a defined frequency range and identifies active signals with their basic parameters. Test against known local signals (FM broadcast, etc.).

#### Session 3: Signal Detection Refinement

- Implement adaptive noise floor estimation (rolling average or median)
- Add signal clustering (group adjacent bins that exceed threshold into single signal detections)
- Implement signal prioritisation by strength
- Add configurable exclusion list for known/benign signals
- Begin detection logging to SQLite (basic schema)

**Acceptance criteria:** System reliably detects and separates multiple signals within a scan. Known FM stations, airport comms, or similar are consistently detected. Exclusion list filters out specified frequencies.

---

### Phase 2: Fingerprinting and Classification (Sessions 4-6)

#### Session 4: Fingerprint Extraction

- Implement modulation type detection (start with AM/FM/FSK/PSK discrimination)
- Implement bandwidth estimation from spectral analysis
- Implement ACF computation from IQ samples
- Package extracted parameters into a structured fingerprint object

**Acceptance criteria:** System extracts modulation type, bandwidth, and ACF from detected signals. Verify against known signal types (e.g., FM broadcast correctly identified as FM with ~200kHz bandwidth).

**Technical note:** Modulation classification is the most technically challenging component. Start with energy-based and spectral feature approaches before considering ML. Key discriminators:
- AM vs FM: envelope variance vs instantaneous frequency variance
- FSK: discrete frequency states visible in instantaneous frequency
- PSK: phase discontinuities in instantaneous phase
- Digital vs analogue: spectral flatness and symbol rate detection

#### Session 5: Artemis Database Integration

- Download and integrate the Artemis SQLite database (from GitHub releases)
- Write a database access layer that queries signals by parametric criteria
- Implement the matching algorithm:
  1. Filter by modulation type (exact match)
  2. Filter by bandwidth (+/- 15% tolerance)
  3. Filter by frequency range (detected frequency within known range)
  4. Score by ACF match (where available)
- Implement confidence scoring with weighted parameters
- Return top 3 matches with confidence scores

**Acceptance criteria:** Given a manually constructed fingerprint (known parameters), the classifier returns the correct Artemis signal as the top match. Test with at least 5 known signal types.

#### Session 6: End-to-End Pipeline

- Connect scanning, detection, fingerprinting, and classification into a continuous pipeline
- Implement the threat assessment lookup table (YAML-based)
- Ensure detection log captures full schema (all v1.0 fields)
- Add JSON export capability for the detection log
- Test the complete pipeline against real signals

**Acceptance criteria:** System continuously scans, detects signals, classifies them, assigns threat levels, and logs everything. The operator can observe the process and review the log.

---

### Phase 3: User Interface (Sessions 7-8)

#### Session 7: Web Server and Real-Time Alerts

- Implement a lightweight local web server (Flask or FastAPI)
- Create WebSocket connection for real-time alert push to browser
- Build the alert display: colour-coded cards showing signal type, threat level, confidence, frequency, strength
- Implement detection history table with sorting and filtering
- Threat level colour coding: CRITICAL=red, HIGH=orange, MEDIUM=yellow, LOW=blue, INFORMATIONAL=grey

**Acceptance criteria:** Browser displays real-time alerts as signals are detected and classified. Detection history is browsable and filterable.

#### Session 8: Configuration UI and Polish

- Add operator controls in the GUI: start/stop scanning, adjust frequency range, set threshold
- Add exclusion list management through the GUI
- Add threat lookup table editor in the GUI (or confirm YAML editing workflow is sufficient)
- Error handling and graceful degradation (SDR disconnected, DB not found, etc.)
- Logging and diagnostics

**Acceptance criteria:** Operator can configure and control the system entirely through the browser GUI. System handles error conditions without crashing.

---

### Phase 4: Hardening and Integration Prep (Sessions 9-10)

#### Session 9: Testing and Reliability

- Write unit tests for each module (detection, fingerprinting, classification, threat assessment)
- Integration tests for the full pipeline
- Stress testing with continuous operation (run for 1+ hour)
- Edge case handling: no signals detected, all signals unknown, database errors
- Performance profiling: measure scan-to-classification latency

**Acceptance criteria:** All tests pass. System runs stably for extended periods. Scan-to-classification latency is under 5 seconds.

#### Session 10: Documentation and SEIARA Integration Interface

- Document the detection log schema and JSON export format for SEIARA integration
- Create a simple polling example (Python script that queries the detection log for new entries since last poll)
- Document configuration options and operational procedures
- Package the application for distribution (pip installable or Docker container for Linux)
- Test on Linux (if Raspberry Pi available, test there too)

**Acceptance criteria:** A clean install on a fresh Linux system works. The SEIARA polling example successfully retrieves new detections. Documentation is complete.

---

## 2. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.10+ | Developer familiarity, rich DSP ecosystem |
| SDR Interface | SoapySDR (via Python bindings) | Cross-platform, clean abstraction over SDRPlay API |
| DSP/Signal Processing | NumPy, SciPy | Industry standard for FFT, filtering, spectral analysis |
| Signal Database | Artemis DB (SQLite) | Offline, 500+ signals, community maintained |
| Detection Log | SQLite (via Python sqlite3) | Lightweight, no server needed, easy to query externally |
| Web Framework | Flask or FastAPI | Lightweight, sufficient for local-only GUI |
| Real-time Comms | WebSocket (flask-socketio or similar) | Push-based alerts to browser |
| Frontend | HTML/CSS/JS (minimal framework) | Keep simple; no build toolchain needed |
| Configuration | YAML (via PyYAML) | Human-readable, easy to edit |
| Plotting/Debug | Matplotlib | Development and debugging of spectral data |
| Testing | pytest | Standard Python testing |

### 2.1 Key Python Libraries

```
soapysdr          # SDR hardware abstraction
numpy             # Array operations, FFT
scipy             # Signal processing, filtering, ACF
flask / fastapi   # Web server
flask-socketio    # WebSocket support (if Flask)
pyyaml            # Configuration files
matplotlib        # Development/debug plotting
pytest            # Testing
```

---

## 3. Project Directory Structure

```
skadi/
├── CLAUDE.md                    # Claude Code project instructions
├── PRD.md                       # Product requirements document
├── PLANNING.md                  # This document
├── config/
│   ├── default.yaml             # Default configuration
│   ├── threat_levels.yaml       # Signal type to threat level mapping
│   └── exclusions.yaml          # Known/benign signal exclusion list
├── data/
│   ├── artemis.db               # Artemis signal database (SQLite)
│   └── detections.db            # Detection log database (SQLite)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Application entry point
│   ├── sdr/
│   │   ├── __init__.py
│   │   ├── interface.py         # SoapySDR connection and IQ capture
│   │   └── scanner.py           # Frequency sweep and spectrum scanning
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py          # Signal detection from spectral data
│   │   └── noise.py             # Noise floor estimation
│   ├── fingerprint/
│   │   ├── __init__.py
│   │   ├── extractor.py         # Parameter extraction from detected signals
│   │   └── modulation.py        # Modulation type classification
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classifier.py        # Artemis DB matching engine
│   │   ├── confidence.py        # Confidence scoring
│   │   └── threat.py            # Threat level assignment
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLite detection log operations
│   │   └── export.py            # JSON export
│   └── web/
│       ├── __init__.py
│       ├── server.py            # Flask/FastAPI web server
│       ├── websocket.py         # WebSocket handler for real-time alerts
│       ├── static/
│       │   ├── css/
│       │   └── js/
│       └── templates/
│           └── index.html
├── tests/
│   ├── test_detector.py
│   ├── test_fingerprint.py
│   ├── test_classifier.py
│   ├── test_threat.py
│   └── test_pipeline.py
├── scripts/
│   ├── seiara_poll_example.py   # Example SEIARA integration polling script
│   └── download_artemis_db.py   # Script to download latest Artemis DB
├── requirements.txt
└── setup.py
```

---

## 4. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Modulation classification accuracy is too low for reliable matching | High | Medium | Start with simple energy/spectral approaches. Fall back to frequency + bandwidth matching only. Log UNKNOWN signals for manual review. Iterate. |
| SoapySDR Python bindings are unstable or incomplete for RSPduo | Medium | Low | Native SDRPlay API is available as fallback. GNU Radio with SoapySDR source block is another option. |
| Artemis DB schema changes between versions | Medium | Low | Pin to a specific Artemis DB release. Write a versioned DB access layer. |
| Real-time performance insufficient on target hardware | Medium | Medium | Profile early (Session 2). Reduce scan range or increase dwell time. For Raspberry Pi, this is a known risk requiring dedicated evaluation. |
| Signal detection threshold tuning is environment-dependent | Medium | High | Implement adaptive noise floor. Make threshold configurable. Accept that initial sessions will involve experimentation. |
| Multiple overlapping signals confuse fingerprinting | Medium | Medium | Prioritise strongest signals first. Implement spectral separation. Accept degraded performance in dense RF environments for v1.0. |

---

## 5. Dependencies and Prerequisites

Before Session 1, the following must be in place:

1. **SDRPlay RSPduo** connected and drivers installed (SDRPlay API 3.x)
2. **SoapySDR** installed with the SDRPlay plugin (SoapySDRPlay3)
3. **Python 3.10+** with virtual environment support
4. **Artemis database** downloaded from https://github.com/AresValley/Artemis-DB/releases (latest release)
5. **Development machine** running macOS (primary) or Linux

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 6 Apr 2026 | Use SoapySDR over native SDRPlay API | Cross-platform compatibility, cleaner Python bindings, sufficient for single-tuner mode |
| 6 Apr 2026 | Start with single-tuner mode | Dual-tuner limits bandwidth to 2MHz per tuner. Single tuner gives 10MHz. Dual-tuner useful later for DF. |
| 6 Apr 2026 | Parametric matching over ML/pattern matching for v1.0 | Simpler, more deterministic, easier to debug. Artemis DB has structured parametric data. ML can be added later. |
| 6 Apr 2026 | VHF/UHF test band | Relevant to target military use case while still having testable signals at the bench. |
| 6 Apr 2026 | Browser-based GUI over desktop GUI | Aligns with eventual SEIARA integration (React frontend). Platform independent. |
| 6 Apr 2026 | SQLite for detection log | Lightweight, queryable, no server dependency. JSON export for flexibility. Natural fit for SEIARA polling. |
| 6 Apr 2026 | Simple threat lookup table over complex risk model | SEIARA already has a correlation engine and risk allocation model. Skaði only needs basic categorisation. |

---

## 7. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 6 April 2026 | Fredrik | Initial draft |

# Project Skaði -- Claude Code Instructions

## Project Overview

Skaði is a standalone, offline RF signal identification tool. It scans the radio frequency spectrum using an SDRPlay RSPduo receiver, detects active signals, extracts a parametric fingerprint, and classifies them against the Artemis signal database. The system provides real-time threat-level alerting through a browser-based GUI.

This is a proof-of-concept/bench test that must be architected for future integration with SEIARA (an intelligence platform with React frontend, Python/Docker backend, Neo4j, Qdrant, and AI capabilities). Integration will be via SEIARA polling Skaði's SQLite detection log.

## Project Status: v1.0 Milestone (7 April 2026)

All 10 planned development sessions are complete. The system is operational as a bench test. See docs/OPERATIONS.md for usage and docs/SEIARA_INTEGRATION.md for the polling interface.

**Known v1.0 limitations:**
- SoapySDRPlay3 C++ plugin crashes on shutdown (cosmetic — data is saved)
- Classification accuracy ~60% for real-world narrowband signals
- Python 3.14 required (SoapySDR homebrew bindings not available for 3.13)
- Linux deployment untested (Dockerfile exists but not verified)

## Key Documents

- **PRD.md** -- Full product requirements with numbered requirements (FR-*, NFR-*)
- **PLANNING.md** -- Implementation phases, tech stack, directory structure, risk register
- **docs/OPERATIONS.md** -- How to install and run Skaði
- **docs/SEIARA_INTEGRATION.md** -- Detection log schema, polling pattern, REST API
- **docs/CONFIGURATION.md** -- All tuneable parameters

Always consult these documents before starting work. Requirements are the source of truth.

## Development Context

- **Developer:** Fredrik is not a professional developer. He uses Specification Driven Development. Code must be clean, well-commented, and maintainable.
- **Platform:** Developed on macOS, must also run on Linux (including Raspberry Pi in future).
- **Offline:** The system operates without internet connectivity. All dependencies must be available locally.

## Architecture Rules

1. **Modular design.** Each component (SDR interface, detection, fingerprinting, classification, logging, web UI) is an independent module with clean interfaces. See PLANNING.md Section 3 for directory structure.
2. **Configuration is external.** All tuneable parameters go in YAML config files under `config/`. Never hardcode frequencies, thresholds, threat levels, or scan parameters.
3. **SQLite for persistence.** Both the Artemis reference database and the detection log use SQLite. The detection log schema is defined in PRD.md Section 3.6.1.
4. **SoapySDR for hardware.** Use SoapySDR Python bindings to interface with the RSPduo. Single-tuner mode only for v1.0.
5. **No internet dependency.** The system must function completely offline. No API calls, no cloud services, no CDN resources in the web UI.

## Coding Standards

- Python 3.10+ with type hints
- Follow PEP 8
- Docstrings on all public functions and classes (Google style)
- Use logging module (not print statements) for all diagnostic output
- Error handling: catch specific exceptions, log them, never silently swallow errors
- Use pathlib for file paths (cross-platform)

## Testing

- pytest for all tests
- Each module should have corresponding test files in `tests/`
- For SDR-dependent code, provide mock/stub interfaces so tests can run without hardware connected
- Test classification against known signal parameters from the Artemis DB

## Signal Processing Notes

- FFT-based power spectral density for spectrum analysis (NumPy/SciPy)
- Adaptive noise floor estimation for signal detection
- Modulation classification approach for v1.0: energy-based and spectral feature extraction (not ML)
  - AM vs FM: envelope variance vs instantaneous frequency variance
  - FSK: discrete frequency states in instantaneous frequency
  - PSK: phase discontinuities
- Bandwidth estimation from spectral analysis
- ACF computation from IQ samples using SciPy correlate

## Artemis Database

- SQLite database from https://github.com/AresValley/Artemis-DB/releases
- Contains 500+ signal types with: frequencies, frequency range, mode, modulation, ACF, emission designator, bandwidth, location, description, audio samples (OGG), waterfall images (PNG)
- Matching strategy (in order of discriminating value):
  1. Modulation type (exact match)
  2. Bandwidth (+/- 15% tolerance)
  3. Frequency range (is detected frequency within known range)
  4. ACF value (where available)
- Return top 3 matches with confidence scores

## Threat Levels

Defined in `config/threat_levels.yaml`. Five levels: CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL. See PRD.md Section 6 for the default mapping. UNKNOWN signals default to MEDIUM.

## Web GUI

- Browser-based, served locally (Flask or FastAPI)
- WebSocket for real-time alert push
- All assets served locally (offline requirement)
- Colour coding: CRITICAL=red, HIGH=orange, MEDIUM=yellow, LOW=blue, INFORMATIONAL=grey
- No external CSS/JS frameworks that require CDN. Bundle everything locally.

## What NOT To Do

- Do not attempt dual-tuner mode in v1.0
- Do not implement ML-based classification in v1.0
- Do not add GPS/location features in v1.0
- Do not add direction finding in v1.0
- Do not add map display in v1.0
- Do not build SEIARA integration beyond ensuring the log DB schema is correct and queryable
- Do not over-engineer the threat model; SEIARA handles sophisticated risk assessment

## Session Workflow

When starting a new development session, always:
1. Read this file, PRD.md, and PLANNING.md
2. Check which phase/session we are working on
3. Review acceptance criteria for the current session
4. Implement, test, and verify against acceptance criteria before moving on

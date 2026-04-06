# Project Skaði -- Product Requirements Document

**Version:** 1.0
**Date:** 6 April 2026
**Author:** Fredrik (SiniusCube Consulting)
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose

Project Skaði is a standalone, offline RF signal identification tool that scans the radio frequency spectrum in real time, detects active signals, extracts measurable characteristics (a "fingerprint"), and classifies them against the Artemis signal database. The tool provides operators with immediate signal identification and threat-level alerting, replacing a manual process that currently takes approximately two weeks.

### 1.2 Background and Motivation

This project originates from a March 2026 meeting with Nick Merrick. His team regularly deploys "Signal Soakers" -- devices that capture all RF signals in a local area. Captured data is then uploaded and sent for external analysis, with results typically returning after two weeks. This delay has direct operational consequences. In one cited example, a European mission identified a Spetsnaz team operating less than 2km away, but this information only became available two weeks after the data was collected.

The core value proposition: if the system can identify the *type* of signal in real time (e.g., "this digital encryption mode is used by Russian military intelligence"), operators gain actionable awareness without needing to decode or decrypt the signal content.

### 1.3 Naming

The project is named Skaði (Old Norse: Skaði), after the Norse goddess associated with hunting and winter. The name is a homage to the Artemis signal database (Artemis being the Greek goddess of hunting) and to Fredrik's Scandinavian heritage.

### 1.4 Strategic Context

Skaði is designed as a standalone tool, independent of SEIARA (SiniusCube's intelligence platform). However, the architecture must support future integration as a modular component. The integration model is lightweight: SEIARA polls Skaði's detection log database at regular intervals (approximately every 60 seconds) for new signal identification events. Skaði runs continuously; SEIARA consumes its output as a data pipeline.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SKAÐI SYSTEM                         │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ SDR      │───>│ Signal       │───>│ Fingerprint  │  │
│  │ Receiver │    │ Detection    │    │ Extraction   │  │
│  │ (SoapySDR)   │ Engine       │    │ Engine       │  │
│  └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                             │          │
│                                             v          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Alert    │<───│ Threat       │<───│ Classifier   │  │
│  │ & GUI    │    │ Assessment   │    │ (Artemis DB) │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│        │                                    │          │
│        v                                    v          │
│  ┌──────────────────────────────────────────────┐      │
│  │           Detection Log (SQLite)             │      │
│  └──────────────────────────────────────────────┘      │
│                        │                               │
└────────────────────────┼───────────────────────────────┘
                         │
                         v (polling)
                ┌─────────────────┐
                │  SEIARA (future)│
                └─────────────────┘
```

### 2.2 Core Components

1. **SDR Interface Layer** -- Manages the SDRPlay RSPduo via SoapySDR. Handles tuning, sample rate configuration, and IQ data streaming. Initial version uses a single tuner with up to 10MHz bandwidth.

2. **Signal Detection Engine** -- Analyses the RF spectrum to identify active signals above a configurable noise threshold. Must distinguish signals from background noise and track multiple concurrent signals. Prioritises detected signals by signal strength (dBm).

3. **Fingerprint Extraction Engine** -- Extracts measurable parameters from each detected signal: centre frequency, bandwidth, modulation type, baud rate (where applicable), and ACF (autocorrelation function). These parameters form the signal's fingerprint.

4. **Classifier** -- Queries the Artemis SQLite database using extracted fingerprint parameters. Returns up to 3 ranked matches with confidence scores.

5. **Threat Assessment** -- Maps classified signal types to threat levels using a configurable lookup table. Initial categories: CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL.

6. **Detection Log** -- SQLite database recording all detection events with full metadata. Serves as the integration point for SEIARA.

7. **Web GUI** -- Browser-based local interface showing real-time alerts, detection history, and (future) map visualisation with transmitter location estimates.

---

## 3. Functional Requirements

### 3.1 RF Spectrum Scanning

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-SCAN-01 | The system shall connect to an SDRPlay RSPduo via SoapySDR | Must |
| FR-SCAN-02 | The system shall support continuous scanning across a configurable frequency range | Must |
| FR-SCAN-03 | Initial test range shall be VHF/UHF (30 MHz -- 3 GHz) | Must |
| FR-SCAN-04 | The system shall support configurable scan step size and dwell time | Must |
| FR-SCAN-05 | The system shall operate in single-tuner mode (10MHz bandwidth) for v1.0 | Must |
| FR-SCAN-06 | The system shall support dual-tuner operation in future versions | Should |
| FR-SCAN-07 | The system shall allow the operator to define sub-bands of interest for focused scanning | Should |

### 3.2 Signal Detection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-DET-01 | The system shall detect active signals above a configurable noise threshold | Must |
| FR-DET-02 | The threshold shall be adjustable by the operator | Must |
| FR-DET-03 | The system shall handle multiple simultaneous signals within the observed bandwidth | Must |
| FR-DET-04 | Detected signals shall be prioritised by signal strength (dBm) | Must |
| FR-DET-05 | The system shall maintain an exclusion list of known/benign signals to skip | Should |
| FR-DET-06 | The exclusion list shall be operator-configurable | Should |

### 3.3 Fingerprint Extraction

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-FP-01 | The system shall extract centre frequency from each detected signal | Must |
| FR-FP-02 | The system shall estimate signal bandwidth | Must |
| FR-FP-03 | The system shall identify modulation type (AM, FM, FSK, PSK, etc.) | Must |
| FR-FP-04 | The system shall estimate baud rate for digital signals where detectable | Should |
| FR-FP-05 | The system shall compute ACF (autocorrelation function) values | Should |
| FR-FP-06 | The system shall capture a short IQ sample for each detected signal for logging | Should |

### 3.4 Signal Classification

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-CLASS-01 | The system shall query the Artemis database using extracted fingerprint parameters | Must |
| FR-CLASS-02 | The system shall return up to 3 ranked matches per detected signal | Must |
| FR-CLASS-03 | Each match shall include a confidence score | Must |
| FR-CLASS-04 | Matching shall use parametric comparison: frequency range, bandwidth, modulation type, ACF | Must |
| FR-CLASS-05 | The system shall handle partial matches (e.g., modulation matches but bandwidth is slightly off) | Should |
| FR-CLASS-06 | Unmatched signals shall be flagged as UNKNOWN and logged | Must |

### 3.5 Threat Assessment

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-THREAT-01 | The system shall assign a threat level to each classified signal | Must |
| FR-THREAT-02 | Threat levels: CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL | Must |
| FR-THREAT-03 | Threat mapping shall be defined in a configurable lookup table | Must |
| FR-THREAT-04 | The lookup table shall map signal type to threat level | Must |
| FR-THREAT-05 | UNKNOWN signals shall default to MEDIUM threat level | Should |

### 3.6 Detection Logging

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-LOG-01 | All detection events shall be recorded in a SQLite database | Must |
| FR-LOG-02 | The log shall support JSON export | Must |
| FR-LOG-03 | Each log entry shall contain the fields defined in section 3.6.1 | Must |
| FR-LOG-04 | The log database shall be queryable by external tools (SEIARA integration) | Must |
| FR-LOG-05 | Logs shall include timestamps in UTC | Must |

#### 3.6.1 Detection Log Schema

| Field | Type | Description | Version |
|-------|------|-------------|---------|
| id | INTEGER | Auto-incrementing primary key | v1.0 |
| timestamp_utc | TEXT (ISO 8601) | Detection time in UTC | v1.0 |
| frequency_hz | REAL | Centre frequency in Hz | v1.0 |
| bandwidth_hz | REAL | Estimated signal bandwidth in Hz | v1.0 |
| modulation | TEXT | Detected modulation type | v1.0 |
| signal_strength_dbm | REAL | Signal strength in dBm | v1.0 |
| signal_type | TEXT | Best match from Artemis DB | v1.0 |
| confidence_score | REAL | Match confidence (0.0 -- 1.0) | v1.0 |
| alt_match_1 | TEXT | Second-best match | v1.0 |
| alt_match_1_confidence | REAL | Second match confidence | v1.0 |
| alt_match_2 | TEXT | Third-best match | v1.0 |
| alt_match_2_confidence | REAL | Third match confidence | v1.0 |
| known_users | TEXT | Known operators of this signal type (from Artemis) | v1.0 |
| threat_level | TEXT | Assigned threat level | v1.0 |
| acf_value | REAL | Measured autocorrelation function value | v1.0 |
| notes | TEXT | Operator notes (optional) | v1.0 |
| bearing_deg | REAL | Signal bearing in degrees (future, DF capability) | v2.0 |
| latitude | REAL | Receiver latitude (future, GPS integration) | v2.0 |
| longitude | REAL | Receiver longitude (future, GPS integration) | v2.0 |
| mgrs | TEXT | MGRS grid reference (future, GPS integration) | v2.0 |
| iq_sample_path | TEXT | Path to stored IQ sample file | v2.0 |

### 3.7 User Interface

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-UI-01 | The system shall provide a browser-based local GUI | Must |
| FR-UI-02 | The GUI shall display real-time alert notifications for detected and classified signals | Must |
| FR-UI-03 | Alerts shall be colour-coded by threat level | Must |
| FR-UI-04 | The GUI shall display a sortable, filterable detection history table | Must |
| FR-UI-05 | The GUI shall allow the operator to configure scan parameters | Should |
| FR-UI-06 | The GUI shall display a local area map with estimated transmitter locations (hexagonal markers) | Could (v2.0) |
| FR-UI-07 | The GUI shall provide a waterfall display of the current scan bandwidth | Could (v2.0) |

---

## 4. Non-Functional Requirements

### 4.1 Platform and Portability

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-PLAT-01 | The system shall run on macOS (development platform) | Must |
| NFR-PLAT-02 | The system shall run on Linux (deployment target, including Raspberry Pi) | Must |
| NFR-PLAT-03 | The system shall be written in Python | Must |
| NFR-PLAT-04 | The system shall operate fully offline (no internet connectivity required) | Must |
| NFR-PLAT-05 | All dependencies shall be installable from vendored packages or standard repositories | Should |

### 4.2 Performance

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-PERF-01 | Signal detection to classification shall complete within 5 seconds of signal acquisition | Should |
| NFR-PERF-02 | The system shall handle at least 10 concurrent signal detections per scan cycle | Should |
| NFR-PERF-03 | The web GUI shall update within 2 seconds of a new detection event | Should |

### 4.3 Modularity and Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-MOD-01 | The system shall be architected as independent, loosely-coupled modules | Must |
| NFR-MOD-02 | The detection log database shall serve as the primary integration interface | Must |
| NFR-MOD-03 | Each module shall be independently testable | Should |
| NFR-MOD-04 | Configuration shall be externalised (YAML or JSON config files) | Must |

---

## 5. The Artemis Database

### 5.1 Overview

The Artemis database is a community-driven SQLite database containing over 500 recognised signal types, sourced from the SigIDWiki project. It is maintained by AresValley and distributed via GitHub (https://github.com/AresValley/Artemis-DB).

### 5.2 Signal Record Structure

Each signal record in the Artemis DB contains (based on SigIDWiki entries):

- **Signal name** -- e.g., "Russian military 20bd 7kHz FSK"
- **Frequencies** -- Known transmission frequencies
- **Frequency range** -- Min/max frequency bounds
- **Mode** -- Demodulation mode (USB, LSB, AM, FM, etc.)
- **Modulation** -- Signal modulation type (FSK, PSK, OFDM, etc.)
- **ACF** -- Autocorrelation function value (where known)
- **Emission designator** -- ITU emission classification
- **Bandwidth** -- Signal bandwidth
- **Location** -- Known countries/regions of origin
- **Short description** -- Human-readable description including known users
- **Audio sample** -- OGG format, max 60 seconds
- **Waterfall image** -- PNG format, spectral visualisation

### 5.3 Matching Strategy

The classifier shall use parametric matching against the following Artemis fields, in order of discriminating value:

1. **Modulation type** -- strongest discriminator; narrows the candidate set significantly
2. **Bandwidth** -- further narrows matches; allow +/- 15% tolerance for measurement variance
3. **Frequency range** -- check if detected frequency falls within known ranges
4. **ACF value** -- where available, provides high-confidence confirmation

Confidence scoring shall weight these parameters, with modulation and bandwidth carrying the highest weight.

---

## 6. Threat Classification Model

### 6.1 Lookup Table Structure

The threat lookup table maps Artemis signal categories to threat levels. This is a simplified model for v1.0; SEIARA's correlation engine and risk allocation model will handle sophisticated threat assessment when integrated.

| Signal Category | Threat Level | Rationale |
|----------------|-------------|-----------|
| Military digital encryption (known hostile) | CRITICAL | Direct operational threat |
| Military communications (unencrypted) | HIGH | Military presence indicator |
| Government/diplomatic signals | HIGH | Potential intelligence value |
| Unknown digital signals | MEDIUM | Unclassified, requires attention |
| Commercial/civilian digital | LOW | Expected background signals |
| Broadcast (AM/FM/DAB) | INFORMATIONAL | Benign, environmental |
| Navigation/timing (GPS, NDB) | INFORMATIONAL | Infrastructure signals |
| Weather/VOLMET | INFORMATIONAL | Benign utility signals |

### 6.2 Customisation

The threat lookup table shall be stored as an editable YAML or JSON file, allowing operators to adjust threat levels based on operational context (e.g., a civilian signal that is unexpected in a given location might warrant a higher threat level).

---

## 7. Future Capabilities (Out of Scope for v1.0)

These are documented for architectural consideration but are not part of the initial build:

1. **Direction finding (DF)** -- Using two SDR receivers and two antennas to triangulate signal bearing. The RSPduo's dual-tuner mode or a second SDR device could support this.
2. **GPS integration** -- USB GPS receiver for automatic position logging in lat/lon and MGRS format.
3. **Map display** -- Local area map in the GUI showing estimated transmitter locations with hexagonal markers, calculated from bearing and signal strength.
4. **Raspberry Pi deployment** -- Compact, portable package with touchscreen display for field use. Compute resource requirements for real-time IQ analysis on Pi hardware need evaluation.
5. **Waterfall display** -- Real-time spectral waterfall in the browser GUI, with zoom capability.
6. **Dual-tuner operation** -- One tuner scanning/classifying, one tuner providing waterfall view of a 10MHz slice (noting the 2MHz-per-tuner limitation in dual mode).
7. **SEIARA integration** -- Polling-based data pipeline from Skaði's detection log.
8. **ML-based classification** -- Neural network or ML model trained on Artemis audio samples and waterfall patterns for improved matching beyond parametric comparison.

---

## 8. Development Methodology

This project follows Specification Driven Development. This PRD and the accompanying planning document serve as the primary input for development sessions using Claude Code in terminal. The developer (Fredrik) is not a professional software developer; code must be clean, well-documented, and maintainable.

---

## 9. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 6 April 2026 | Fredrik | Initial draft |

# Skadi Configuration Reference

All configuration files are in the `config/` directory.

## config/default.yaml

### SDR Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sdr.driver` | `sdrplay` | SoapySDR driver name |
| `sdr.mode` | `ST` | Device mode: ST (Single Tuner) for v1.0 |
| `sdr.sample_rate` | `2048000` | Sample rate in Hz |
| `sdr.bandwidth` | `0` | IF bandwidth (0 = auto) |
| `sdr.gain_reduction` | `0` | Manual gain reduction in dB (0 = use AGC) |
| `sdr.agc` | `true` | Enable automatic gain control |

### Scan Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scan.freq_start` | `30000000` | Start frequency in Hz (30 MHz) |
| `scan.freq_stop` | `174000000` | Stop frequency in Hz (174 MHz) |
| `scan.step_size` | `2000000` | Frequency step size in Hz |
| `scan.dwell_time` | `0.5` | Seconds to capture per step |
| `scan.fft_size` | `8192` | FFT points (determines Hz/bin resolution) |
| `scan.fft_averages` | `10` | Number of FFT frames to average |

**FFT resolution:** `sample_rate / fft_size`. At 2.048 MHz sample rate:
- 4096 points = 500 Hz/bin
- 8192 points = 250 Hz/bin
- 16384 points = 125 Hz/bin (required for HF narrowband)

### Scan Presets

Override the scan section with band-specific parameters:

| Preset | freq_start | freq_stop | step_size | fft_size | dwell_time |
|--------|-----------|----------|-----------|---------|------------|
| `hf` | 3 MHz | 30 MHz | 500 kHz | 16384 | 1.0s |
| `military_hf` | 2 MHz | 30 MHz | 500 kHz | 16384 | 2.0s |
| `vhf` | 30 MHz | 174 MHz | 2 MHz | 8192 | 0.5s |
| `uhf` | 225 MHz | 512 MHz | 2 MHz | 8192 | 0.5s |
| `airband` | 108 MHz | 137 MHz | 2 MHz | 8192 | 0.5s |

### Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection.threshold_db` | `10.0` | Signal must be this many dB above noise floor |
| `detection.min_bandwidth_hz` | `100` | Minimum bandwidth to count as detection |
| `detection.max_signals_per_step` | `20` | Maximum concurrent signals per step |
| `detection.noise_window_size` | `10` | Rolling noise floor window (scan steps) |
| `detection.noise_alpha` | `0.3` | Noise EMA weight (0=all history, 1=current only) |
| `detection.edge_guard_fraction` | `0.0` | Fraction of outer PSD bins ignored for detection (filter rolloff at band edges). `0.0` = off. Pair with `step_size ≈ 0.8 × sample_rate`. |

### Capture-Quality Settings

These improve the fidelity of live captures. All default to a no-op so behaviour
is unchanged until enabled. They affect processing only — recorded IQ is always
stored raw, so recordings can be reprocessed as these settings change.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture.flush_samples` | `0` | Samples read and discarded after each retune before the real capture (clears settling/stale samples). `0` = off. |
| `capture.retune_settle_time` | `0.01` | Settling delay after retuning, in seconds. |
| `capture.dc_removal` | `false` | Subtract the complex mean per FFT segment and blank the centre PSD bin(s) to kill the DC spike (otherwise a false detection appears at every step centre). |
| `capture.dc_blank_bins` | `1` | Central bins each side of DC to blank when `dc_removal` is on. |

### Fingerprint Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprint.min_snr_db` | `8.0` | Minimum SNR for modulation classification |
| `fingerprint.guard_factor` | `3.0` | Filter bandwidth multiplier for isolation |
| `fingerprint.filter_numtaps` | `101` | Minimum FIR filter taps (auto-scaled for narrowband) |
| `fingerprint.min_filter_bw_hz` | `500` | Floor for the isolation filter bandwidth (Hz) |
| `fingerprint.acf_min_lag_ms` | `1.0` | ACF search range minimum (ms) |
| `fingerprint.acf_max_lag_ms` | `5000.0` | ACF search range maximum (ms) |
| `fingerprint.acf_min_peak_strength` | `0.3` | Minimum ACF peak to report |

### Modulation Classifier Thresholds (`fingerprint.modulation.*`)

Feature-led decision thresholds for the deterministic classifier. Omit the block to
use the in-code defaults; override to tune against the benchmark corpus.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env_cv_threshold` | `0.25` | Envelope CV above this → amplitude-modulated (AM/OOK) |
| `ook_bimodality` | `0.80` | Envelope bimodality above this → OOK vs AM |
| `freq_state_fit` | `0.85` | R² of a 2-level instantaneous-frequency model (FSK) |
| `freq_state_flat` | `0.50` | Plateau (held-state) fraction for FSK |
| `freq_state_min_share` | `0.12` | Minimum sample share in each frequency state |
| `freq_state_gap_frac` | `0.30` | State gap as a fraction of inst-freq spread |
| `medfilt_window` | `11` | Median-filter window (odd) for inst-freq smoothing |
| `psk_concentration` | `0.60` | Order-2 phase moment above this → PSK |
| `psk_order1_max` | `0.50` | Order-1 phase moment must be below this for PSK |
| `wideband_fm_min_bw_hz` | `50000` | Bandwidth above this qualifies as wideband FM |
| `fm_inst_freq_var_min` | `0.0005` | Minimum inst-freq variance for FM/NFM |

**Analog-AM hybrid gate** (applied only when an ML model is loaded — see below). Routes
the unambiguous analog-AM-voice signature back to AM before deferring to the model,
which has too little real AM to match the deterministic envelope cue. Tuned on real
airband voice (caught 15/16 AM with 1 digital false-positive in the A/B).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `am_gate_env_cv_min` | `0.45` | Lower envelope-CV bound (strong speech variation) |
| `am_gate_env_cv_max` | `0.65` | Upper envelope-CV bound (excludes OOK/noisy-digital) |
| `am_gate_order2_max` | `0.13` | Max order-2 phase concentration (excludes PSK) |
| `am_gate_order1_max` | `0.12` | Max order-1 phase concentration (excludes carrier-PSK) |
| `am_gate_if_var_min` | `0.05` | Min residual inst-freq variance (excludes FM ≈ 0) |

### ML Classifier (`fingerprint.ml.*`) — v2.0

Optional trained RandomForest modulation classifier. **Off by default** — the
deterministic classifier is used. When enabled and a model is loaded, the classifier
runs hybrid: the analog-AM gate above keeps AM voice on the deterministic path; all
other signals are decided by the model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprint.ml.enabled` | `false` | Enable the trained model (falls back to deterministic if the file is missing/invalid) |
| `fingerprint.ml.model_path` | `data/modulation_model.joblib` | Path to the locally-built model bundle |

Build the model locally (it is gitignored, not shipped):

```
python -m src.ml.train --build --sessions-dir sessions --out data/modulation_model.joblib
```

### Classification Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classification.bandwidth_tolerance` | `0.15` | Bandwidth matching tolerance (15%) |
| `classification.max_matches` | `3` | Return top N Artemis matches |
| `classification.min_confidence` | `0.1` | Minimum confidence to include |

### Web Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `web.host` | `127.0.0.1` | Web server bind address |
| `web.port` | `8050` | Web server port |

## config/threat_levels.yaml

Keyword-based threat assignment. Rules are evaluated in order; first match wins.

```yaml
default_threat_level: MEDIUM

rules:
  - threat_level: CRITICAL
    keywords:
      - "military encrypted"
      - "russian military"
  - threat_level: HIGH
    keywords:
      - "military"
      - "stanag"
      - "nato"
  - threat_level: LOW
    keywords:
      - "amateur"
      - "commercial"
  - threat_level: INFORMATIONAL
    keywords:
      - "broadcast"
      - "navigation"
```

Matching is case-insensitive against the Artemis signal name and description.

## config/exclusions.yaml

Suppress known/benign signals from detection results.

```yaml
exclusions:
  - freq_hz: 89100000       # Centre frequency in Hz
    bandwidth_hz: 200000     # Exclusion zone width
    label: "Local FM station"
```

Signals whose frequency band overlaps any exclusion entry are removed from results.

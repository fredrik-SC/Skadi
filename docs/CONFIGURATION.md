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

### Fingerprint Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprint.min_snr_db` | `8.0` | Minimum SNR for modulation classification |
| `fingerprint.guard_factor` | `3.0` | Filter bandwidth multiplier for isolation |
| `fingerprint.filter_numtaps` | `101` | Minimum FIR filter taps (auto-scaled for narrowband) |
| `fingerprint.acf_min_lag_ms` | `1.0` | ACF search range minimum (ms) |
| `fingerprint.acf_max_lag_ms` | `5000.0` | ACF search range maximum (ms) |
| `fingerprint.acf_min_peak_strength` | `0.3` | Minimum ACF peak to report |

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

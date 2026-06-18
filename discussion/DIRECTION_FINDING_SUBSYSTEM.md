# Direction-Finding Subsystem (Dual-Tuner, Coherent) — Discussion

**Status:** Discussion (exploratory — not committed scope)
**Date:** 18 June 2026
**Author:** Fredrik / Claude
**Relates to:** PRD §7 (Future Capabilities 1–3: DF, GPS, map), PRD §3.6.1 (v2.0 log
fields `bearing_deg`, `latitude`, `longitude`, `mgrs`), CLAUDE.md "no DF / no dual-tuner
in v1.0".

---

## 1. Purpose

Sketch a feasible direction-finding (DF) subsystem for a **mobile SIGINT collection
unit** — a vehicle-mounted Skaði box that not only identifies signals (current
capability) but estimates the **bearing** to an emitter and, over a driven track,
**geolocates** it. This document is a feasibility sketch and phasing proposal, not a
specification. It exists to capture the design before any of it becomes committed scope.

The motivating mission (PRD §1.2): cut the "two weeks to actionable" delay by letting an
operator in the field not just classify a hostile signal in real time but get a *bearing
and a location fix* on it.

## 2. Hardware baseline (already owned)

The unit uses an **SDRplay RSPduo** — SDRplay's coherent dual-tuner receiver. This is the
key enabler: its two 14-bit tuners share a single clock and ADC reference, so the two
channels are **frequency-coherent** (unlike two independent RSP boxes, which cannot be
synchronised over SoapySDR). Two antenna inputs (Tuner 1 50Ω, Tuner 2 50Ω) feed a
**2-element interferometer**.

Reference platform (from the mobile-demo discussion): Raspberry Pi 5 (8 GB) + RSPduo +
NVMe + active cooling + USB GPS + small display + battery, in a 3D-printed vehicle box.
Compute is **not** the constraint — KrakenSDR runs 5-channel coherent DF DSP on a Pi 4,
so a Pi 5 handles 2-channel comfortably.

## 3. What changes from the current system

Skaði today drives the RSPduo in **single-tuner mode** (up to 10 MHz, wideband scan).
DF requires the **dual-tuner synchronous (coherent) mode**, which is a different
operating mode with two important consequences:

- **Bandwidth drops to ~2 MHz per tuner** in dual-tuner mode (PRD already notes this).
  You cannot do a 10 MHz scan and DF at the same time on one RSPduo.
- The two modes are therefore **mutually exclusive** → the unit operates in one of two
  states and toggles between them (see §5).

DF is otherwise **entirely new software** — there is no coherent capture, phase
calibration, angle-of-arrival, or geolocation code in the project today.

## 4. Signal-processing pipeline

```
 Antenna A ─┐                                  ┌─ φ_cal (stored)
 Antenna B ─┤  RSPduo dual-tuner (coherent)    │
            └─► [coherent capture] ─► [sample  ▼     ┌─ GPS (lat/lon/heading/time)
                 2× IQ @ ~2 Msms      align] ─► [phase ─► [AoA θ] ─► [running-fix ─► emitter
                 same f0, locked clk            cal]      (interfer.)   geolocation]   lat/lon
                                                                                        + CEP
                                                                          │
                                                                          ▼
                                                      detection log (bearing_deg, lat, lon, mgrs)
                                                                + web map
```

### 4.1 Coherent capture
Both tuners tuned to the same `f0`, same sample rate, streamed as two sample-aligned IQ
buffers. **Open risk:** verify that SoapySDRPlay3 exposes coherent dual-channel streaming
with deterministic sample alignment (see §7). May require a startup cross-correlation to
measure and correct an integer-sample offset between the channels.

### 4.2 Phase calibration
Shared clock gives frequency coherence, but the inter-channel phase offset `φ_cal` is
fixed-but-unknown each session and must be removed before bearings mean anything. The
RSPduo has **no built-in calibration noise source** (KrakenSDR does), so options are:
- **Injected reference:** a noise source / CW tone split into both ports through a known
  network; measure Δφ, store as `φ_cal`. Needs a splitter + (ideally) an RF switch.
- **Known-bearing self-cal:** calibrate against a transmitter at a known location/bearing.
  Practical to bootstrap, awkward to repeat in the field.
- The two RF paths (cables, connectors, tuner front-ends) must also be **phase-matched or
  characterised** — differences fold into `φ_cal`.

### 4.3 Angle of arrival (2-element interferometry)
Bearing from inter-channel phase: `Δφ = (2π · d · sinθ) / λ`, solve for `θ`, where `d`
is the antenna baseline and `λ` the wavelength.
- A single 2-element baseline yields an **ambiguous** bearing (front/back, and phase-wrap
  if `d > λ/2`). Resolve via **platform motion** (multiple bearings along the track break
  the ambiguity) and/or a directional element.
- 2 elements ⇒ phase interferometry is the natural estimator; MUSIC is possible but a
  2-element array has minimal degrees of freedom, so it buys little over interferometry.

### 4.4 Running-fix geolocation
A single bearing from a moving platform isn't a location. Log `(bearing, GPS position,
heading, time)` continuously; over a track, intersect multiple bearings to a stationary
emitter (least-squares / Stansfield estimator) to produce an **emitter lat/lon + an
uncertainty ellipse (CEP)**. This is classic mobile "running fix" DF and is where the
GPS + motion of a vehicle becomes an asset rather than a complication.

## 5. Operating concept (scan ↔ DF toggle)

1. **Collect mode (single-tuner, today's behaviour):** wideband scan, detect, fingerprint,
   classify, log with GPS position. This already produces an RSSI/position trail — the
   single-SDR "heat-map geolocation" fallback (§8) lives here for free.
2. **DF mode (dual-tuner coherent):** operator (or an auto-cue rule, e.g. threat ≥ HIGH or
   `unexpected_for_band`) selects a signal of interest → retune both tuners to it →
   measure bearing → accumulate fixes over motion → geolocate.

The two modes time-share one receiver. A natural loop: scan → cue a signal → drop into DF
for N seconds while driving → return to scan.

## 6. Proposed module layout

New code, mirroring the existing `src/<subsystem>/` convention:

| Module | Responsibility |
|--------|----------------|
| `src/sdr/coherent.py` | `CoherentDualSource` — dual-tuner synchronous capture, sample alignment (drop-in alongside the existing `SdrSource` interface) |
| `src/df/calibration.py` | Measure / store / apply `φ_cal`; path-match bookkeeping |
| `src/df/aoa.py` | Angle-of-arrival from coherent IQ (interferometry; optional MUSIC) |
| `src/df/geolocate.py` | Running-fix triangulation → emitter lat/lon + CEP |
| `src/gps/reader.py` | NMEA-over-USB GPS reader (position, heading, time, fix quality) |
| `src/web/` (extend) | Map view: emitter markers, bearing lines, uncertainty ellipses |

Integration points that **already exist**:
- Detection-log schema already carries `bearing_deg`, `latitude`, `longitude`, `mgrs`
  (PRD §3.6.1, v2.0) — DF populates them; SEIARA consumes them unchanged.
- `ReplaySource` pattern → add a **coherent two-channel recording/replay** format so DF
  algorithms are testable offline against recorded coherent IQ (essential — DF is hard to
  debug live in a moving vehicle).

## 7. Constraints, risks & open questions

| # | Item | Note |
|---|------|------|
| R1 | **SoapySDRPlay3 coherent dual-stream support** | *Biggest unknown.* Verify the API delivers two sample-aligned streams in dual-tuner mode. If not, fall back to the native SDRplay API 3.x (also the path for the shutdown-crash fix). **Resolve first (DF-0).** |
| R2 | 2 MHz/tuner in dual mode | Can't scan wide and DF simultaneously; drives the mode-toggle design (§5). |
| R3 | Phase calibration hardware | No built-in noise source; needs splitter/switch or a cal procedure (§4.2). |
| R4 | Baseline geometry | Corner-to-corner (~2–4 m) is ambiguous at VHF/UHF (`d ≫ λ/2`) and insensitive at HF (`d ≪ λ`). Prefer a **roof-mounted λ/2 pair for a chosen band**. |
| R5 | Band coverage | DF is realistically **VHF/UHF**. HF DF needs an impractically large baseline → out of scope for the vehicle build. |
| R6 | 2-element ambiguity | Front/back + wrap; resolved by motion and/or a directional element, not by the array alone. |
| R7 | Vehicle multipath | Body reflections and re-radiation bias bearings; mounting and calibration matter; field trials required. |
| R8 | Antenna/cable phase match | Two RF paths must be matched or characterised into `φ_cal`. |

## 8. Lower-risk precursor / fallback: RSSI geolocation

Independent of coherent DF, **Collect mode already enables single-SDR geolocation**: log
detection RSSI + GPS along the track and estimate emitter position from the spatial
signal-strength gradient (and/or max-RSSI closest-approach). No coherence, no calibration,
no second tuner. It's lower-accuracy than interferometry but robust, works on the hardware
as-is, and is a sensible **DF v0** that de-risks the GPS/geolocation/map plumbing before
the coherent work begins.

## 9. Suggested phasing

| Phase | Goal | Exit criterion |
|-------|------|----------------|
| **DF-0** | Feasibility spike: coherent dual-tuner capture via SoapySDR (or native API). Bench test with a common CW source into both ports. | Two sample-aligned coherent IQ streams captured; stable measured Δφ for a fixed setup. **Go/no-go gate for R1.** |
| **DF-1** | Phase calibration + bench interferometry against a known source at a known bearing, fixed baseline. | Bearing accuracy characterised (e.g. ±X° at VHF) on the bench. |
| **DF-2** | GPS integration + running-fix geolocation; coherent record/replay for offline test. | Emitter geolocated from a recorded mobile track; results written to the detection-log DF fields. |
| **DF-3** | Web map UI + operator cueing + scan↔DF mode toggle. | Operator can cue a logged signal, take a fix while driving, and see the emitter + uncertainty on the map. |
| **DF-4** | Field trials: calibration robustness, multipath, baseline tuning, ambiguity handling. | Repeatable fixes in a real environment; documented accuracy envelope. |
| (parallel) | **RSSI geolocation (DF v0, §8)** — can proceed immediately, independent of R1. | Heat-map / closest-approach emitter estimate logged with GPS. |

## 10. Recommendation

The RSPduo removes the hardest hardware blocker (a coherent 2-channel front end), so a
2-element interferometric DF subsystem is **feasible as a bounded future phase** — but it
is a genuinely new subsystem (coherent capture + calibration + AoA + geolocation), gated
on one unknown (R1, SoapySDR coherent streaming) that should be settled by a short DF-0
spike before committing scope. In the meantime, **RSSI geolocation (§8) is the low-risk
first step**: it stands up the GPS/map/geolocation pipeline on the current hardware and
delivers mobile localisation value while the coherent path is de-risked.

Nothing here is committed scope. If we pursue it, fold the agreed pieces (DF requirements,
GPS, map) into PRD §7 / a PLANNING phase and link back to this document.

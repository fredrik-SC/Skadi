# Skaði Live Digimode Capture Protocol

Generate **known** digital-mode signals with your own transmitters and capture
them live on the RSPduo, to build a lossless, ground-truth-labelled corpus for
calibrating and (later) training the modulation classifier. Because you transmit
it, the label is certain — and unlike the Artemis OGG samples there is no lossy
compression destroying the phase/frequency structure.

Transmitters: **Icom ID-50** (2 m/70 cm, D-STAR DV = GMSK) and **Icom IC-705**
(HF, via *SDR-Control* on iPad: RTTY, FT4, FT8). Receiver: RSPduo + the machine
running Skaði.

---

## 0. SAFETY FIRST — no attenuator

You have **no attenuator**, so the only things protecting the RSPduo front end are
**distance, low power, and receiver gain reduction.** A strong signal can *destroy*
the SDR (damage ≈ +10 dBm; overload/clipping far lower). Free-space path loss over
~200 ft is only **~31 dB at 14 MHz** but **~51 dB at 145 MHz** — so **HF is the
real hazard**: at 5 W on 20 m, 200 ft, you could see roughly **0 dBm** at the
antenna, which is too hot.

**Hard rules:**
1. **Start weak, ramp up.** Begin at the transmitter's *minimum* power (IC-705 ≈
   0.5–1 W; ID-50 ≈ 0.5–1 W). Only increase if the capture is too weak.
2. **De-sensitize the receiver** (see §1): AGC off, high gain reduction, and a
   **short/minimal RX antenna** (a short wire is a free attenuator for a close,
   strong transmitter).
3. **Check for clipping** after the first capture (§5) *before* trusting any data
   or increasing power.
4. **HF needs the most care.** Do the VHF/UHF D-STAR capture first to get
   comfortable — it has ~20 dB more path loss and is far more forgiving.
5. Never connect a transmitter to the SDR by cable. Over-the-air only, with
   separation.

---

## 1. Receiver setup for strong local signals

Edit `config/default.yaml` `sdr:` for these captures (revert afterwards):
```yaml
sdr:
  agc: false            # AGC can't protect against a strong local signal
  gain_reduction: 50    # high IFGR (dB) = low gain; raise toward 59 if clipping
```
Use a **short, deliberately poor RX antenna** on the RSPduo (a few-inch wire is
ideal for a close transmitter). The goal is the *weakest usable* signal, not the
strongest.

---

## 2. Geometry and logistics

- **Transmitter ~200 ft away**, lowest power. RSPduo + laptop at the house.
- The TX must run **while you operate Skaði** at the laptop, so make it transmit
  **hands-free / repeatedly**:
  - **IC-705 + SDR-Control (iPad):** the iPad goes to the TX site. Set FT8/FT4 to
    auto-repeat (call CQ / auto-sequence) or RTTY to loop a long macro, so it keys
    every cycle on its own. Then walk back and capture.
  - **ID-50 (D-STAR):** key PTT (use PTT-lock if available) at the TX site for the
    capture window — D-STAR DV transmits a continuous data frame while keyed even
    in silence. This one is the most awkward hands-free; a short timed run or a
    helper helps.

---

## 3. Per-mode capture recipes

For each, transmit the mode, then run **one** Skaði record. Pick a **clear**
in-band frequency (avoid the busy FT8/FT4 watering holes so the capture is one
clean signal and you cause no QRM). Confirm every frequency is legal for your
licence/band plan.

| Mode | TX | Modulation (label) | Notes |
|------|----|--------------------|-------|
| **D-STAR DV** | ID-50, 2 m/70 cm simplex | FSK (GMSK) | ~6 kHz wide, continuous while keyed. **Easiest + safest — do first.** |
| **RTTY** | IC-705 / SDR-Control, HF | FSK | 170 Hz shift, 45 baud. Loop a long RYRY test. Good easy HF case. |
| **FT4** | IC-705 / SDR-Control, HF | FSK | 4-FSK, ~90 Hz wide, 7.5 s bursts. Narrow — harder. |
| **FT8** | IC-705 / SDR-Control, HF | FSK | 8-FSK, ~50 Hz wide, 15 s bursts. Very narrow — hardest. |

**Capture command** (one session per mode; HF uses the finer-resolution `hf`
preset, VHF a custom range around the DV frequency):

```bash
# D-STAR on, say, 145.375 MHz (2m DV simplex)
python -m src.main --no-web --single --start 144.4e6 --stop 145.4e6 \
    --record sessions/live_dstar --label "D-STAR DV"

# HF modes — e.g. transmitting around 14.083 MHz; --preset hf gives 125 Hz/bin
python -m src.main --no-web --single --preset hf \
    --record sessions/live_rtty --label "RTTY"
# (repeat for FT4 -> sessions/live_ft4, FT8 -> sessions/live_ft8)
```

Timing: for **continuous** modes (D-STAR, looped RTTY) capture anytime while
transmitting. For **burst** modes (FT4/FT8) the capture must land inside a TX
window — set it auto-repeating and just re-run the capture until one lands in a
burst, or raise `scan.dwell_time` in config to ≥ 8 s (FT4) / ≥ 16 s (FT8) so a
single capture is guaranteed to span a full burst.

---

## 4. Labelling (ground truth)

Each session needs a `truth.json`. You know the mode and roughly the frequency;
find the exact detected centre by replaying once and reading the detected
frequency, then write:

```json
{
  "session": "live_rtty",
  "match_tolerance_hz": 5000,
  "signals": [
    { "center_freq_hz": 14083000, "label": "RTTY 45/170",
      "modulation": "FSK", "signal_type": "RTTY", "verified": true }
  ]
}
```
Modulation labels: RTTY/FT4/FT8/D-STAR are all **FSK**. (signal_type is free text —
use the mode name.)

---

## 5. Verify every capture before trusting it

```bash
# Replay and inspect: one clean signal, good SNR, NOT clipping
python -m src.benchmark --session sessions/live_rtty
```
Checks:
- **One** detection near your TX frequency, high SNR.
- **No clipping**: a quick check that the recorded IQ isn't railing —
  `venv/bin/python -c "import numpy as np; from src.sdr.sigmf import read_iq; x=read_iq('sessions/live_rtty/<step>'); print('max |I/Q|:', float(np.max(np.abs(np.concatenate([x.real,x.imag])))))"`.
  If it's pinned near the ADC full scale (clipping), **increase gain_reduction or
  lower TX power and recapture** — clipped data is useless and means the front end
  is being hammered.

---

## 6. Corpus notes

- **Class balance:** RTTY, FT4, FT8 and D-STAR are **all FSK-family**. That's
  excellent for nailing FSK (the bulk of the mission), but the corpus will have no
  true **PSK** example. If *SDR-Control* (or fldigi on a laptop) can send **PSK31**,
  add one PSK capture for balance; otherwise note PSK as a gap to fill later.
- **Repeat for variety:** capture each mode a few times, ideally at slightly
  different power/positions, so the model sees real-signal variation (fading,
  noise) rather than one perfect example.
- One session per mode keeps labelling trivial; the benchmark scores each, and the
  collection becomes the training set for Phase E (feature-based ML).

---

## 7. After capturing — revert receiver settings

Restore `config/default.yaml` `sdr.agc: true` and `sdr.gain_reduction: 0` for
normal scanning, and reconnect your proper antenna.
```

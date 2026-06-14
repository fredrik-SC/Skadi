"""Convert demodulated digimode audio samples into labelled IQ sessions.

BARTG and Artemis ship audio recordings of digital modes (RTTY, PSK31, MFSK,
JT65, ...). Those are demodulated, real, audio-rate signals — not RF IQ. But an
AFSK/SSB audio recording is spectrally equivalent to the RF signal shifted to
baseband, so we can recover a complex analytic signal, centre it, resample to a
chosen IQ rate, drop it into a noisy wideband step, and emit a SigMF session the
existing replay/benchmark pipeline understands.

This builds a labelled digital-mode corpus (the FSK/PSK family the mission cares
about) without depending on live HF propagation — ideal, reproducible regression
material for the modulation classifier.
"""

from src.ingest.audio_iq import (
    audio_to_iq,
    build_session,
    infer_modulation,
    load_audio,
)

__all__ = ["audio_to_iq", "build_session", "infer_modulation", "load_audio"]

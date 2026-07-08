"""Silero-VAD speech interval detection."""
from pathlib import Path

import librosa
import numpy as np
import torch

# Silero-VAD auto-downloads model on first import
MODEL, UTILS = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
)

get_speech_timestamps = UTILS[0]


def get_speech_intervals(
    audio_path: str | Path,
    sample_rate: int = 22050,
    threshold: float = 0.5,
    min_speech_dur: float = 0.3,
    min_silence_dur: float = 0.3,
) -> list[dict]:
    """Scan audio with Silero-VAD, return list of {start, end} speech intervals.

    Steps:
    1. Load audio via librosa (mono, resample to 16000 for VAD)
    2. Use Silero-VAD's built-in ``get_speech_timestamps`` for frame-level
       processing, thresholding, and interval merging.
    3. Scale timestamps back to input ``sample_rate`` timebase.
    """
    # Load audio
    audio, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Resample to 16kHz for VAD (Silero expects 16k)
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        vad_sr = 16000
    else:
        audio_16k = audio
        vad_sr = 16000

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_16k.astype(np.float32))

    # Run Silero-VAD built-in detector
    raw = get_speech_timestamps(
        audio_tensor,
        MODEL,
        threshold=threshold,
        sampling_rate=vad_sr,
        min_speech_duration_ms=int(min_speech_dur * 1000),
        min_silence_duration_ms=int(min_silence_dur * 1000),
        return_seconds=True,
    )

    intervals = []
    for segment in raw:
        intervals.append({
            "start": segment["start"],
            "end": segment["end"],
        })

    return intervals

"""Load mono 16 kHz audio for Essentia, with optional ffmpeg fallback."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from essentia.standard import MonoLoader, AudioLoader
import numpy as np
import gc
import time
import os
ESSENTIA_FORMAT = {'.wav', '.mp3', '.flac', '.aiff', '.ogg'}

def load_audio(
    path: Path | str,
    track: Literal["mono", "stereo"] = "mono",
    sample_rate: int = 44100,
    resample_quality: int = 1,
) -> Any:
    """Return mono float32 audio (numpy). Try Essentia ``MonoLoader``; on failure decode via ffmpeg to a temp WAV."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    use_essentia = ext in ESSENTIA_FORMAT
    if ext in ESSENTIA_FORMAT:
        try:
            audio, out_sample_rate = _load_audio(
                path=p,
                track=track,
                sample_rate=sample_rate,
                resample_quality=resample_quality,
            )
        except Exception:
            raise Exception(f"the file {p.name} do not work with essentia (fix code with this file extention)")
    else:
        tmp_path = _load_via_ffmpeg(p)
        audio, out_sample_rate = _load_audio(
            path=tmp_path,
            track=track,
            sample_rate=sample_rate,
            resample_quality=resample_quality,
        )
        tmp_path.unlink(missing_ok=True)
    return audio, out_sample_rate

def _load_audio(
    path: Path,
    track: str,
    sample_rate: int,
    resample_quality: int,
) -> Any:
    if track == "mono":
        audio = MonoLoader(
            filename=str(path.resolve()),
            sampleRate=sample_rate,
            resampleQuality=resample_quality,
        )()
        out_sample_rate = sample_rate
    elif track == "stereo":
        audio, out_sample_rate, _, _, _, _ = AudioLoader(filename=str(path.resolve()))()
    else:
        raise Exception("track property impossible")
    return np.asarray(audio, dtype=np.float32), out_sample_rate
    


def _load_via_ffmpeg(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path.resolve()),
        "-f",
        "wav",
        str(tmp_path.resolve()),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return tmp_path

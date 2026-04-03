"""Chargement audio mono 16 kHz pour Essentia, avec repli ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000


def load_mono_16k(
    path: Path | str,
    *,
    resample_quality: int = 4,
    ffmpeg_fallback: bool = True,
) -> Any:
    """
    Charge l'audio en mono flottant, 16 kHz (numpy).

    Essaie `MonoLoader` Essentia ; en cas d'échec, décode via ffmpeg en WAV temporaire.
    """

    import numpy as np

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    try:
        from essentia.standard import MonoLoader

        loader = MonoLoader(
            filename=str(p.resolve()),
            sampleRate=SAMPLE_RATE,
            resampleQuality=resample_quality,
        )
        audio = loader()
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        if not ffmpeg_fallback:
            raise
        return _load_via_ffmpeg(p, resample_quality=resample_quality)


def _load_via_ffmpeg(path: Path, *, resample_quality: int) -> Any:
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(path.resolve()),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "wav",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        from essentia.standard import MonoLoader

        loader = MonoLoader(
            filename=str(tmp_path),
            sampleRate=SAMPLE_RATE,
            resampleQuality=resample_quality,
        )
        return np.asarray(loader(), dtype=np.float32)
    finally:
        tmp_path.unlink(missing_ok=True)


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

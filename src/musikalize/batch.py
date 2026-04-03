"""Liste de fichiers et traitement parallèle."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from musikalize.config import AnalysisConfig, ClassificationHeadSpec, ExportConfig, ModelPath, TaggingConfig
from musikalize.process import MusicProcess

_DEFAULT_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".mpc",
        ".wv",
    }
)


def list_audio_files(
    root: Path | str,
    *,
    extensions: frozenset[str] | set[str] | None = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """Liste les fichiers audio sous ``root``."""

    r = Path(root)
    if not r.is_dir():
        raise NotADirectoryError(r)
    ext = extensions if extensions is not None else _DEFAULT_EXTENSIONS
    norm = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext}
    out: list[Path] = []
    if recursive:
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in norm:
                out.append(p)
    else:
        for p in r.iterdir():
            if p.is_file() and p.suffix.lower() in norm:
                out.append(p)
    if sort_paths:
        out.sort()
    return out


def _worker_process_one(
    audio_path: str,
    model_path_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
    tagging_dict: dict[str, Any],
    export_dict: dict[str, Any] | None,
) -> tuple[str, bool, str | None]:
    """Exécute un fichier dans un worker (sérialisable)."""

    try:
        mp = ModelPath(
            embedding_model=Path(model_path_dict["embedding_model"]),
            embedding_output=model_path_dict.get("embedding_output", "PartitionedCall:1"),
            heads=tuple(
                ClassificationHeadSpec(
                    name=h["name"],
                    graph_path=Path(h["graph_path"]),
                    labels_path=Path(h["labels_path"]) if h.get("labels_path") else None,
                    input_tensor=h.get("input_tensor", "serving_default_model_Placeholder"),
                    output_tensor=h.get("output_tensor", "PartitionedCall:0"),
                )
                for h in model_path_dict.get("heads", [])
            ),
        )
        ad = dict(analysis_dict)
        gs = ad.get("genre_separators")
        if isinstance(gs, list):
            ad["genre_separators"] = tuple(gs)
        ac = AnalysisConfig(**ad)
        tc = TaggingConfig(**tagging_dict)
        ec = (
            ExportConfig(**{**export_dict, "output_root": Path(export_dict["output_root"])})
            if export_dict
            else None
        )
        proc = MusicProcess(
            audio_file=Path(audio_path),
            model_path=mp,
            analysis_config=ac,
            tagging_config=tc,
            export_config=ec,
        )
        proc.process_file()
        return (audio_path, True, None)
    except Exception as e:
        return (audio_path, False, str(e))


def _serialize_model_path(mp: ModelPath) -> dict[str, Any]:
    return {
        "embedding_model": str(mp.embedding_model),
        "embedding_output": mp.embedding_output,
        "heads": [
            {
                "name": h.name,
                "graph_path": str(h.graph_path),
                "labels_path": str(h.labels_path) if h.labels_path else None,
                "input_tensor": h.input_tensor,
                "output_tensor": h.output_tensor,
            }
            for h in mp.heads
        ],
    }


def _serialize_dataclass(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def process_files_parallel(
    paths: Sequence[Path | str],
    *,
    model_path: ModelPath,
    analysis_config: AnalysisConfig | None = None,
    tagging_config: TaggingConfig | None = None,
    export_config: ExportConfig | None = None,
    max_workers: int | None = None,
) -> list[tuple[str, bool, str | None]]:
    """
    Traite plusieurs fichiers en parallèle (un processus par worker).

    Chaque worker charge Essentia / TensorFlow pour chaque fichier (coûteux) ;
    pour de très gros lots, préférez un pool externe ou un cache de modèles.
    """

    ac = analysis_config or AnalysisConfig()
    tc = tagging_config or TaggingConfig()
    mp_d = _serialize_model_path(model_path)
    ad = _serialize_dataclass(ac)
    td = _serialize_dataclass(tc)
    ed = _serialize_dataclass(export_config) if export_config is not None else None
    if ed is not None:
        ed["output_root"] = str(export_config.output_root)

    str_paths = [str(Path(p)) for p in paths]
    results: list[tuple[str, bool, str | None]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_worker_process_one, p, mp_d, ad, td, ed) for p in str_paths
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda x: x[0])
    return results

"""Audio file listing and optional parallel processing."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from musikalyze.config import EmbeddingModel, ExportConfig, LabelExtractor, TaggingConfig
from musikalyze.process import MusicProcess

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

def sample_audio_files(
    root: Path | str,
    sample: int | float = 0,
    extensions: frozenset[str] | set[str] | None = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """List a sample of audio files under ``root`` with sample as a ratio if 0<sample<1 or a sample."""
    import random as rnd
    full_list = list_audio_files(root=root,extensions=extensions,recursive=recursive,sort_paths=sort_paths)
    if sample == 0:
        return full_list
    elif sample < 1:
        return rnd.sample(full_list, round(sample * len(full_list)))
    else:
        return rnd.sample(full_list, round(sample))

def list_audio_files(
    root: Path | str,
    *,
    extensions: frozenset[str] | set[str] | None = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """List audio files under ``root``."""

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


def _embedding_from_dict(d: dict[str, Any]) -> EmbeddingModel:
    return EmbeddingModel(
        name=d["name"],
        embedding_model=Path(d["embedding_model"]),
        embedding_output=d.get("embedding_output", "PartitionedCall:1"),
        backend=d.get("backend", "effnet"),
        input_tensor=d.get("input_tensor"),
        patch_size=d.get("patch_size"),
        patch_hop_size=d.get("patch_hop_size"),
        batch_size=d.get("batch_size"),
    )


def _extractor_from_dict(d: dict[str, Any]) -> LabelExtractor:
    gs = d.get("genre_separators")
    if isinstance(gs, list):
        d = {**d, "genre_separators": tuple(gs)}
    return LabelExtractor(
        name=d["name"],
        category=d["category"],
        embedder=d["embedder"],
        graph_path=Path(d["graph_path"]),
        labels_path=Path(d["labels_path"]) if d.get("labels_path") else None,
        label_names=d.get("label_names"),
        input_tensor=d.get("input_tensor", "serving_default_model_Placeholder"),
        output_tensor=d.get("output_tensor", "PartitionedCall:0"),
        task=d.get("task", "classification"),
        genre_main=d.get("genre_main", True),
        genre_count=d.get("genre_count", 5),
        genre_thold=d.get("genre_thold"),
        genre_separators=tuple(d.get("genre_separators", ("---", "//"))),
        genre_join_separator=d.get("genre_join_separator", ";"),
        count_thold_policy=d.get("count_thold_policy", "intersection"),
    )


def _worker_process_one(
    audio_path: str,
    embedder_dicts: list[dict[str, Any]],
    extractor_dicts: list[dict[str, Any]],
    tagging_dict: dict[str, Any],
    export_dict: dict[str, Any] | None,
) -> tuple[str, bool, str | None]:
    try:
        embedders = tuple(_embedding_from_dict(x) for x in embedder_dicts)
        extractors = tuple(_extractor_from_dict(x) for x in extractor_dicts)
        td = dict(tagging_dict)
        if "extra" in td and hasattr(td["extra"], "items"):
            pass
        tc = TaggingConfig(**td)
        ec = (
            ExportConfig(**{**export_dict, "output_root": Path(export_dict["output_root"])})
            if export_dict
            else None
        )
        proc = MusicProcess(
            audio_file=Path(audio_path),
            embedders=embedders,
            label_extractors=extractors,
            tagging_config=tc,
            export_config=ec,
        )
        proc.process_file()
        return (audio_path, True, None)
    except Exception as e:
        return (audio_path, False, str(e))


def _serialize_embedding(e: EmbeddingModel) -> dict[str, Any]:
    d = asdict(e)
    d["embedding_model"] = str(e.embedding_model)
    return d


def _serialize_extractor(e: LabelExtractor) -> dict[str, Any]:
    d = asdict(e)
    d["graph_path"] = str(e.graph_path)
    d["labels_path"] = str(e.labels_path) if e.labels_path else None
    if isinstance(e.embedder, EmbeddingModel):
        d["embedder"] = e.embedder.name
    return d


def process_files_parallel(
    paths: Sequence[Path | str],
    *,
    embedders: Sequence[EmbeddingModel],
    label_extractors: Sequence[LabelExtractor],
    tagging_config: TaggingConfig | None = None,
    export_config: ExportConfig | None = None,
    max_workers: int | None = None,
) -> list[tuple[str, bool, str | None]]:
    """Process paths in parallel (one subprocess per worker)."""

    tc = tagging_config or TaggingConfig()
    td = asdict(tc)
    ed = asdict(export_config) if export_config is not None else None
    if ed is not None:
        ed["output_root"] = str(export_config.output_root)

    emb_d = [_serialize_embedding(e) for e in embedders]
    ex_d = [_serialize_extractor(e) for e in label_extractors]

    str_paths = [str(Path(p)) for p in paths]
    results: list[tuple[str, bool, str | None]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_worker_process_one, p, emb_d, ex_d, td, ed) for p in str_paths
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda x: x[0])
    return results

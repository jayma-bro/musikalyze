"""Batch processing: file discovery and the unified ``MusicBatch`` pipeline."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from musikalyze.config import EmbeddingModel, ExportConfig, LabelExtractor, TaggingConfig
from musikalyze.process import MusicProcess

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

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


def sample_audio_files(
    root: Path | str,
    sample: float = 0,
    extensions: frozenset[str] | set[str] | None = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """List a sample of audio files under ``root`` with sample as a ratio if 0<sample<1 or a sample."""
    import random as rnd

    full_list = list_audio_files(root=root, extensions=extensions, recursive=recursive, sort_paths=sort_paths)
    if sample == 0:
        return full_list
    elif sample < 1:
        return rnd.sample(full_list, round(sample * len(full_list)))
    else:
        return rnd.sample(full_list, round(sample))


# ---------------------------------------------------------------------------
# Config serialization for ProcessPoolExecutor workers
# ---------------------------------------------------------------------------


def _serialize_embedding(e: EmbeddingModel) -> dict[str, Any]:
    d = asdict(e)
    d["embedding_model"] = str(e.embedding_model)
    return d


def _serialize_extractor(e: LabelExtractor) -> dict[str, Any]:
    d = asdict(e)
    d["graph_path"] = str(e.graph_path)
    d["labels_path"] = str(e.labels_path) if e.labels_path else None
    return d


def _deserialize_embedding(d: dict[str, Any]) -> EmbeddingModel:
    d = dict(d)
    d["embedding_model"] = Path(d["embedding_model"])
    return EmbeddingModel(**d)


def _deserialize_extractor(d: dict[str, Any]) -> LabelExtractor:
    d = dict(d)
    d["graph_path"] = Path(d["graph_path"])
    if d.get("labels_path"):
        d["labels_path"] = Path(d["labels_path"])
    return LabelExtractor(**d)


def _worker_analyze_one(
    audio_path: str,
    embedder_dicts: list[dict[str, Any]],
    extractor_dicts: list[dict[str, Any]],
    separator: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Run a single file through the full analysis pipeline and return its labels dict."""
    try:
        embedders = tuple(_deserialize_embedding(d) for d in embedder_dicts)
        extractors = tuple(_deserialize_extractor(d) for d in extractor_dicts)
        proc = MusicProcess(
            audio_file=Path(audio_path),
            embedders=embedders,
            extractors=extractors,
            separator=separator,
        )
        return (audio_path, proc.labels, None)
    except Exception as e:
        return (audio_path, None, f"{type(e).__name__}: {e}")


def _worker_process_one(
    audio_path: str,
    embedder_dicts: list[dict[str, Any]],
    extractor_dicts: list[dict[str, Any]],
    tagging_dict: dict[str, Any],
    export_dict: dict[str, Any],
    separator: str,
) -> tuple[str, bool, str | None]:
    """Run a single file through the full tag + export pipeline."""
    try:
        embedders = tuple(_deserialize_embedding(d) for d in embedder_dicts)
        extractors = tuple(_deserialize_extractor(d) for d in extractor_dicts)
        tc = TaggingConfig(**tagging_dict)
        ed = dict(export_dict)
        ed["output_root"] = Path(ed["output_root"])
        ec = ExportConfig(**ed)
        proc = MusicProcess(
            audio_file=Path(audio_path),
            embedders=embedders,
            extractors=extractors,
            tagging_config=tc,
            export_config=ec,
            separator=separator,
        )
        proc.process_file()
        return (audio_path, True, None)
    except Exception as e:
        return (audio_path, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _row_from_labels(
    audio_path: str,
    labels: dict[str, Any] | None,
    key: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one DataFrame row: ``_path`` plus ``key`` (exploded if the value is a dict)."""
    row: dict[str, Any] = {"_path": audio_path}
    if error is not None or labels is None:
        logger.warning("Analysis failed for %s: %s", audio_path, error or "unknown error")
        return row
    if key not in labels:
        logger.warning("Key %r not found for %s", key, audio_path)
        row[key] = None
        return row
    value = labels[key]
    if isinstance(value, dict):
        row.update(value)
    elif isinstance(value, (list, tuple)):
        row[key] = json.dumps(value, ensure_ascii=False)
    else:
        row[key] = value
    return row


# ---------------------------------------------------------------------------
# MusicBatch
# ---------------------------------------------------------------------------


class MusicBatch:
    """Batch pipeline over a directory of audio files.

    Unified entry point for listing, analyzing, and exporting a music library.
    Heavy work is delegated to one :class:`MusicProcess` per file; each file is
    independent, so one failure is logged and skipped instead of aborting.

    Example
    -------
    >>> batch = MusicBatch("./library", embedders=[effnet], extractors=[genre400])
    >>> batch.files                      # list[str] of audio paths
    >>> df = batch.analyze("genre400_all")  # rows = files, columns = labels
    >>> batch.export("./out")            # transcode + tags, with progress bar
    """

    def __init__(
        self,
        audio_path: Path | str,
        embedders: Sequence[EmbeddingModel] = (),
        extractors: Sequence[LabelExtractor] = (),
        tagging_config: TaggingConfig | None = None,
        export_config: ExportConfig | None = None,
        separator: str = ";",
        *,
        recursive: bool = True,
        extensions: Iterable[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.root = Path(audio_path)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.embedders = tuple(embedders)
        self.extractors = tuple(extractors)
        self.tagging_config = tagging_config or TaggingConfig()
        self.export_config = export_config
        self.separator = separator
        self.recursive = recursive
        self._extensions = frozenset(extensions) if extensions is not None else None
        self.max_workers = max_workers
        self._paths: list[Path] | None = None
        self._failures: list[tuple[str, str]] = []

    # -- discovery ----------------------------------------------------------

    @property
    def paths(self) -> list[Path]:
        """Audio file paths under :attr:`root` (sorted, cached after first access)."""
        if self._paths is None:
            self._paths = list_audio_files(
                self.root,
                extensions=self._extensions,
                recursive=self.recursive,
                sort_paths=True,
            )
        return self._paths

    @property
    def files(self) -> list[str]:
        """Audio file paths as strings."""
        return [str(p) for p in self.paths]

    def sample(self, n_or_ratio: float) -> MusicBatch:
        """Return a new ``MusicBatch`` restricted to a random sample of files.

        ``n_or_ratio`` is an absolute count (``>= 1``) or a ratio (``0 < x < 1``).
        """
        clone = MusicBatch(
            self.root,
            embedders=self.embedders,
            extractors=self.extractors,
            tagging_config=self.tagging_config,
            export_config=self.export_config,
            separator=self.separator,
            recursive=self.recursive,
            extensions=self._extensions,
            max_workers=self.max_workers,
        )
        clone._paths = sorted(sample_audio_files(self.root, sample=n_or_ratio, extensions=self._extensions))
        return clone

    def summary(self) -> dict[str, Any]:
        """Cheap overview: file count, extension histogram, total size on disk."""
        paths = self.paths
        ext_counts = Counter(p.suffix.lower() for p in paths)
        total_bytes = sum(p.stat().st_size for p in paths if p.exists())
        return {
            "root": str(self.root),
            "n_files": len(paths),
            "extensions": dict(sorted(ext_counts.items())),
            "total_size_bytes": total_bytes,
            "total_size_human": _human_size(total_bytes),
        }

    # -- heavy pipeline -----------------------------------------------------

    def analyze(self, key: str) -> pd.DataFrame:
        """Analyze every file and return a DataFrame with one row per file.

        ``key`` is a metadata key (``meta_`` prefix added if missing; ``tag_*``
        keys are read from file tags). If the resolved value is a dict (e.g.
        ``"genre400_all"`` → ``{label: score}``), it is exploded into one column
        per label; scalar keys yield a single column. Failed files produce a
        row with ``None`` values and a logged warning.
        """
        import pandas as pd

        if not key:
            raise ValueError("key is required")
        norm_key = key if key.startswith(("tag_", "meta")) else f"meta_{key}"
        if not self.paths:
            raise ValueError(f"No audio files found in {self.root}")

        results: list[tuple[str, dict[str, Any] | None, str | None]] = []
        if self._use_pool():
            emb_d = [_serialize_embedding(e) for e in self.embedders]
            ex_d = [_serialize_extractor(e) for e in self.extractors]
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futs = [
                    pool.submit(_worker_analyze_one, str(p), emb_d, ex_d, self.separator)
                    for p in self.paths
                ]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Analyzing {norm_key}", unit="file"):
                    results.append(fut.result())
        else:
            for p in tqdm(self.paths, desc=f"Analyzing {norm_key}", unit="file"):
                try:
                    results.append((str(p), self._make_process(p).labels, None))
                except Exception as e:
                    results.append((str(p), None, f"{type(e).__name__}: {e}"))

        rows = [_row_from_labels(path, labels, norm_key, error) for path, labels, error in results]
        columns: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for c in row:
                if c not in seen:
                    seen.add(c)
                    columns.append(c)
        df = pd.DataFrame(rows, columns=columns)
        return df.sort_values("_path", kind="stable").reset_index(drop=True)

    def export(self, folder: Path | str) -> None:
        """Run the full pipeline (tags → analyze → tag templates → ffmpeg export) on every file.

        ``folder`` becomes the export output root, overriding
        ``export_config.output_root``. If no ``export_config`` was given, a
        default one (``formats="opus"``, default path template) is used.
        Failures are logged and skipped; see ``self._failures`` afterwards.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not self.paths:
            logger.warning("No audio files found in %s", self.root)
            return

        export_cfg = self._effective_export_config(folder)
        failures: list[tuple[str, str]] = []
        if self._use_pool():
            emb_d = [_serialize_embedding(e) for e in self.embedders]
            ex_d = [_serialize_extractor(e) for e in self.extractors]
            td = asdict(self.tagging_config)
            ed = asdict(export_cfg)
            ed["output_root"] = str(export_cfg.output_root)
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futs = [
                    pool.submit(_worker_process_one, str(p), emb_d, ex_d, td, ed, self.separator)
                    for p in self.paths
                ]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Exporting", unit="file"):
                    path, ok, error = fut.result()
                    if not ok:
                        logger.warning("Export failed for %s: %s", path, error)
                        failures.append((path, error or "unknown error"))
        else:
            bar = tqdm(total=len(self.paths), desc="Exporting", unit="file")
            for p in self.paths:
                try:
                    self._make_process(p, export_config=export_cfg).process_file()
                except Exception as e:
                    logger.warning("Export failed for %s: %s", p, e)
                    failures.append((str(p), f"{type(e).__name__}: {e}"))
                bar.update(1)
                bar.set_postfix(failed=len(failures))
            bar.close()

        self._failures = failures
        if failures:
            logger.warning("%d/%d exports failed", len(failures), len(self.paths))

    def preview_paths(self, ext: str = "opus") -> list[Path]:
        """Dry-run: resolved destination paths per file without writing anything.

        Requires ``export_config`` (the preview uses its ``output_root``).
        """
        if self.export_config is None:
            raise ValueError("export_config is required for preview_paths()")
        out: list[Path] = []
        for p in tqdm(self.paths, desc="Previewing paths", unit="file"):
            try:
                out.append(self._make_process(p).preview_path(ext))
            except Exception as e:
                logger.warning("Preview failed for %s: %s", p, e)
        return out

    # -- pythonic access ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[MusicProcess]:
        for p in self.paths:
            yield self._make_process(p)

    def __getitem__(self, index: int | slice) -> MusicProcess | list[MusicProcess]:
        if isinstance(index, slice):
            return [self._make_process(p) for p in self.paths[index]]
        return self._make_process(self.paths[index])

    def __repr__(self) -> str:
        return f"MusicBatch(root={str(self.root)!r}, n_files={len(self)})"

    # -- internals ----------------------------------------------------------

    def _use_pool(self) -> bool:
        return self.max_workers is not None and self.max_workers > 1

    def _make_process(self, audio_file: Path, export_config: ExportConfig | None = None) -> MusicProcess:
        return MusicProcess(
            audio_file=audio_file,
            embedders=self.embedders,
            extractors=self.extractors,
            tagging_config=self.tagging_config,
            export_config=export_config if export_config is not None else self.export_config,
            separator=self.separator,
        )

    def _effective_export_config(self, folder: Path) -> ExportConfig:
        if self.export_config is None:
            return ExportConfig(output_root=folder)
        return replace(self.export_config, output_root=folder)

"""Command-line interface for musikalyze."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from musikalyze.batch import MusicBatch
from musikalyze.config import EmbeddingModel, ExportConfig, LabelExtractor, TaggingConfig
from musikalyze.process import MusicProcess

log = logging.getLogger("musikalyze")


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> tuple[list[EmbeddingModel], list[LabelExtractor], TaggingConfig, dict[str, Any]]:
    """Load and resolve a CLI JSON configuration.

    Relative model, tempo-model and output paths are resolved relative to the
    configuration file. Returns the embedding models, label extractors,
    ``TaggingConfig`` and runtime/export options used by :func:`main`.
    """
    base = path.resolve().parent
    data = json.loads(path.read_text(encoding="utf-8"))
    embedders = []
    for name, raw in data.get("embedding_models", {}).items():
        item = dict(raw)
        item.setdefault("name", name)
        item["embedding_model"] = _path(item["embedding_model"], base)
        embedders.append(EmbeddingModel(**item))

    extractors = []
    for raw in data.get("label_extractors", []):
        item = dict(raw)
        item["graph_path"] = _path(item["graph_path"], base)
        if item.get("labels_path"):
            item["labels_path"] = _path(item["labels_path"], base)
        if isinstance(item.get("label_names"), dict):
            item["label_names"] = {
                str(label): tuple(bounds) for label, bounds in item["label_names"].items()
            }
        extractors.append(LabelExtractor(**item))

    tagging = TaggingConfig(**data.get("tagging_config", {}))
    export_data = dict(data.get("export_config", {}))
    if "output_root" in export_data:
        export_data["output_root"] = _path(export_data["output_root"], base)
    else:
        export_data["output_root"] = base / "output"
    export_config = ExportConfig(**export_data)
    tempo_model_path = data.get("tempo_model_path")
    return embedders, extractors, tagging, {
        "export_config": export_config,
        "tempo_model_path": _path(tempo_model_path, base) if tempo_model_path else None,
        "recursive": data.get("recursive", True),
        "extensions": data.get("extensions"),
    }


def _export_config(config: ExportConfig, output: Path) -> ExportConfig:
    return replace(config, output_root=output)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musikalyze", description="Analyse, tag and export audio files")
    parser.add_argument("input", type=Path, help="Audio file or directory containing audio files")
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Analyse, tag and export audio")
    export.add_argument("output", type=Path, help="Destination directory")


    analyze = sub.add_parser("analyze", help="Analyse audio and write a JSON result")
    analyze.add_argument("--key", default="analyze", help="Metadata key or 'analyze'")
    analyze.add_argument("--output", type=Path, help="Optional JSON output file")

    preview = sub.add_parser("preview", help="Print the resolved output paths")
    preview.add_argument("--extension", default="opus", help="Extension used for preview paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``musikalyze`` command-line application.

    ``argv`` may provide arguments explicitly for embedding or tests; when it
    is ``None``, arguments are read from the process command line. The command
    returns zero on success and a non-zero status when batch exports fail.
    """
    parser = _make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if not args.config.is_file():
        parser.error(f"Configuration does not exist: {args.config}")

    embedders, extractors, tagging, options = load_config(args.config)
    configured_export = options["export_config"]
    if args.command == "export":
        output = args.output.resolve()
        if args.input.is_dir():
            batch = MusicBatch(
                args.input,
                embedders=embedders,
                extractors=extractors,
                tagging_config=tagging,
                export_config=configured_export,
                recursive=options["recursive"],
                extensions=options["extensions"],
                tempo_model_path=options["tempo_model_path"],
            )
            batch.export(output)
            return 1 if batch._failures else 0
        export_config = _export_config(configured_export, output)
        process = MusicProcess(
            audio_file=args.input,
            embedders=embedders,
            extractors=extractors,
            tagging_config=tagging,
            export_config=export_config,
            tempo_model_path=options["tempo_model_path"],
        )
        process.process_file()
        return 0

    if args.command == "analyze":
        if args.input.is_dir():
            batch = MusicBatch(args.input, embedders=embedders, extractors=extractors)
            result = batch.analyze(args.key)
        else:
            process = MusicProcess(audio_file=args.input, embedders=embedders, extractors=extractors)
            result = process.analyze(args.key if args.key == "analyze" else [args.key])
        text = result.to_json(orient="records", indent=2) if hasattr(result, "to_json") else json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    if args.command == "preview":
        config = _export_config(configured_export, configured_export.output_root)
        if args.input.is_dir():
            batch = MusicBatch(args.input, embedders=embedders, extractors=extractors, tagging_config=tagging, export_config=config)
            for path in batch.preview_paths(args.extension):
                print(path)
        else:
            process = MusicProcess(audio_file=args.input, embedders=embedders, extractors=extractors, tagging_config=tagging, export_config=config)
            print(process.preview_path(args.extension))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

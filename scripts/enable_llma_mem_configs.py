#!/usr/bin/env python3
"""Enable memory.llma_mem across selected MARBLE YAML configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import yaml


def find_yaml_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix in {".yaml", ".yml"}:
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob("*.yaml")))
            files.extend(sorted(p.rglob("*.yml")))
    # de-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def update_config_dict(
    data: dict,
    topology: str,
    consolidation_interval: int,
    auto_embedding: bool,
    embedding_model: str,
) -> bool:
    memory = data.setdefault("memory", {})
    if not isinstance(memory, dict):
        return False

    llma_mem = memory.setdefault("llma_mem", {})
    if not isinstance(llma_mem, dict):
        return False

    before = dict(llma_mem)
    llma_mem["enabled"] = True
    llma_mem["topology"] = topology
    llma_mem["consolidation_interval"] = consolidation_interval
    llma_mem["auto_embedding"] = auto_embedding
    llma_mem["embedding_model"] = embedding_model

    return llma_mem != before


def update_one_config(
    path: Path,
    topology: str,
    consolidation_interval: int,
    auto_embedding: bool,
    embedding_model: str,
) -> tuple[bool, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return False, {}
    changed = update_config_dict(
        data=data,
        topology=topology,
        consolidation_interval=consolidation_interval,
        auto_embedding=auto_embedding,
        embedding_model=embedding_model,
    )
    return changed, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-enable memory.llma_mem for selected MARBLE configs."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["marble/configs"],
        help="Files or directories to scan for YAML configs.",
    )
    parser.add_argument(
        "--topology",
        default="hybrid",
        choices=["local", "shared", "hybrid"],
        help="LLMA-Mem topology value.",
    )
    parser.add_argument(
        "--consolidation-interval",
        type=int,
        default=10,
        help="Consolidation trigger interval.",
    )
    parser.add_argument(
        "--auto-embedding",
        action="store_true",
        default=True,
        help="Enable automatic embedding generation (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-embedding",
        action="store_true",
        help="Disable automatic embedding generation.",
    )
    parser.add_argument(
        "--embedding-model",
        default="amazon.titan-embed-text-v2:0",
        help="Embedding model ID.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    auto_embedding = args.auto_embedding and not args.no_auto_embedding
    files = find_yaml_files(args.paths)

    changed_files: List[Path] = []
    for path in files:
        try:
            changed, data = update_one_config(
                path=path,
                topology=args.topology,
                consolidation_interval=args.consolidation_interval,
                auto_embedding=auto_embedding,
                embedding_model=args.embedding_model,
            )
            if changed:
                changed_files.append(path)
                if args.apply:
                    with path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f"[skip] {path}: {e}")

    print(f"scanned={len(files)} changed={len(changed_files)} apply={args.apply}")
    for p in changed_files[:100]:
        print(p)
    if len(changed_files) > 100:
        print(f"... ({len(changed_files) - 100} more)")


if __name__ == "__main__":
    main()

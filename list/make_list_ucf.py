#!/usr/bin/env python3
"""Generate a UCF-Crime CLIP-feature manifest.

The feature root is supplied at runtime so this script never bakes a
machine-specific path into the public source tree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_indices(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative feature indices."""
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("indices must contain non-negative integers")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True,
                        help="directory containing the class subdirectories")
    parser.add_argument("--split", type=Path, default=Path("list/Anomaly_Train.txt"),
                        help="UCF split file")
    parser.add_argument("--output", type=Path, default=Path("list/ucf_CLIP_rgb.csv"),
                        help="output CSV path")
    parser.add_argument("--indices", type=parse_indices, default=tuple(range(10)),
                        help="feature suffixes, e.g. 0,1,...,9 or 5 for test features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[tuple[str, str]] = []
    missing = 0

    for raw_line in args.split.read_text(encoding="utf-8").splitlines():
        relative_video = raw_line.strip()
        if not relative_video:
            continue
        label = relative_video.split("/", maxsplit=1)[0]
        video_stem = (args.feature_root / Path(relative_video)).with_suffix("")
        for index in args.indices:
            feature_path = video_stem.parent / f"{video_stem.name}__{index}.npy"
            if feature_path.is_file():
                rows.append((str(feature_path), label))
            else:
                missing += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label"))
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}; missing features: {missing}")


if __name__ == "__main__":
    main()

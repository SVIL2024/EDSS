#!/usr/bin/env python3
"""Generate an XD-Violence CLIP-feature manifest.

The feature root is supplied at runtime so this script never bakes a
machine-specific path into the public source tree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True,
                        help="directory containing the extracted .npy features")
    parser.add_argument("--output", type=Path, default=Path("list/xd_CLIP_rgb.csv"),
                        help="output CSV path")
    parser.add_argument("--indices", nargs="+", type=int,
                        help="feature suffixes; use 0 for test features (default: all)")
    return parser.parse_args()


def label_from_name(name: str) -> str:
    """Return the XD label encoded between ``_label_`` and ``__``."""
    if "_label_A" in name:
        return "A"
    try:
        return name.split("_label_", maxsplit=1)[1].split("__", maxsplit=1)[0]
    except IndexError as exc:
        raise ValueError(f"feature filename has no XD label: {name}") from exc


def main() -> None:
    args = parse_args()
    paths = sorted(args.feature_root.glob("*.npy"),
                   key=lambda path: (label_from_name(path.name) == "A", path.name))
    if args.indices is not None:
        suffixes = tuple(f"__{index}.npy" for index in args.indices)
        paths = [path for path in paths if path.name.endswith(suffixes)]
    rows = [(str(path), label_from_name(path.name)) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

"""Reset auto-only imported review records to quarantine.

The script writes a ``<file>.bak`` backup before rewriting the JSONL file in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file to reset in place")
    parser.add_argument("--yes", action="store_true", help="Rewrite the file without an interactive confirmation")
    return parser.parse_args()


def _reset_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        # Reset stale acceptance so build-semantic-dataset auto-only exports only
        # records accepted by the current label-v2/semantic-v3 pass.
        row["status"] = "quarantine"

        auto = row.get("auto_metadata")
        if isinstance(auto, dict):
            auto.pop("label_v2", None)
            auto.pop("semantic_v3", None)

        rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    path = args.path
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")

    if not args.yes:
        response = input(f"Rewrite {path} after creating {path}.bak? Type 'yes' to continue: ")
        if response != "yes":
            raise SystemExit("aborted")

    rows = _reset_rows(path)
    backup_path = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup_path)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"reset {len(rows)} records in {path}; backup written to {backup_path}")


if __name__ == "__main__":
    main()

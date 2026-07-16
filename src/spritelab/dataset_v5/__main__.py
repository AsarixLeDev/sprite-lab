"""Module entry point for historical and raw Dataset-v5 commands."""

from __future__ import annotations

import sys

from spritelab.dataset_v5.raw_cli import RAW_COMMANDS


def main() -> int:
    if sys.argv[1:2] and sys.argv[1] in RAW_COMMANDS:
        from spritelab.dataset_v5.raw_cli import main as raw_main

        return raw_main(sys.argv[1:])
    from spritelab.dataset_v5.cli import main as historical_main

    return historical_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

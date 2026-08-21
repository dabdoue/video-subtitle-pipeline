from __future__ import annotations

import sys
from typing import Sequence

from .pipeline import PipelineError, execute, parse_args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

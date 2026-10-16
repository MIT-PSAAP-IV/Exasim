"""Verify this case through the shared geometry-aware comparator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verify_frontends import verify_case  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(1 if verify_case(Path(__file__).resolve().parent) else 0)

#!/usr/bin/env python3
"""Repository convenience entry point. Installed users can run `indicdlp`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from indicdlp.cli import entrypoint, main, parser_for_cli  # noqa: F401,E402


if __name__ == "__main__":
    entrypoint()

"""Verify environment-based W&B authentication without creating a training run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicdlp.environment import load_environment
from indicdlp.tracking.wandb import authenticate_online


def main():
    load_environment()
    import wandb

    authenticate_online(wandb)
    print(f"W&B authentication OK (SDK {wandb.__version__}); no run created.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

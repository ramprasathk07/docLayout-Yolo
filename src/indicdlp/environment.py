"""Load local credentials without overriding exported server settings."""

from pathlib import Path

from dotenv import load_dotenv


def load_environment():
    # Prefer the directory where the command runs. Also support repository scripts
    # invoked from elsewhere. Never search arbitrary ancestor directories.
    cwd_file = Path.cwd() / ".env"
    repository = Path(__file__).resolve().parents[2]
    candidates = [cwd_file]
    if (repository / "train_indicdlp.py").is_file():
        candidates.append(repository / ".env")
    for path in dict.fromkeys(candidates):
        load_dotenv(path, override=False, interpolate=False)

"""Tests never contact HF or download model/dataset files."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not open network connections")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)


@pytest.fixture
def fold_dataset(tmp_path):
    from tests.helpers import make_prepared as prepared
    from indicdlp.data.folds import make_folds

    root = tmp_path / "data"
    prepared(root)
    output = tmp_path / "folds"
    make_folds(root, output, 3)
    return root, output

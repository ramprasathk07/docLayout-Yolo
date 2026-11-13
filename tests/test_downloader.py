import importlib.util
from pathlib import Path


def test_downloader_only_requests_the_eight_domains(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/download_selected_indicdlp.py"
    spec = importlib.util.spec_from_file_location("download_selected", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "main", lambda argv: called.append(argv))
    module.download(["--max-train", "4", "--max-transfer-mib", "32"])
    assert module.TYPES == [
        "acts-and-rules",
        "brochures",
        "forms",
        "newspapers",
        "manuals",
        "notices",
        "novels",
        "syllabi",
    ]
    assert called[0][0] == "prepare" and called[0][-4:] == [
        "--max-train",
        "4",
        "--max-transfer-mib",
        "32",
    ]
    assert "snapshot_download" not in path.read_text()

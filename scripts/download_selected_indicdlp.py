"""Stream only the eight requested IndicDLP document domains into YOLO files.

HF_TOKEN is read by huggingface_hub from the environment. All prepare arguments
can be appended, including schema overrides and transfer/sample limits.
"""

import sys
import argparse
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_indicdlp import main


CONFIG = Path(__file__).resolve().parents[1] / "configs/data/selected.yaml"
TYPES = yaml.safe_load(CONFIG.read_text())["types"]


def download(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection-config", default=str(CONFIG))
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    config = yaml.safe_load(Path(args.selection_config).read_text(encoding="utf-8"))
    allowed = {
        "repo",
        "revision",
        "output",
        "max_train",
        "max_val",
        "max_test",
        "max_transfer_mib",
        "max_group_mib",
        "types",
    }
    if not isinstance(config, dict) or set(config) - allowed or not config.get("types"):
        raise ValueError("Invalid selection config")
    options = ["prepare", "--types", *config["types"]]
    for key, value in config.items():
        if key != "types":
            options += ["--" + key.replace("_", "-"), str(value)]
    return main([*options, *remaining])


if __name__ == "__main__":
    download()

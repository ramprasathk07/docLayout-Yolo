import json
import sys
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

from indicdlp import environment
from indicdlp.engine.training import load_model, run_train
from indicdlp.tracking.wandb import Tracking
from tests.helpers import fake_wandb_sdk
from train_indicdlp import parser_for_cli


@pytest.fixture
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        environment, "__file__", str(tmp_path / "src/indicdlp/environment.py")
    )
    monkeypatch.setattr(environment, "load_dotenv", load_dotenv)
    for key in (
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_BASE_URL",
        "HF_TOKEN",
    ):
        # Register an undo even when the variable was initially absent; dotenv
        # writes directly to os.environ rather than through monkeypatch.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_dotenv_loaded_before_cli_defaults_without_interpolating_secrets(
    isolated_environment,
):
    (isolated_environment / ".env").write_text(
        'WANDB_API_KEY="synthetic-${HOME}-key"\nWANDB_PROJECT=local-project\n'
        "WANDB_ENTITY=local-team\nHF_TOKEN=synthetic-hf\n"
    )
    args = parser_for_cli().parse_args(["train"])
    import os

    assert os.environ["WANDB_API_KEY"] == "synthetic-${HOME}-key"
    assert os.environ["HF_TOKEN"] == "synthetic-hf"
    assert args.wandb_project == "local-project" and args.wandb_entity == "local-team"


def test_exported_settings_override_dotenv(monkeypatch, isolated_environment):
    (isolated_environment / ".env").write_text(
        "WANDB_API_KEY=file-key\nWANDB_BASE_URL=https://file.invalid\n"
    )
    monkeypatch.setenv("WANDB_API_KEY", "exported-key")
    monkeypatch.setenv("WANDB_BASE_URL", "https://exported.invalid")
    environment.load_environment()
    import os

    assert os.environ["WANDB_API_KEY"] == "exported-key"
    assert os.environ["WANDB_BASE_URL"] == "https://exported.invalid"


def test_repository_env_supported_from_another_working_directory(
    monkeypatch, isolated_environment
):
    (isolated_environment / "train_indicdlp.py").touch()
    (isolated_environment / ".env").write_text("WANDB_PROJECT=repository-project\n")
    elsewhere = isolated_environment / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert parser_for_cli().parse_args(["train"]).wandb_project == "repository-project"


def test_missing_online_key_fails_without_sdk_init(monkeypatch, isolated_environment):
    calls, _ = fake_wandb_sdk(monkeypatch)
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        Tracking(args, {}).start()
    assert calls == []


def test_online_uses_dotenv_key_without_putting_it_in_run_config(
    monkeypatch, isolated_environment
):
    secret = "wandb_v1_" + "x" * 77  # New-style 86-character fixture, not a real key.
    (isolated_environment / ".env").write_text(
        f"WANDB_API_KEY={secret}\nWANDB_BASE_URL=https://api.forge.coreweave.com\n"
    )
    calls, _ = fake_wandb_sdk(monkeypatch)
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    tracker = Tracking(args, {"train": {"batch": 16}}).start()
    assert calls[0]["mode"] == "online" and secret not in json.dumps(calls)
    assert sys.modules["wandb"].login_calls == [
        {
            "key": secret,
            "prompt": False,
            "verify": True,
            "host": "https://api.forge.coreweave.com",
        }
    ]
    tracker.finish()


def test_offline_requires_no_key(monkeypatch, isolated_environment):
    calls, _ = fake_wandb_sdk(monkeypatch)
    args = parser_for_cli().parse_args(["train", "--wandb", "offline"])
    tracker = Tracking(args, {}).start()
    assert calls[0]["mode"] == "offline"
    assert sys.modules["wandb"].login_calls == []
    tracker.finish()


def test_model_failure_precedes_wandb_run(monkeypatch, tmp_path):
    calls, _ = fake_wandb_sdk(monkeypatch)
    monkeypatch.setattr(
        "indicdlp.engine.training.verify_dataset", lambda _: tmp_path / "data.yaml"
    )

    def incompatible(*args):
        raise RuntimeError("unsupported torch fixture")

    monkeypatch.setattr("indicdlp.engine.training.load_model", incompatible)
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    with pytest.raises(RuntimeError, match="unsupported torch"):
        run_train(args)
    assert calls == []


def test_new_torch_error_identifies_active_interpreter(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__="2.15.0+cu132")
    )
    with pytest.raises(RuntimeError, match="Unsupported torch 2.15.0") as error:
        load_model("does-not-exist.pt", "cpu")
    assert sys.executable in str(error.value) and "torch==2.5.1" in str(error.value)


def test_old_sdk_is_rejected_before_login_and_init(monkeypatch, isolated_environment):
    calls, _ = fake_wandb_sdk(monkeypatch)
    sys.modules["wandb"].__version__ = "0.19.11"
    monkeypatch.setenv("WANDB_API_KEY", "wandb_v1_" + "x" * 77)
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    with pytest.raises(RuntimeError, match="pip install --upgrade"):
        Tracking(args, {}).start()
    assert calls == [] and sys.modules["wandb"].login_calls == []


def test_failed_login_never_prompts_or_creates_run_or_echoes_key(
    monkeypatch, isolated_environment
):
    calls, _ = fake_wandb_sdk(monkeypatch)
    key = "synthetic-secret-do-not-echo"
    monkeypatch.setenv("WANDB_API_KEY", key)

    def reject(**kwargs):
        assert (
            kwargs["key"] == key
            and kwargs["prompt"] is False
            and kwargs["verify"] is True
        )
        raise ValueError(f"simulated SDK failure containing {key}")

    sys.modules["wandb"].login = reject
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        Tracking(args, {}).start()
    assert key not in str(error.value) and calls == []


def test_unsuccessful_login_is_not_followed_by_init(monkeypatch, isolated_environment):
    calls, _ = fake_wandb_sdk(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-key")
    sys.modules["wandb"].login = lambda **kwargs: False
    args = parser_for_cli().parse_args(["train", "--wandb", "online"])
    with pytest.raises(RuntimeError, match="did not succeed"):
        Tracking(args, {}).start()
    assert calls == []


def test_auth_check_uses_env_without_creating_run(
    monkeypatch, isolated_environment, capsys
):
    from scripts.check_wandb import main

    key = "wandb_v1_" + "x" * 77
    (isolated_environment / ".env").write_text(f"WANDB_API_KEY={key}\n")
    calls, _ = fake_wandb_sdk(monkeypatch)
    main()
    output = capsys.readouterr().out
    assert "authentication OK" in output and key not in output
    assert sys.modules["wandb"].login_calls[0]["prompt"] is False
    assert calls == []

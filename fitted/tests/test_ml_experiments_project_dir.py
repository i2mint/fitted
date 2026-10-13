"""``project_dir`` is honoured, and importing touches no disk (fitted#4)."""

import os
import subprocess
import sys
from pathlib import Path

import fitted
from fitted.ml_experiments import (
    ExperimentConfig,
    ExperimentRunner,
    load_experiment_results,
    uri_to_project_store,
)


def test_runner_uses_project_dir_for_a_named_store(tmp_path):
    runner = ExperimentRunner(
        project_store="proj",
        project_dir=str(tmp_path),
        config=ExperimentConfig(verbose=False),
    )
    runner.project_store["k"] = 1
    assert (tmp_path / "proj" / "k").exists()
    assert load_experiment_results("proj", projects_dir=str(tmp_path))["k"] == 1


def test_uri_to_project_store_with_an_absolute_name_needs_no_default(
    tmp_path, monkeypatch
):
    # an unusable default location must not matter when the name is a full path
    monkeypatch.setenv("HOME", str(tmp_path / "no" / "such"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    store = uri_to_project_store(str(tmp_path / "abs"))
    store["k"] = 1
    assert (tmp_path / "abs" / "k").exists()


def test_import_survives_an_unwritable_home(tmp_path):
    blocker = tmp_path / "home_is_a_file"
    blocker.write_text("")  # the app folder can't be created below a file
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("FITTED_DFLT_EXP_PROJECTS_DIR", "XDG_CONFIG_HOME")
    }
    env["HOME"] = str(blocker)
    env["PYTHONPATH"] = str(Path(fitted.__file__).parent.parent)
    out = subprocess.run(
        [sys.executable, "-c", "import fitted.ml_experiments"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-500:]

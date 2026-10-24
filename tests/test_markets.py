"""C034: market configuration (developed = Part 1 folder, emerging = Part 2 emerging folder)."""
import os
import subprocess
import sys

import pytest

from aiiaa import markets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PART1 = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-1")
EMERGING = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-2", "emerging")


def test_default_market_is_developed_part1_folder(monkeypatch):
    monkeypatch.delenv(markets.ENV, raising=False)
    m = markets.get()
    assert m.name == "developed"
    assert os.path.abspath(m.data_dir) == PART1
    assert os.path.abspath(m.out_dir) == os.path.join(ROOT, "output")
    assert m.years == (2021, 2022, 2023, 2024)


def test_env_var_selects_emerging_folder_and_output_subfolder(monkeypatch):
    monkeypatch.setenv(markets.ENV, "emerging")
    m = markets.get()
    assert m.name == "emerging"
    assert os.path.abspath(m.data_dir) == EMERGING
    assert os.path.abspath(m.out_dir) == os.path.join(ROOT, "output", "emerging")
    assert m.years == (2023, 2024)


def test_explicit_name_overrides_env_var(monkeypatch):
    monkeypatch.setenv(markets.ENV, "emerging")
    assert markets.get("developed").name == "developed"


def test_unknown_market_raises(monkeypatch):
    monkeypatch.setenv(markets.ENV, "mars")
    with pytest.raises(ValueError, match="mars"):
        markets.get()


@pytest.mark.parametrize("argv", [["s.py", "motor", "--market", "emerging"], ["s.py", "--market=emerging", "motor"]])
def test_market_option_in_argv_sets_env_var(monkeypatch, argv):
    monkeypatch.delenv(markets.ENV, raising=False)
    markets.select_from_argv(argv)
    assert os.environ[markets.ENV] == "emerging"


def test_argv_without_market_option_leaves_env_unchanged(monkeypatch):
    monkeypatch.delenv(markets.ENV, raising=False)
    markets.select_from_argv(["s.py", "motor"])
    assert markets.ENV not in os.environ


def test_emerging_trend_options():
    trends = markets.get("emerging").trends
    assert trends["none"] == {"motor": 1.0, "buildings": 1.0, "contents": 1.0}
    assert trends["developed-half"] == {"motor": 1.0175, "buildings": 1.0948, "contents": 1.0636}
    assert markets.get("developed").trend_share == 0.5


def _script_data_dir(env_value):
    env = {k: v for k, v in os.environ.items() if k != markets.ENV}
    if env_value is not None:
        env[markets.ENV] = env_value
    code = "import sys; sys.path[:0] = [r'%s', r'%s']; import train_validate as t; print(t.DATA); print(t.OUT)" % (
        os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, check=True)
    return [os.path.abspath(p) for p in out.stdout.split()]


def test_train_validate_resolves_part1_folder_by_default():
    assert _script_data_dir(None) == [PART1, os.path.join(ROOT, "output")]


def test_train_validate_resolves_emerging_folder_from_env_var():
    assert _script_data_dir("emerging") == [EMERGING, os.path.join(ROOT, "output", "emerging")]

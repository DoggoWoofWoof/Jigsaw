"""Protocol settings as paper numbers (protocol_numbers.py)."""

from __future__ import annotations

import inspect
import json

import pytest

from scripts.canonical import build_paper, numbers_json, protocol, protocol_numbers, stats

FAKE = {
    "TIME": {"primary_T_seconds": 30.0, "run_T_max_seconds": {"E8": [30.0, 600.0]}, "step_cap_seconds": 0.25},
    "QUERY_SIZES": [20, 50, 100],
    "DATASETS": {"mag": {"partitions": 2000, "ladder": [20, 1000], "encoder": "rgcn"}},
    "SERVING": {"memory_caps_gb": [2, 1.5], "scale_out": {"host_memory_gb_min": 256}},
    "SEEDS": {"training": [7301], "validation": ["31415"], "test": "sealed", "partition": 1},
    "STATISTICS": {"bootstrap_seed": 0, "noninferiority_margin_pp": 1.0, "alpha": 0.05,
                   "paired_test": "sign_flip_randomization_100k", "rate_interval": "wilson_95"},
    "TRAINING": {"lr": 5e-05, "residual": True, "arch": "rgcn"},
    "PARTITIONING": {"scale": [0.5], "metis_x0.5": {"blocks": 3}},
    "RANKERS": {"router_tau": {"mag": 1.8061799739838871}, "component_cap": None},
    "DEV_TUNABLE": {"TIME.step_cap_seconds": [0.5, 1.0]},
}
FAKE_TEXT = {
    "protocol.TIME.primary_T_seconds": r"30\,s",
    "protocol.TIME.run_T_max_seconds.E8.0": r"30\,s",
    "protocol.TIME.run_T_max_seconds.E8.1": r"600\,s",
    "protocol.TIME.step_cap_seconds": r"0.25\,s",
    "protocol.QUERY_SIZES.0": "20",
    "protocol.QUERY_SIZES.1": "50",
    "protocol.QUERY_SIZES.2": "100",
    "protocol.DATASETS.mag.partitions": "2{,}000",
    "protocol.DATASETS.mag.ladder.0": "20",
    "protocol.DATASETS.mag.ladder.1": "1{,}000",
    "protocol.SERVING.memory_caps_gb.0": r"2\,GiB",
    "protocol.SERVING.memory_caps_gb.1": r"1.5\,GiB",
    "protocol.SERVING.scale_out.host_memory_gb_min": r"256\,GiB",
    "protocol.SEEDS.training.0": "7301",
    "protocol.SEEDS.validation.0": "31415",
    "protocol.SEEDS.partition": "1",
    "protocol.STATISTICS.bootstrap_seed": "0",
    "protocol.STATISTICS.noninferiority_margin_pp": r"1\,pp",
    "protocol.STATISTICS.alpha": "0.05",
    "protocol.STATISTICS.paired_test.resamples": "100{,}000",
    "protocol.TRAINING.lr": "0.00005",
    "protocol.PARTITIONING.scale.0": "0.5",
    "protocol.PARTITIONING.metis_x0p5.blocks": "3",
    "protocol.RANKERS.router_tau.mag": "1.806",
    "protocol.DEV_TUNABLE.TIME_step_cap_seconds.0": r"0.5\,s",
    "protocol.DEV_TUNABLE.TIME_step_cap_seconds.1": r"1\,s",
}


def _text(nums, key):
    entry = nums.numbers[key]
    return build_paper.format_value(entry["value"], entry["unit"], entry.get("digits"), key=key)


def _numeric_leaves(value, path=()):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _numeric_leaves(item, path + (protocol_numbers.segment(name),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, path + (str(index),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield "protocol." + ".".join(path), value


def test_every_numeric_setting_of_the_protocol_is_a_key():
    nums = protocol_numbers.protocol_numbers()
    expected = dict(_numeric_leaves(protocol.protocol_dict()))
    assert expected and set(expected) <= set(nums.numbers)
    for key, value in expected.items():
        entry = nums.numbers[key]
        want = value * 2 ** 30 if entry["unit"] == "bytes" else str(value) if entry["unit"] == "text" else value
        assert entry["value"] == want, key
    digit_seeds = {f"protocol.SEEDS.{split}.{i}" for split in ("dev", "validation")
                   for i, s in enumerate(protocol.SEEDS[split]) if isinstance(s, str)}
    derived = {"protocol.STATISTICS.paired_test.resamples", "protocol.STATISTICS.paired_interval.resamples",
               "protocol.hash", "protocol.hash_short"}
    assert set(nums.numbers) - set(expected) == digit_seeds | derived
    assert nums.numbers["protocol.hash"]["value"] == protocol.protocol_hash()
    assert _text(nums, "protocol.TIME.primary_T_seconds") == rf"{protocol.TIME['primary_T_seconds']:g}\,s"


def test_resample_counts_and_statistics_settings_are_what_stats_uses():
    nums = protocol_numbers.protocol_numbers()

    def default(fn, name):
        return inspect.signature(fn).parameters[name].default

    perm = nums.numbers["protocol.STATISTICS.paired_test.resamples"]["value"]
    boot = nums.numbers["protocol.STATISTICS.paired_interval.resamples"]["value"]
    assert perm == default(stats.sign_flip_pvalue, "n_perm") == default(stats.paired_compare, "n_perm")
    assert perm == default(stats.paired_log_ratio, "n_perm")
    assert boot == default(stats.paired_bootstrap_ci, "n_boot") == default(stats.paired_compare, "n_boot")
    assert boot == default(stats.paired_log_ratio, "n_boot")
    assert protocol.STATISTICS["alpha"] == default(stats.paired_bootstrap_ci, "alpha")
    assert protocol.STATISTICS["bootstrap_seed"] == default(stats.paired_compare, "seed")


def test_units_names_and_paper_text(monkeypatch, capsys):
    monkeypatch.setattr(protocol, "protocol_dict", lambda: json.loads(json.dumps(FAKE)))
    nums = protocol_numbers.protocol_numbers()
    assert set(nums.numbers) == set(FAKE_TEXT) | {"protocol.hash", "protocol.hash_short"}
    assert {key: _text(nums, key) for key in FAKE_TEXT} == FAKE_TEXT
    assert "rounded" in nums.numbers["protocol.RANKERS.router_tau.mag"]["note"]
    assert nums.numbers["protocol.SERVING.memory_caps_gb.0"]["value"] == 2 * 2 ** 30
    assert nums.numbers["protocol.hash_short"]["value"] == protocol.protocol_hash()[:12]
    assert protocol_numbers.main(["--list"]) == 0
    listed = dict(line.split("\t") for line in capsys.readouterr().out.splitlines())
    assert listed["protocol.TIME.primary_T_seconds"] == r"30\,s" and len(listed) == len(nums.numbers)


def test_build_paper_merges_them_and_builds_camera_ready_after_the_freeze(tmp_path, monkeypatch, capsys):
    out = tmp_path / "protocol"
    assert protocol_numbers.main(["--out", str(out)]) == 0
    data = numbers_json.load(out / "numbers.json")
    assert data["split"] == "test" and data["frozen"] is protocol.is_frozen()
    assert data["protocol_hash"] == protocol.protocol_hash() and data["sources"] == {}
    other = numbers_json.Numbers(split="test")
    other.number("E1.solved", 95.4, "pct")
    other.write(tmp_path / "main")
    gen = tmp_path / "gen"
    assert build_paper.main(["--numbers", str(out / "numbers.json"), str(tmp_path / "main" / "numbers.json"),
                             "--out", str(gen), "--allow-dev"]) == 0
    tex = (gen / "numbers.tex").read_text(encoding="utf-8")
    assert "jnum@protocol.TIME.primary_T_seconds" in tex and "jnum@E1.solved" in tex
    if not protocol.is_frozen():
        assert build_paper.main(["--numbers", str(out / "numbers.json"), "--out", str(tmp_path / "draft")]) == 2
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", protocol.protocol_hash())
    assert protocol_numbers.main(["--out", str(tmp_path / "frozen")]) == 0
    assert build_paper.main(["--numbers", str(tmp_path / "frozen" / "numbers.json"),
                             "--out", str(tmp_path / "ready")]) == 0
    assert json.loads((tmp_path / "ready" / "build.json").read_text(encoding="utf-8"))["mode"] == "frozen"
    capsys.readouterr()


@pytest.mark.parametrize("spec, match", [
    ({"A": {"x.y": 1, "x_y": 2}}, "added twice"),
    ({"SEEDS": {"partition": 1.5}}, "a seed must be an integer"),
    ({"SERVING": {"memory_caps_gb": [1e-12]}}, "not a whole number of bytes"),
    ({"A": {"": 1}}, "empty key segment"),
])
def test_refusals(monkeypatch, spec, match):
    monkeypatch.setattr(protocol, "protocol_dict", lambda: spec)
    with pytest.raises(numbers_json.NumbersError, match=match):
        protocol_numbers.protocol_numbers()


def test_names_and_digits():
    assert protocol_numbers.segment("metis_x0.5") == "metis_x0p5"
    assert protocol_numbers.segment("TIME.step_cap_seconds") == "TIME_step_cap_seconds"
    assert protocol_numbers.segment("R2-512") == "R2-512" and protocol_numbers.segment(3) == "3"
    assert [protocol_numbers.exact_digits(v) for v in (30.0, 0.25, 5e-05, 5e-06, 7)] == [0, 2, 5, 6, 0]
    assert protocol_numbers.exact_digits(1.8061799739838871) is None
    with pytest.raises(SystemExit):
        protocol_numbers.main([])

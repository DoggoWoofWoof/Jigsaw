"""Tests for the sealed-seed guard (scripts/canonical/seal.py).

Every test works on files under tmp_path. None reads internal/sealed/SEALED_SEEDS.json, none calls
seal.load_test_masters, and authorize_sealed_run is exercised only on refusal paths with the unseal
log redirected to tmp_path.
"""

from __future__ import annotations

import json

import pytest

from scripts.canonical import manifest, protocol, seal


@pytest.fixture(autouse=True)
def _tmp_unseal_log(tmp_path, monkeypatch):
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    return log


def test_generate_writes_once(tmp_path):
    path = tmp_path / "sealed" / "SEEDS.json"
    sealed = seal.generate(path)
    assert set(sealed["masters"]) == set(seal.MASTER_NAMES)
    assert len({*sealed["masters"].values(), sealed["salt"]}) == 4  # independent draws
    assert all(len(v) == 64 for v in sealed["masters"].values())
    assert seal.read_sealed(path) == sealed
    with pytest.raises(seal.SealError, match="refusing to overwrite"):
        seal.generate(path)
    assert seal.read_sealed(path) == sealed


def test_read_sealed_refusals(tmp_path):
    with pytest.raises(seal.SealError, match="not found"):
        seal.read_sealed(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "salt": "00", "masters": {"test": "11"}}), encoding="utf-8")
    with pytest.raises(seal.SealError, match="unexpected set of masters"):
        seal.read_sealed(bad)


def test_commitment_binds_salt_and_masters_only(tmp_path):
    sealed = seal.generate(tmp_path / "s.json")
    value = seal.commitment_of(sealed)
    assert value == seal.commitment_of(json.loads(json.dumps(sealed)))
    assert seal.commitment_of({**sealed, "note": "edited", "created_utc": "later"}) == value
    for name in seal.MASTER_NAMES:
        changed = {**sealed, "masters": {**sealed["masters"], name: "0" * 64}}
        assert seal.commitment_of(changed) != value
    assert seal.commitment_of({**sealed, "salt": "0" * 64}) != value
    assert seal.commitment_of({**sealed, "version": 2}) != value


def test_write_commitment_sets_once_then_needs_force(tmp_path):
    proto = tmp_path / "protocol.py"
    proto.write_text('X = 1\nSEAL_COMMITMENT: str | None = None\nY = 2\n', encoding="utf-8")
    first, second = "a" * 64, "b" * 64
    seal.write_commitment(first, proto)
    assert f'SEAL_COMMITMENT: str | None = "{first}"' in proto.read_text(encoding="utf-8")
    seal.write_commitment(first, proto)  # the same value again is a no-op
    with pytest.raises(seal.SealError, match="different commitment"):
        seal.write_commitment(second, proto)
    seal.write_commitment(second, proto, force=True)
    text = proto.read_text(encoding="utf-8")
    assert f'"{second}"' in text and first not in text and text.startswith("X = 1\n") and text.endswith("Y = 2\n")
    missing = tmp_path / "other.py"
    missing.write_text("SEAL_COMMITMENT = None\n", encoding="utf-8")
    with pytest.raises(seal.SealError, match="could not find"):
        seal.write_commitment(first, missing)


def test_verify_checks_the_file_against_the_commitment(tmp_path, monkeypatch):
    path = tmp_path / "s.json"
    sealed = seal.generate(path)
    monkeypatch.setattr(protocol, "SEAL_COMMITMENT", None)
    with pytest.raises(seal.SealError, match="not set"):
        seal.verify(path)
    monkeypatch.setattr(protocol, "SEAL_COMMITMENT", "0" * 64)
    with pytest.raises(seal.SealError, match="do not match"):
        seal.verify(path)
    monkeypatch.setattr(protocol, "SEAL_COMMITMENT", seal.commitment_of(sealed))
    assert seal.verify(path) == seal.commitment_of(sealed)
    tampered = {**sealed, "masters": {**sealed["masters"], "test": "f" * 64}}
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(seal.SealError, match="do not match"):
        seal.verify(path)


def test_authorize_sealed_run_refuses_and_logs_nothing(monkeypatch, _tmp_unseal_log):
    with pytest.raises(seal.SealError, match="explicit --unseal"):
        seal.authorize_sealed_run(unseal=False, reason="t")
    monkeypatch.setattr(protocol, "is_frozen", lambda: False)
    with pytest.raises(seal.SealError, match="not frozen"):
        seal.authorize_sealed_run(unseal=True, reason="t")
    monkeypatch.setattr(protocol, "is_frozen", lambda: True)
    monkeypatch.setattr(manifest, "git_info", lambda: {"available": True, "dirty_code": True, "head": "x"})
    with pytest.raises(seal.SealError, match="differs from HEAD"):
        seal.authorize_sealed_run(unseal=True, reason="t")
    monkeypatch.setattr(manifest, "git_info", lambda: {"available": False})
    with pytest.raises(seal.SealError, match="differs from HEAD"):
        seal.authorize_sealed_run(unseal=True, reason="t")
    assert not _tmp_unseal_log.exists()


def test_split_masters_routes_only_test_splits_to_the_seal(monkeypatch):
    for split in ("dev", "validation"):
        assert seal.split_masters(split) == [str(s) for s in protocol.SEEDS[split]]
    calls = []

    def refuse(*, unseal, reason, **_):
        calls.append((unseal, reason))
        raise seal.SealError("stub: sealed")

    monkeypatch.setattr(seal, "load_test_masters", refuse)
    for split in seal.MASTER_NAMES:
        with pytest.raises(seal.SealError, match="stub: sealed"):
            seal.split_masters(split, unseal=True, reason="")
    assert calls == [(True, name) for name in seal.MASTER_NAMES]
    with pytest.raises(seal.SealError, match="unknown split"):
        seal.split_masters("train")


def test_reveal_needs_the_finished_flag(capsys):
    assert seal.main(["reveal"]) == 2
    assert "--i-have-finished-the-sealed-run" in capsys.readouterr().err

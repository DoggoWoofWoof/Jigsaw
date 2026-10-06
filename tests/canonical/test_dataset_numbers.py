"""Dataset statistics as paper numbers, read from prepared stores (dataset_numbers.py)."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

pytest.importorskip("pymetis")

from test_prepare_data import cora_root, write_mag_raw  # noqa: E402  (tests/canonical is on sys.path)

from scripts.canonical import build_paper, dataset_numbers, manifest, numbers_json  # noqa: E402
from scripts.canonical import prepare_data as prep  # noqa: E402


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dn")
    data_root, _ = cora_root(tmp / "c", n=120)
    root = tmp / "stores"
    cora = prep.prepare("corafull", data_root=data_root, store_root=root, partitions=4, fine_per_block=2,
                        alternatives=True, tool="pymetis")
    write_mag_raw(tmp / "m" / "data")
    mag = prep.prepare("mag", data_root=tmp / "m" / "data", store_root=root, partitions=3, fine_per_block=2,
                       tool="pymetis")
    return root, {"corafull": cora, "mag": mag}


def _text(data, key):
    entry = data["numbers"][key]
    return build_paper.format_value(entry["value"], entry["unit"], entry.get("digits"), key=key)


def test_the_keys_are_the_prepare_reports(stores, tmp_path, capsys):
    root, reports = stores
    assert dataset_numbers.main(["--store-root", str(root), "--out", str(tmp_path / "out")]) == 0
    err = capsys.readouterr().err
    assert "absent:" in err and "arxiv" in err and "2 stores" in err
    data = numbers_json.load(tmp_path / "out" / "numbers.json")
    assert data["split"] == "test" and set(data["sources"]) == {"data.corafull.prepare", "data.mag.prepare"}
    src = data["sources"]["data.mag.prepare"]
    run = root / "mag" / dataset_numbers.PREPARE_RUN
    assert src["experiment"] == "prepare_data" and src["dataset"] == "mag"
    assert src["manifest_sha256"] == hashlib.sha256((run / manifest.MANIFEST_NAME).read_bytes()).hexdigest()
    n = data["numbers"]
    for d, report in reports.items():
        assert n[f"data.{d}.nodes"]["value"] == report["nodes"]
        assert n[f"data.{d}.edges"]["value"] == report["undirected_edges"]
        assert n[f"data.{d}.types"]["value"] == len(report["type_names"])
        assert all(n[k]["source"] == [f"data.{d}.prepare"] for k in n if k.startswith(f"data.{d}."))
        for regime, s in report["sigma"].items():
            assert n[f"data.{d}.sigma.{regime}.codes"]["value"] == s["num_codes"]
            share = n[f"data.{d}.sigma.{regime}.singleton_share"]
            assert share["unit"] == "pct" and share["value"] == pytest.approx(100 * s["singleton_codes"]
                                                                              / report["nodes"])
    mag = reports["mag"]
    assert [n[f"data.mag.type.{t}.nodes"]["value"] for t in mag["type_names"]] == mag["type_counts"]
    assert n["data.mag.labelled_nodes"]["value"] == 60
    cora_parts = reports["corafull"]["partitions"]
    assert {"metis_x0.5", "metis_x2", "bfs_random"} <= set(cora_parts)
    assert n["data.corafull.partition.metis_x0p5.blocks"]["value"] == cora_parts["metis_x0.5"]["num_blocks"]
    cut = cora_parts["primary"]["cut_fraction"]
    assert n["data.corafull.partition.primary.cut_share"]["value"] == pytest.approx(100 * cut)
    for key in ("data.corafull.degree.median", "data.corafull.partition.primary.block_nodes_median"):
        entry = n[key]  # a whole-number median is a count, any other a number with one decimal
        assert entry["unit"] == "count" or (entry["unit"] == "number" and entry["digits"] == 1)
    table = data["tables"][dataset_numbers.TABLE]
    assert [row[0] for row in table["rows"]] == ["CoraFull", "ogbn-mag"]
    assert table["columns"][:6] == ["Dataset", "Nodes", "Edges", "Node types", "Attribute dim.", "Blocks"]
    assert table["rows"][1][1] == {"key": "data.mag.nodes"} and len(table["rows"][0]) == len(table["columns"])
    assert _text(data, "data.mag.nodes") == str(mag["nodes"])


def test_only_the_named_datasets_and_the_table_marks_absent_regimes(stores, tmp_path, monkeypatch, capsys):
    root, _ = stores
    assert dataset_numbers.main(["--store-root", str(root), "--datasets", "mag", "--out", str(tmp_path)]) == 0
    data = numbers_json.load(tmp_path / "numbers.json")
    assert {k.split(".")[1] for k in data["numbers"]} == {"mag"}
    capsys.readouterr()
    real = dataset_numbers.read_store

    def without_r1(store):  # as for a dataset without class labels
        report, doc, sha = real(store)
        if store.name == "mag":
            report = {**report, "sigma": {k: v for k, v in report["sigma"].items() if k != "R1"}}
        return report, doc, sha

    monkeypatch.setattr(dataset_numbers, "read_store", without_r1)
    table = dataset_numbers.dataset_numbers([root / "mag", root / "corafull"]).tables[dataset_numbers.TABLE]
    col = table["columns"].index("R1 labels")
    assert [row[0] for row in table["rows"]] == ["CoraFull", "ogbn-mag"]
    assert table["rows"][0][col] == {"key": "data.corafull.sigma.R1.codes"} and table["rows"][1][col] == "--"


def test_refusals(stores, tmp_path, capsys):
    root, _ = stores
    copy = tmp_path / "stores"
    shutil.copytree(root / "corafull", copy / "corafull")
    assert dataset_numbers.main(["--store-root", str(copy), "--datasets", "mag", "--out", str(tmp_path / "o")]) == 2
    assert "no prepare run" in capsys.readouterr().err
    report = copy / "corafull" / dataset_numbers.REPORT
    report.write_text(report.read_text(encoding="utf-8").replace('"nodes": ', '"nodes": 1'), encoding="utf-8")
    with pytest.raises(dataset_numbers.DatasetNumbersError, match="not the report its prepare run recorded"):
        dataset_numbers.read_store(copy / "corafull")
    shutil.copytree(root / "corafull", copy / "other")
    with pytest.raises(dataset_numbers.DatasetNumbersError, match="in the store 'other'"):
        dataset_numbers.read_store(copy / "other")
    marker = copy / "other" / dataset_numbers.PREPARE_RUN / "stages" / "load.json"
    marker.write_text(marker.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(dataset_numbers.DatasetNumbersError, match="differs from its manifest"):
        dataset_numbers.read_store(copy / "other")
    with pytest.raises(dataset_numbers.DatasetNumbersError, match="two stores of one dataset"):
        dataset_numbers.dataset_numbers([root / "mag", root / "mag"])
    assert dataset_numbers.main(["--store-root", str(tmp_path / "empty"), "--out", str(tmp_path / "o")]) == 2
    assert "no store with a prepare run" in capsys.readouterr().err
    doc = json.loads((root / "mag" / dataset_numbers.PREPARE_RUN / manifest.MANIFEST_NAME).read_text("utf-8"))
    assert doc["status"] == "complete"

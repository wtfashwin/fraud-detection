"""Check only the safe notebook's data-loading cell with synthetic CSVs."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import zipfile

import pandas as pd
import pytest

from training.preprocess import validate_dataset


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/feature_engineering_verified.ipynb"


def fixture_csv(first_time=1):
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    first = [str(first_time), "0.12345678901234568", *("0" for _ in range(27)), "10", "0"]
    second = [str(first_time + 1), *("1" for _ in range(28)), "20", "1"]
    return (",".join(columns) + "\n" + ",".join(first) + "\n" + ",".join(second) + "\n").encode()


def write_source(tmp_path, content, suffix=".csv"):
    source = tmp_path / f"input{suffix}"
    source.write_bytes(content)
    return source


def sidecar_for(source, content, **fields):
    metadata = {"csv_sha256": hashlib.sha256(content).hexdigest(), **fields}
    source.with_suffix(".metadata.json").write_text(json.dumps(metadata))


def load_cell(source, tmp_path, monkeypatch, *, kind=None):
    monkeypatch.setenv("DATA_CSV", str(source))
    if kind is None:
        monkeypatch.delenv("DATA_KIND", raising=False)
    else:
        monkeypatch.setenv("DATA_KIND", kind)
    notebook = json.loads(NOTEBOOK.read_text())
    code = next(cell for cell in notebook["cells"] if cell.get("id") == "de8cbc6c")
    scope = {
        "__name__": "__notebook_load_cell_test__", "hashlib": hashlib,
        "json": json, "os": os, "Path": Path, "pd": pd,
        "project_root": tmp_path,
        "validate_dataset": validate_dataset,
    }
    exec(compile("".join(code["source"]), "<verified notebook load cell>", "exec"), scope)
    return scope


def test_new_notebook_contains_no_saved_results_or_execution_metadata():
    notebook = json.loads(NOTEBOOK.read_text())
    assert "version" not in notebook.get("metadata", {}).get("language_info", {})
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            assert "execution" not in cell.get("metadata", {})


def test_matching_sidecar_and_data_kind_override_preserve_default_parser(tmp_path, monkeypatch):
    content = fixture_csv()
    source = write_source(tmp_path, content)
    sidecar_for(source, content, data_kind="synthetic", source_url="synthetic fixture")
    expected_frame = pd.read_csv(source)

    result = load_cell(source, tmp_path, monkeypatch, kind="real")

    pd.testing.assert_frame_equal(result["frame"], expected_frame)
    assert result["dataset_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["data_kind"] == "real"


@pytest.mark.parametrize("kind,expected", [(None, "unspecified"), ("synthetic", "synthetic")])
def test_missing_sidecar_uses_existing_data_kind_precedence(tmp_path, monkeypatch, kind, expected):
    content = fixture_csv()
    source = write_source(tmp_path, content)
    result = load_cell(source, tmp_path, monkeypatch, kind=kind)
    assert result["data_kind"] == expected
    assert result["dataset_sha256"] == hashlib.sha256(content).hexdigest()


def test_sidecar_without_hash_remains_supported(tmp_path, monkeypatch):
    source = write_source(tmp_path, fixture_csv())
    source.with_suffix(".metadata.json").write_text('{"data_kind":"synthetic"}')
    assert load_cell(source, tmp_path, monkeypatch)["data_kind"] == "synthetic"


def test_mismatched_sidecar_is_rejected_before_csv_parsing(tmp_path, monkeypatch):
    source = write_source(tmp_path, fixture_csv())
    source.with_suffix(".metadata.json").write_text(json.dumps({"csv_sha256": "0" * 64}))

    def parse_would_be_unsafe(*_args, **_kwargs):
        pytest.fail("CSV was parsed before its declared hash was checked")

    monkeypatch.setattr(pd, "read_csv", parse_would_be_unsafe)
    with pytest.raises(ValueError, match="csv_sha256|CSV hash"):
        load_cell(source, tmp_path, monkeypatch)


def test_replacement_after_parse_keeps_frame_and_hash_on_same_snapshot(tmp_path, monkeypatch):
    original = fixture_csv(first_time=1)
    replacement = fixture_csv(first_time=101)
    source = write_source(tmp_path, original)
    sidecar_for(source, original, data_kind="synthetic")
    real_read_csv = pd.read_csv

    def replace_after_read(*args, **kwargs):
        frame = real_read_csv(*args, **kwargs)
        source.write_bytes(replacement)
        return frame

    monkeypatch.setattr(pd, "read_csv", replace_after_read)
    result = load_cell(source, tmp_path, monkeypatch)
    assert source.read_bytes() == replacement
    assert result["frame"]["Time"].tolist() == [1, 2]
    assert result["dataset_sha256"] == hashlib.sha256(original).hexdigest()


def test_replacement_without_sidecar_still_reports_hash_of_the_parsed_snapshot(tmp_path, monkeypatch):
    original = fixture_csv(first_time=1)
    replacement = fixture_csv(first_time=101)
    source = write_source(tmp_path, original)
    real_read_csv = pd.read_csv

    def replace_after_read(*args, **kwargs):
        frame = real_read_csv(*args, **kwargs)
        source.write_bytes(replacement)
        return frame

    monkeypatch.setattr(pd, "read_csv", replace_after_read)
    result = load_cell(source, tmp_path, monkeypatch)
    assert source.read_bytes() == replacement
    assert result["frame"]["Time"].tolist() == [1, 2]
    assert result["dataset_sha256"] == hashlib.sha256(original).hexdigest()


def test_replacement_cannot_make_a_mismatched_sidecar_validate_an_old_frame(tmp_path, monkeypatch):
    original = fixture_csv(first_time=1)
    replacement = fixture_csv(first_time=101)
    source = write_source(tmp_path, original)
    sidecar_for(source, replacement, data_kind="synthetic")
    real_read_csv = pd.read_csv

    def replace_after_read(*args, **kwargs):
        frame = real_read_csv(*args, **kwargs)
        source.write_bytes(replacement)
        return frame

    monkeypatch.setattr(pd, "read_csv", replace_after_read)
    with pytest.raises(ValueError, match="csv_sha256|CSV hash"):
        load_cell(source, tmp_path, monkeypatch)
    assert source.read_bytes() == original


@pytest.mark.parametrize("suffix", [".csv.gz", ".csv.zip", ".csv.tar.gz"])
def test_compressed_input_infers_from_filename_and_hashes_raw_bytes(tmp_path, monkeypatch, suffix):
    raw = fixture_csv()
    if suffix == ".csv.gz":
        compressed = gzip.compress(raw, mtime=0)
    else:
        buffer = io.BytesIO()
        if suffix == ".csv.zip":
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("rows.csv", raw)
        else:
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                member = tarfile.TarInfo("rows.csv")
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        compressed = buffer.getvalue()
    source = write_source(tmp_path, compressed, suffix=suffix)
    sidecar_for(source, compressed, data_kind="synthetic")
    expected_frame = pd.read_csv(source)

    result = load_cell(source, tmp_path, monkeypatch)

    pd.testing.assert_frame_equal(result["frame"], expected_frame)
    assert result["dataset_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert result["data_kind"] == "synthetic"


def test_dangling_sidecar_is_rejected(tmp_path, monkeypatch):
    source = write_source(tmp_path, fixture_csv())
    source.with_suffix(".metadata.json").symlink_to(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="sidecar"):
        load_cell(source, tmp_path, monkeypatch)

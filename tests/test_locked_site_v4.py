"""Turku locked-site handling: the label seal must never be guessed, and the
label loader must refuse until both v4 model locks exist."""

import json

import pytest

from scripts import locked_site_v4


def test_load_labels_refuses_without_any_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    with pytest.raises(RuntimeError, match="outer-models-lock"):
        locked_site_v4.load_labels()


def test_load_labels_refuses_with_only_outer_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    (tmp_path / "outer-models-lock.json").write_text("{}")
    with pytest.raises(RuntimeError, match="final-models-lock"):
        locked_site_v4.load_labels()


def test_load_labels_succeeds_once_both_locks_exist(tmp_path, monkeypatch):
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    (sealed_dir / "labels.json").write_text(json.dumps({"n_labels": 1, "labels": {"01": {"label": 1}}}))
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    monkeypatch.setattr(locked_site_v4, "SEALED_DIR", sealed_dir)
    (tmp_path / "outer-models-lock.json").write_text("{}")
    (tmp_path / "final-models-lock.json").write_text("{}")
    loaded = locked_site_v4.load_labels()
    assert loaded["n_labels"] == 1


def _rows(header, records):
    idx = {name: header.index(name) for name in header}
    out = [tuple(header)]
    for r in records:
        out.append(tuple(r.get(h) for h in header))
    return out, idx


HEADER = ["id", "PD group", "age", "sex", "LEDD", "PD duration"]


def test_resolves_pd_group_coding_from_clinical_fields_not_hardcoded():
    # 1 = PD here, mirroring the real Turku file: LEDD/duration only for group 1.
    records = [
        {"id": "01", "PD group": 1, "age": 60, "sex": "M", "LEDD": 400, "PD duration": 5},
        {"id": "02", "PD group": 1, "age": 61, "sex": "F", "LEDD": 300, "PD duration": None},
        {"id": "03", "PD group": 0, "age": 62, "sex": "M", "LEDD": None, "PD duration": None},
        {"id": "04", "PD group": 0, "age": 63, "sex": "F", "LEDD": None, "PD duration": None},
    ]
    rows, idx = _rows(HEADER, records)
    data_rows = rows[1:]
    pd_value, control_value, ledd_by_group, duration_by_group = locked_site_v4.resolve_pd_group_coding(
        data_rows, idx["PD group"], idx["LEDD"], idx["PD duration"]
    )
    assert pd_value == 1
    assert control_value == 0


def test_resolves_pd_group_coding_even_when_flipped():
    # Same invariant, opposite numeric coding: whichever value has LEDD is PD.
    records = [
        {"id": "01", "PD group": 0, "age": 60, "sex": "M", "LEDD": 400, "PD duration": 5},
        {"id": "02", "PD group": 0, "age": 61, "sex": "F", "LEDD": 300, "PD duration": None},
        {"id": "03", "PD group": 1, "age": 62, "sex": "M", "LEDD": None, "PD duration": None},
        {"id": "04", "PD group": 1, "age": 63, "sex": "F", "LEDD": None, "PD duration": None},
    ]
    rows, idx = _rows(HEADER, records)
    data_rows = rows[1:]
    pd_value, control_value, *_ = locked_site_v4.resolve_pd_group_coding(
        data_rows, idx["PD group"], idx["LEDD"], idx["PD duration"]
    )
    assert pd_value == 0
    assert control_value == 1


def test_raises_rather_than_guesses_when_coding_is_ambiguous():
    # LEDD populated in both groups: no clean invariant, must not guess.
    records = [
        {"id": "01", "PD group": 1, "LEDD": 400, "PD duration": 5},
        {"id": "02", "PD group": 0, "LEDD": 100, "PD duration": None},
    ]
    rows, idx = _rows(["id", "PD group", "LEDD", "PD duration"], records)
    data_rows = rows[1:]
    with pytest.raises(ValueError, match="Could not resolve"):
        locked_site_v4.resolve_pd_group_coding(
            data_rows, idx["PD group"], idx["LEDD"], idx["PD duration"]
        )


def test_raises_rather_than_guesses_on_unexpected_group_values():
    records = [
        {"id": "01", "PD group": 2, "LEDD": 400, "PD duration": 5},
        {"id": "02", "PD group": 0, "LEDD": None, "PD duration": None},
    ]
    rows, idx = _rows(["id", "PD group", "LEDD", "PD duration"], records)
    data_rows = rows[1:]
    with pytest.raises(ValueError, match="Unexpected"):
        locked_site_v4.resolve_pd_group_coding(
            data_rows, idx["PD group"], idx["LEDD"], idx["PD duration"]
        )

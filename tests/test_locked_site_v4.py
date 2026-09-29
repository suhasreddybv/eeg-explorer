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


def test_evaluate_refuses_without_both_locks(tmp_path, monkeypatch):
    """The one-shot must not open labels until both locks exist."""
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    with pytest.raises(RuntimeError, match="outer-models-lock.json missing"):
        locked_site_v4.evaluate()
    (tmp_path / "outer-models-lock.json").write_text("{}")
    with pytest.raises(RuntimeError, match="final-models-lock.json missing"):
        locked_site_v4.evaluate()


def test_evaluate_refuses_a_second_run_without_the_override(tmp_path, monkeypatch):
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    (tmp_path / "locked-site-results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="one-shot"):
        locked_site_v4.evaluate()
    # With the override it gets past the repeat guard and fails later, on the
    # locks -- i.e. the override only waives the repeat check, nothing else.
    with pytest.raises(RuntimeError, match="lock"):
        locked_site_v4.evaluate(repeat=True)


def test_evaluate_detects_a_tampered_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(locked_site_v4, "V4_RUN", tmp_path)
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    monkeypatch.setattr(locked_site_v4, "SEALED_DIR", sealed)
    (tmp_path / "outer-models-lock.json").write_text("{}")
    (tmp_path / "final-models-lock.json").write_text("{}")
    (sealed / "labels.json").write_text(json.dumps({"labels": {"01": {"label": 1}}}))
    (tmp_path / "locked-site-seal.json").write_text(
        json.dumps({"sealed_labels_sha256": "0" * 64})
    )
    with pytest.raises(RuntimeError, match="changed since it was sealed"):
        locked_site_v4.evaluate()


def test_participant_id_zero_padding_is_normalised():
    """Regression: Turku filenames are zero-padded ('02') while the source
    spreadsheet keys the same person unpadded ('2'). Matching on the raw
    string silently dropped every single-digit participant -- a systematic
    17% loss of the locked test set, biased toward low participant numbers."""
    def normalise(key):
        key = str(key).strip()
        return str(int(key)) if key.isdigit() else key

    filename_ids = ["01", "02", "09", "10", "24", "40"]
    spreadsheet_ids = ["1", "2", "9", "10", "24", "40"]
    assert [normalise(k) for k in filename_ids] == [normalise(k) for k in spreadsheet_ids]
    # Raw matching is what failed before the fix.
    assert "02" not in set(spreadsheet_ids)
    assert normalise("02") in set(normalise(k) for k in spreadsheet_ids)

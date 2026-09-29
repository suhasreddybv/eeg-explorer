"""BrainLat second locked site: site-qualified identity (the bug class that
cost the Turku one-shot), credential handling, and the label seal."""

import json

import pytest

from scripts import brainlat_v4


def test_person_keys_are_site_qualified_so_repeated_subject_ids_do_not_collide():
    # BrainLat reuses sub-40001 under both PD/AR and PD/CL.
    ar = brainlat_v4.person_key("AR", "sub-40001")
    cl = brainlat_v4.person_key("CL", "sub-40001")
    assert ar != cl
    assert ar == "brainlat:AR:sub-40001"


def test_person_key_rejects_unknown_site():
    with pytest.raises(ValueError, match="Unknown BrainLat site"):
        brainlat_v4.person_key("UV", "sub-1")  # Colombia: no PD EEG, excluded


def test_download_refuses_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNAPSE_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(brainlat_v4.Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(RuntimeError, match="No Synapse credentials"):
        brainlat_v4._client()


def test_load_labels_refuses_without_both_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(brainlat_v4, "V4_RUN", tmp_path)
    with pytest.raises(RuntimeError, match="outer-models-lock"):
        brainlat_v4.load_labels()
    (tmp_path / "outer-models-lock.json").write_text("{}")
    with pytest.raises(RuntimeError, match="final-models-lock"):
        brainlat_v4.load_labels()


def test_load_labels_reads_sealed_file_once_both_locks_exist(tmp_path, monkeypatch):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "labels.json").write_text(
        json.dumps({"n_labels": 1, "labels": {"brainlat:CL:sub-40001": {"label": 1, "site": "CL"}}})
    )
    monkeypatch.setattr(brainlat_v4, "V4_RUN", tmp_path)
    monkeypatch.setattr(brainlat_v4, "SEALED_DIR", sealed)
    (tmp_path / "outer-models-lock.json").write_text("{}")
    (tmp_path / "final-models-lock.json").write_text("{}")
    assert brainlat_v4.load_labels()["n_labels"] == 1

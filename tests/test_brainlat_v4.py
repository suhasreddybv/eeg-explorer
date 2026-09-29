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


def _rows():
    return [
        {"group": "3_PD", "site": "AR", "subject": "sub-2", "folder_id": "a", "files": ["s2_AR_PD_reject.set", "s2.fdt"]},
        {"group": "3_PD", "site": "AR", "subject": "sub-1", "folder_id": "b", "files": ["s1_AR_PD_reject.set"]},
        {"group": "3_PD", "site": "CL", "subject": "sub-3", "folder_id": "c", "files": ["sub-3_rs_eeg.set"]},
        {"group": "5_HC", "site": "CL", "subject": "sub-4", "folder_id": "d", "files": ["sub-4_rs_eeg.set"]},
        {"group": "5_HC", "site": "CL", "subject": "sub-5", "folder_id": "e", "files": []},
        {"group": "5_HC", "site": "AR", "subject": "sub-6", "folder_id": "f", "files": ["a.set", "b.set"]},
    ]


def test_naming_summary_counts_preprocessing_flags_by_group_and_site():
    s = brainlat_v4.naming_summary(_rows())
    assert s["3_PD/AR"] == {"flagged": 2, "plain": 0, "none_or_ambiguous": 0}
    assert s["3_PD/CL"] == {"flagged": 0, "plain": 1, "none_or_ambiguous": 0}
    assert s["5_HC/CL"] == {"flagged": 0, "plain": 1, "none_or_ambiguous": 1}
    assert s["5_HC/AR"]["none_or_ambiguous"] == 1  # two .set files: refuse to pick


def test_inspection_sample_is_one_flagged_and_one_plain_per_group_in_sorted_order():
    sample = brainlat_v4.pick_inspection_sample(_rows())
    picked = {(s["group"], s["naming_class"]): s["subject"] for s in sample}
    assert picked[("3_PD", "flagged")] == "sub-1"  # sorted by (site, subject), not listing order
    assert picked[("3_PD", "plain")] == "sub-3"
    assert picked[("5_HC", "plain")] == "sub-4"
    assert ("5_HC", "flagged") not in picked  # no flagged HC file exists: nothing is invented


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

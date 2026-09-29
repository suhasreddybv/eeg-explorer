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


def test_cohort_filter_keeps_chile_plain_files_only_and_counts_the_rest_without_names():
    kept, excluded = brainlat_v4.select_subjects(_rows())
    assert [r["subject"] for r in kept] == ["sub-3", "sub-4"]
    assert excluded["3_PD/AR"] == {"other_site": 2}
    assert excluded["5_HC/CL"] == {"no_or_ambiguous_eeg_file": 1}
    assert excluded["5_HC/AR"] == {"other_site": 1}
    # Counts only: no subject id appears anywhere in the exclusion record.
    assert "sub-" not in json.dumps(excluded)


def test_cohort_filter_excludes_flagged_chile_pd():
    rows = _rows() + [{"group": "3_PD", "site": "CL", "subject": "sub-9", "folder_id": "g", "files": ["s9_CL_PD_reject.set"]}]
    kept, excluded = brainlat_v4.select_subjects(rows)
    assert "sub-9" not in [r["subject"] for r in kept]
    assert excluded["3_PD/CL"] == {"preprocessing_flagged_name": 1}


def test_interpolation_recovers_a_smooth_scalp_field_at_the_17_v3_positions():
    """A field that is linear in scalp direction must be reproduced at the
    virtual 10-20 sites from the 128 Biosemi electrodes. This is the whole
    basis for scoring BrainLat with a 17-channel model."""
    import mne
    import numpy as np

    from pd_pipeline.cohort_v3 import CONFIG

    bio = mne.channels.make_standard_montage("biosemi128")
    pos = bio.get_positions()["ch_pos"]
    names = list(pos)
    unit = np.array([pos[n] / np.linalg.norm(pos[n]) for n in names])
    field = unit[:, 0] + 0.5 * unit[:, 1] - 0.3 * unit[:, 2]  # linear in direction
    t = np.linspace(0, 1, 256)
    data = field[:, None] * np.sin(2 * np.pi * 10 * t)[None, :] * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(names, 256.0, "eeg"), verbose=False)
    raw.set_montage(bio, verbose=False)

    out = brainlat_v4.interpolate_to_v3(raw)
    assert out.ch_names == list(CONFIG["channels"])
    # Directions must be taken about the standard montage's own sphere
    # centre (a realistic head, ~16 mm behind the origin), as the code does.
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    std_centre, _ = brainlat_v4._fit_sphere(np.array(list(std.values())))
    target_unit = np.array([(std[c] - std_centre) / np.linalg.norm(std[c] - std_centre) for c in out.ch_names])
    expected = target_unit[:, 0] + 0.5 * target_unit[:, 1] - 0.3 * target_unit[:, 2]
    got = out.get_data()[:, 64] / (np.sin(2 * np.pi * 10 * t[64]) * 1e-5)
    assert np.corrcoef(expected, got)[0, 1] > 0.99
    assert np.abs(expected - got).max() < 0.15  # field spans about 2 units


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

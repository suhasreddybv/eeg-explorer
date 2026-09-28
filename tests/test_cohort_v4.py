"""v4 cohort registry: A1 one-session-per-person selection, Turku
locked-site records, and that v3's extraction/dedup logic is reused
unmodified (not reimplemented, so its duplicate-signal and cross-identity
checks still apply)."""

import json

import pytest

from pd_pipeline import cohort_v3, cohort_v4


def _record(**over):
    base = {
        "id": "id",
        "path": "path",
        "dataset": "ds000000",
        "site": "site",
        "person": "person",
        "role": "supervised",
        "label": 0,
        "interval": None,
        "protocol_exclusion": None,
        "session": "single",
        "medication": "unknown",
        "age": "60",
        "sex": "M",
        "sources": [],
    }
    base.update(over)
    return base


def test_reuses_cohort_v3_build_rather_than_reimplementing_it():
    # If this ever stops being the same function object, v4 has started
    # reimplementing extraction/labelling/dedup logic instead of reusing
    # it, and v3's duplicate-signal / cross-identity checks (which live
    # only inside cohort_v3.build) would no longer be guaranteed to apply.
    assert cohort_v4.build_v3 is cohort_v3.build


def test_ucsd_pd_selects_on_session_control_selects_hc():
    records = [
        _record(id="a", dataset="ds002778", person="ds002778:sub-pd1", label=1, session="ses-on"),
        _record(id="b", dataset="ds002778", person="ds002778:sub-pd1", label=1, session="ses-off"),
        _record(id="c", dataset="ds002778", person="ds002778:sub-hc1", label=0, session="ses-hc"),
    ]
    kept, notes = cohort_v4.select_a1_session(records)
    kept_ids = {r["id"] for r in kept}
    assert kept_ids == {"a", "c"}
    assert notes["ds002778:sub-pd1"] == "ses-on"
    assert notes["ds002778:sub-hc1"] == "ses-hc"


def test_unm_pd_selects_on_medication_session_control_keeps_single():
    records = [
        _record(id="a", dataset="ds003490", person="ds003490:sub-001", label=1, session="ses-01", medication="ON"),
        _record(id="b", dataset="ds003490", person="ds003490:sub-001", label=1, session="ses-02", medication="OFF"),
        _record(id="c", dataset="ds003490", person="ds003490:sub-003", label=0, session="ses-01", medication="n/a"),
    ]
    kept, notes = cohort_v4.select_a1_session(records)
    kept_ids = {r["id"] for r in kept}
    assert kept_ids == {"a", "c"}
    assert notes["ds003490:sub-001"] == "ses-01"


def test_tel_aviv_single_recording_kept_as_is():
    records = [_record(id="a", dataset="ds007526", person="ds007526:sub-001", label=1, session="single")]
    kept, notes = cohort_v4.select_a1_session(records)
    assert {r["id"] for r in kept} == {"a"}


def test_iowa_selects_earliest_session_by_days_since_first(tmp_path):
    sub_dir = tmp_path / "datasets" / "openneuro" / "ds008768" / "sub-1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-1_sessions.tsv").write_text(
        "session_id\tdays_since_first_session\n"
        "ses-01\t0\n"
        "ses-02\t933\n"
    )
    records = [
        _record(id="a", dataset="ds008768", person="ds008768:sub-1", label=0, session="ses-01"),
        _record(id="b", dataset="ds008768", person="ds008768:sub-1", label=0, session="ses-02"),
    ]
    kept, notes = cohort_v4.select_a1_session(records, archive=tmp_path)
    assert {r["id"] for r in kept} == {"a"}
    assert notes["ds008768:sub-1"] == "ses-01"


def test_exactly_one_session_per_person_across_mixed_sites(tmp_path):
    sub_dir = tmp_path / "datasets" / "openneuro" / "ds008768" / "sub-1"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-1_sessions.tsv").write_text(
        "session_id\tdays_since_first_session\nses-01\t0\nses-02\t100\n"
    )
    records = [
        _record(id="a", dataset="ds008768", person="ds008768:sub-1", label=0, session="ses-01"),
        _record(id="b", dataset="ds008768", person="ds008768:sub-1", label=0, session="ses-02"),
        _record(id="c", dataset="ds002778", person="ds002778:sub-pd1", label=1, session="ses-on"),
        _record(id="d", dataset="ds002778", person="ds002778:sub-pd1", label=1, session="ses-off"),
        _record(id="e", role="auxiliary", dataset="ds005385", person="ds005385:sub-1"),
    ]
    kept, _ = cohort_v4.select_a1_session(records, archive=tmp_path)
    supervised = [r for r in kept if r["role"] == "supervised"]
    people = [r["person"] for r in supervised]
    assert len(people) == len(set(people))
    assert {r["id"] for r in kept} == {"a", "c", "e"}


def test_a1_selection_raises_rather_than_guesses_when_ambiguous():
    # Two "ON" sessions for the same PD person: not resolvable, must raise.
    records = [
        _record(id="a", dataset="ds003490", person="ds003490:sub-001", label=1, session="ses-01", medication="ON"),
        _record(id="b", dataset="ds003490", person="ds003490:sub-001", label=1, session="ses-02", medication="ON"),
    ]
    with pytest.raises(ValueError, match="stop and ask"):
        cohort_v4.select_a1_session(records)


def test_turku_records_have_no_labels_and_correct_count(tmp_path):
    locked = tmp_path / "locked-sites" / "osf-pehj9"
    locked.mkdir(parents=True)
    manifest = {
        "files": [
            {"name": f"{i:02d}_eyesOpen.set", "role": "raw", "sha256": f"hash{i}", "bytes": 100}
            for i in range(1, 41)
        ]
        + [{"name": "InfoAboutParticipants.xlsx", "role": "provenance", "sha256": "x", "bytes": 1}]
    }
    (locked / "manifest.json").write_text(json.dumps(manifest))
    records = cohort_v4.turku_locked_records(archive=tmp_path)
    assert len(records) == 40
    assert all(r["label"] is None for r in records)
    assert all(r["role"] == "locked_test" for r in records)
    assert all(r["site"] == "turku" for r in records)


def test_turku_records_raise_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        cohort_v4.turku_locked_records(archive=tmp_path)


def test_clinical_data_reads_iowa_session_and_tel_aviv_participants_tsv(tmp_path):
    iowa_dir = tmp_path / "datasets" / "openneuro" / "ds008768" / "sub-1"
    iowa_dir.mkdir(parents=True)
    (iowa_dir / "sub-1_sessions.tsv").write_text(
        "session_id\tage_at_visit\tmoca\tupdrs_part_iii\tdisease_duration\t"
        "levodopa_equivalent_daily_dose\tdays_since_first_session\n"
        "ses-01\t67.1\t28\tn/a\tn/a\tn/a\t0\n"
        "ses-02\t69\tn/a\tn/a\tn/a\tn/a\t933\n"
    )
    tel_aviv_dir = tmp_path / "provenance" / "openneuro" / "ds007526"
    tel_aviv_dir.mkdir(parents=True)
    (tel_aviv_dir / "participants.tsv").write_text(
        "participant_id\tmoca\tupdrs_part_iii\tupdrs_total\tdisease_duration\tledd\n"
        "sub-001\t30.0\t0.0\t0.0\tn/a\tn/a\n"
    )
    records = [
        {
            "id": "a",
            "dataset": "ds008768",
            "site": "iowa",
            "person": "ds008768:sub-1",
            "role": "supervised",
            "session": "ses-01",
            "label": 0,
        },
        {
            "id": "b",
            "dataset": "ds007526",
            "site": "tel_aviv",
            "person": "ds007526:sub-001",
            "role": "supervised",
            "session": "single",
            "label": 1,
        },
    ]
    clinical = cohort_v4.clinical_data(records, archive=tmp_path, run=tmp_path)
    assert clinical["ds008768:sub-1"]["moca"] == "28"
    assert clinical["ds007526:sub-001"]["moca"] == "30.0"

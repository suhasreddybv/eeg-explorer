"""Turku eyes-closed control: the pre-registered decision rule, participant
matching across the two conditions, and record construction."""

import json

import pytest

from scripts import turku_eyesclosed_control_v4 as ec


def test_verdict_thresholds_match_protocol_section_14_2():
    assert ec.verdict(0.0) == "site_generalisation_interpretable"
    assert ec.verdict(0.05) == "site_generalisation_interpretable"  # boundary inclusive
    assert ec.verdict(0.0501) == "partially_confounded"
    assert ec.verdict(0.10) == "partially_confounded"  # boundary inclusive
    assert ec.verdict(0.1001) == "confounded_by_condition"


def test_negative_delta_is_not_a_condition_penalty():
    # EC scoring *better* than EO is not evidence the condition hurt.
    assert ec.verdict(-0.2) == "site_generalisation_interpretable"


def test_participants_match_across_conditions_despite_zero_padding():
    # Eyes-open files are '02_eyesOpen.set', eyes-closed '02_eyesClosed.set',
    # the spreadsheet says '2'. All three are one person.
    assert ec.normalise_id("02") == ec.normalise_id("2") == ec.normalise_id(" 2 ")
    assert ec.normalise_id("10") == "10"
    assert ec.normalise_id("sub-x") == "sub-x"  # non-numeric passes through


def _ec_entries(n=ec.EXPECTED_EC, start=1):
    return [
        {"name": f"{i:02d}_eyesClosed.set", "role": "raw", "sha256": f"h{i}", "bytes": 10}
        for i in range(start, start + n)
    ]


def test_ec_records_are_built_from_the_eyes_closed_manifest_only(tmp_path):
    manifest = {
        "files": _ec_entries()
        + [
            {"name": "01_eyesOpen.set", "role": "raw", "sha256": "eo", "bytes": 10},
            {"name": "InfoAboutParticipants.xlsx", "role": "provenance", "sha256": "x", "bytes": 1},
        ]
    }
    path = tmp_path / "manifest-eyesClosed.json"
    path.write_text(json.dumps(manifest))
    records = ec.ec_records(path)
    assert len(records) == ec.EXPECTED_EC == 39
    assert all(r["role"] == "locked_test_ec" for r in records)
    assert all(r["label"] is None for r in records)
    assert records[0]["subject"] == "1" and records[0]["person"] == "osf-pehj9-ec:01"
    # The eyes-open file present in the manifest must not leak into EC records.
    assert not any("eyesOpen" in r["path"] for r in records)


def test_ec_records_refuse_wrong_count(tmp_path):
    path = tmp_path / "manifest-eyesClosed.json"
    path.write_text(json.dumps({"files": [{"name": "01_eyesClosed.set", "role": "raw", "sha256": "h", "bytes": 1}]}))
    with pytest.raises(ValueError, match="Expected 39"):
        ec.ec_records(path)


def test_ec_records_refuse_duplicate_files_masquerading_as_full_count(tmp_path):
    # The bug that hid a missing file: 38 distinct entries plus the same file
    # listed again must not pass as 39.
    entries = _ec_entries(38) + [_ec_entries(1, start=38)[0]]
    path = tmp_path / "manifest-eyesClosed.json"
    path.write_text(json.dumps({"files": entries}))
    with pytest.raises(ValueError, match="distinct"):
        ec.ec_records(path)

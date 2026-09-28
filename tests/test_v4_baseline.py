"""Gate G1: the v4 baseline-reproduction script's fold construction must
match the frozen v3 splits (site sets, grouping, disjointness) before its
recomputed AUROC numbers can be trusted."""

import json

import pytest

from pd_pipeline.cohort_v3 import RUN as V3_RUN
from pd_pipeline.compare_v3 import validate_disjoint
from scripts.reproduce_v3_baseline import build_folds

pytestmark = pytest.mark.skipif(
    not (V3_RUN / "splits.json").exists(), reason="v3 run artifacts not present"
)


def test_fold_site_sets_match_frozen_v3_splits():
    people, splits = build_folds()
    frozen = json.loads((V3_RUN / "splits.json").read_text())
    assert [p["key"] for p in people] == [p["key"] for p in frozen["people"]]
    assert [s["name"] for s in splits] == [s["name"] for s in frozen["folds"]]
    for built, saved in zip(splits, frozen["folds"]):
        assert built["kind"] == saved["kind"]
        assert built["train"] == saved["train"]
        assert built["test"] == saved["test"]


def test_folds_are_participant_disjoint():
    people, splits = build_folds()
    for split in splits:
        train = [people[i] for i in split["train"]]
        test = [people[i] for i in split["test"]]
        validate_disjoint(train, test)
        assert set(split["train"]).isdisjoint(split["test"])


def test_cross_site_folds_partition_by_site_only():
    people, splits = build_folds()
    for split in splits:
        if split["kind"] != "cross_site":
            continue
        test_sites = {people[i]["site"] for i in split["test"]}
        train_sites = {people[i]["site"] for i in split["train"]}
        assert len(test_sites) == 1
        assert test_sites.isdisjoint(train_sites)

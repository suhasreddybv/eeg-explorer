"""Stage 2: the v4 implementation policy must freeze cleanly and refuse a
silent re-freeze if the hashed source changed underneath it."""

import json

from scripts import run_pd_v4


def test_freeze_policy_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pd_v4, "RUN", tmp_path)
    first = run_pd_v4.freeze_policy()
    second = run_pd_v4.freeze_policy()
    assert first == second
    assert json.loads((tmp_path / "implementation-policy.json").read_text()) == first


def test_freeze_policy_refuses_when_source_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pd_v4, "RUN", tmp_path)
    run_pd_v4.freeze_policy()
    # Simulate a source edit after the freeze without touching real files.
    tampered = dict(run_pd_v4.policy())
    tampered["scope"] = "changed after freeze"
    monkeypatch.setattr(run_pd_v4, "policy", lambda: tampered)
    try:
        run_pd_v4.freeze_policy()
        raised = False
    except ValueError:
        raised = True
    assert raised

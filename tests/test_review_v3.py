"""Independent checks for post-lock uncertainty calculations."""

import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.audit_pd_v3 import auc_boot


def test_bootstrap_respects_perfect_and_tied_rankings():
    y = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(auc_boot(y, y, np.random.default_rng(7), 100), 1)
    np.testing.assert_allclose(auc_boot(y, np.ones(6), np.random.default_rng(7), 100), 0.5)


def test_vectorized_bootstrap_matches_reference_on_same_participants():
    y = np.array([0, 1, 0, 1, 0, 1, 1])
    p = np.array([0.3, 0.6, 0.3, 0.8, 0.9, 0.4, 0.6])
    rng = np.random.default_rng(19)
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    indices = np.c_[rng.choice(neg, (30, len(neg))), rng.choice(pos, (30, len(pos)))]
    expected = [roc_auc_score(y[ix], p[ix]) for ix in indices]
    np.testing.assert_allclose(auc_boot(y, p, np.random.default_rng(19), 30), expected)

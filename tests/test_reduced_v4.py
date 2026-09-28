"""v4 reduced-channel montages: the re-referencing rule (the scientific
crux of T2), distillation targets, and the greedy channel search."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pd_pipeline import reduced_v4 as rv  # noqa: E402


def test_montage_channel_lists_match_the_frozen_protocol():
    assert rv.MONTAGES["M4F"] == ["Fp1", "Fp2", "F7", "F8"]
    assert rv.MONTAGES["M4T"] == ["T7", "T8", "P7", "P8"]
    assert rv.MONTAGES["M5S"] == ["F8", "F4", "T7", "O1", "O2"]
    assert rv.MONTAGES["M8"] == ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "O1", "O2"]
    for name, channels in rv.MONTAGES.items():
        assert len(set(channels)) == len(channels), f"{name} repeats a channel"


def test_rereference_uses_only_montage_channels():
    """The crux of T2. After re-referencing, the montage's own channels must
    sum to zero at every sample -- and the result must not depend in any way
    on the electrodes the montage does not have."""
    rng = np.random.default_rng(0)
    preref = rng.normal(size=(3, 17, 64))
    indices = rv.MONTAGE_INDEX["M4F"]

    out = rv.rereference(preref, indices)
    assert out.shape == (3, 4, 64)
    assert np.abs(out.sum(axis=1)).max() < 1e-9

    # Scramble every non-montage channel; the montage output must be identical.
    other = [i for i in range(17) if i not in indices]
    altered = preref.copy()
    altered[:, other, :] = rng.normal(size=(3, len(other), 64)) * 1000
    assert np.allclose(out, rv.rereference(altered, indices))


def test_rereference_differs_from_17_channel_average_reference():
    """A montage re-referenced to 17 channels would leak absent electrodes;
    confirm the two are genuinely different so the distinction is real."""
    rng = np.random.default_rng(1)
    preref = rng.normal(size=(2, 17, 64))
    indices = rv.MONTAGE_INDEX["M4T"]
    montage_ref = rv.rereference(preref, indices)
    seventeen_ref = (preref - preref.mean(axis=1, keepdims=True))[:, indices, :]
    assert not np.allclose(montage_ref, seventeen_ref)


def test_montage_windows_shapes_and_mask():
    rng = np.random.default_rng(2)
    t = np.arange(512) / rv.PREREF_RATE
    base = np.stack([[20.0 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 1, 512) for _ in range(17)]])
    windows = rv.montage_windows(base, rv.MONTAGE_INDEX["M8"])
    assert windows["tokens"].shape == (1, 17, 40)
    assert windows["lpc"].shape == (1, 17, rv.LPC_ORDER)
    assert windows["cov"].shape == (1, 8, 8)
    assert windows["mask"].sum() == 8
    # Channels outside the montage stay exactly zero in the padded arrays.
    outside = [i for i in range(17) if i not in rv.MONTAGE_INDEX["M8"]]
    assert np.all(windows["tokens"][:, outside, :] == 0)


def test_distillation_blends_hard_label_and_teacher():
    hard = np.array([0, 1, 1, 0])
    teacher = np.array([0.2, 0.9, 0.6, 0.4])
    blended = rv.distillation_targets(hard, teacher, weight=0.5)
    assert blended == pytest.approx([0.1, 0.95, 0.8, 0.2])
    # weight=0 recovers the hard labels exactly
    assert rv.distillation_targets(hard, teacher, weight=0.0) == pytest.approx(hard)


def test_distillation_rejects_misaligned_teacher():
    with pytest.raises(ValueError, match="align"):
        rv.distillation_targets(np.array([0, 1]), np.array([0.5]))


def test_greedy_channel_search_is_monotone_and_bounded():
    # Score improves with each added channel from a known-good subset.
    target = set(rv.MONTAGE_INDEX["M4F"])

    def score(channels):
        return len(target & set(channels)) - 0.01 * len(channels)

    trace = rv.greedy_channel_search(score, k_max=5)
    assert len(trace) == 5
    assert [step["k"] for step in trace] == [1, 2, 3, 4, 5]
    # The first four picks should all come from the informative set.
    assert set(rv.CHANNELS.index(c) for c in trace[3]["channels"]) == target


def test_greedy_channel_search_respects_restricted_pool():
    pool = rv.MONTAGE_INDEX["M4T"]
    trace = rv.greedy_channel_search(lambda channels: len(channels), k_max=8, channels=pool)
    assert len(trace) == len(pool)  # cannot exceed the pool
    chosen = {rv.CHANNELS.index(c) for c in trace[-1]["channels"]}
    assert chosen == set(pool)

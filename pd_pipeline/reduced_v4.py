"""v4 reduced-channel montages (T2).

Re-referencing from preref128 within a montage's own channels only,
per-montage feature computation, distillation from the T1 stacker's
inner-OOF probabilities, and the nested greedy channel-budget search.

The re-referencing rule is the point of this stage. v4's 17-channel
features use an average reference over all 17 electrodes, so they encode
information from electrodes a 4-channel headband does not have. Reporting
those as "wearable" performance would be wrong. Every montage here is
therefore rebuilt from `preref128` -- band-passed but deliberately not
average-referenced (see qc_v4, protocol v4.1) -- and re-referenced to the
mean of the montage's own channels.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md S9 for the frozen montage list and gate G6.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

from .cohort_v3 import CONFIG
from .features_v4 import (
    entropy_features,
    lpc_summary,
    relative_spectra,
    robust_distributions,
    specparam_features,
    vectorize_relative_spectra,
    vectorize_robust_distributions,
    vectorize_specparam,
)
from .qc_v4 import LPC_ORDER, levinson_durbin_lpc

RUN = Path("data/pd-runs/v4")
CHANNELS = CONFIG["channels"]
PREREF_RATE = 128

# Frozen montage list (protocol S9).
MONTAGES = {
    "M4F": ["Fp1", "Fp2", "F7", "F8"],
    "M4T": ["T7", "T8", "P7", "P8"],
    "M5S": ["F8", "F4", "T7", "O1", "O2"],
    "M8": ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "O1", "O2"],
}
MONTAGE_INDEX = {
    name: [CHANNELS.index(c) for c in channels] for name, channels in MONTAGES.items()
}


def rereference(preref128, channel_indices):
    """Average-reference a montage using only its own channels.

    preref128 is (n_windows, 17, n_samples), band-passed and un-referenced.
    Returns (n_windows, k, n_samples) for the k montage channels.
    """
    montage = np.asarray(preref128, dtype="float64")[:, channel_indices, :]
    return montage - montage.mean(axis=1, keepdims=True)


def _pad_to_17(values, channel_indices, width):
    """Place k-channel values back into a 17-wide array so the existing
    17-channel feature functions can be reused unchanged, with a mask that
    marks only the montage's channels as present."""
    out = np.zeros((values.shape[0], 17, width), dtype="float32")
    out[:, channel_indices, :] = values
    return out


def montage_windows(preref128, channel_indices):
    """Re-reference, then recompute the per-window arrays the feature
    families need, at the stored 128 Hz rate."""
    montage = rereference(preref128, channel_indices)
    n_windows, k, n_samples = montage.shape

    freqs, psd = signal.welch(montage, fs=PREREF_RATE, nperseg=PREREF_RATE, axis=-1)
    band = np.log10(np.maximum(psd[:, :, (freqs >= 1) & (freqs < 41)], 1e-12)).astype("float32")

    lpc = np.zeros((n_windows, k, LPC_ORDER), dtype="float32")
    for w in range(n_windows):
        for c in range(k):
            lpc[w, c], _ = levinson_durbin_lpc(montage[w, c], LPC_ORDER)

    covs = np.zeros((n_windows, k, k), dtype="float32")
    for w in range(n_windows):
        cov = np.cov(montage[w])
        covs[w] = 0.9 * cov + 0.1 * np.eye(k) * np.trace(cov) / k

    mask = np.zeros(17, dtype=bool)
    mask[channel_indices] = True
    return {
        "tokens": _pad_to_17(band, channel_indices, band.shape[-1]),
        "lpc": _pad_to_17(lpc, channel_indices, LPC_ORDER),
        "preref": _pad_to_17(montage.astype("float32"), channel_indices, n_samples),
        "cov": covs,
        "mask": mask,
    }


def montage_person_features(preref128, montage):
    """Per-person montage features: B-1, B-2, B-4, B-5, B-6 and the montage
    covariance for B-3."""
    indices = MONTAGE_INDEX[montage]
    windows = montage_windows(preref128, indices)
    mask = windows["mask"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b1 = vectorize_relative_spectra(relative_spectra(windows["tokens"], mask))
        b2 = vectorize_specparam(specparam_features(windows["tokens"], mask))
        b4 = lpc_summary(windows["lpc"], mask)
        b5 = entropy_features(windows["preref"], mask).ravel()
        b6 = vectorize_robust_distributions(robust_distributions(windows["tokens"], mask))
        from pyriemann.utils.mean import mean_riemann

        cov_mean = mean_riemann(windows["cov"].astype("float64"))
    return {
        "b1": b1,
        "b2": b2,
        "b4": b4,
        "b5": b5,
        "b6": b6,
        "cov": cov_mean.astype("float32"),
        "mask": mask,
        "n_channels": len(indices),
    }


def build_montage_features(people_keys, montage, run=RUN, registry=None):
    """Compute and cache montage features for the given person keys."""
    registry = registry or json.loads((run / "registry.json").read_text())
    prep = json.loads((run / "preparation.json").read_text())
    status = {r["id"]: r for r in prep["records"]}
    records_by_person = {}
    for record in registry["records"]:
        if status.get(record["id"], {}).get("status") == "ready":
            records_by_person.setdefault(record["person"], []).append(record["id"])

    folder = run / "montage-features" / montage
    folder.mkdir(parents=True, exist_ok=True)
    built = 0
    for key in people_keys:
        target = folder / (key.replace(":", "__") + ".npz")
        if target.exists():
            continue
        ids = records_by_person.get(key)
        if not ids:
            continue
        stacked = []
        for record_id in sorted(ids):
            with np.load(run / "windows" / (record_id + ".npz")) as data:
                stacked.append(data["preref128"])
        preref = np.concatenate(stacked, axis=0)[:15]  # A2 first-15-window rule
        features = montage_person_features(preref, montage)
        np.savez(target, **features)
        built += 1
    return built


def load_montage_people(base_people, montage, run=RUN):
    """Return copies of compare_v4's people records with the 17-channel
    feature blocks swapped for this montage's, so the same nested
    machinery can score them unchanged."""
    folder = run / "montage-features" / montage
    out = []
    for person in base_people:
        path = folder / (person["key"].replace(":", "__") + ".npz")
        if not path.exists():
            continue
        with np.load(path) as data:
            entry = dict(person)
            entry.update(
                {
                    "b1": data["b1"],
                    "b2": data["b2"],
                    "b4": data["b4"],
                    "b5": data["b5"],
                    "b6": data["b6"],
                    "cov": data["cov"],
                    "montage": montage,
                }
            )
        out.append(entry)
    return out


def distillation_targets(hard_labels, teacher_probabilities, weight=0.5):
    """Soft targets for the T2 student: a blend of the hard label and the
    T1 stacker's inner-OOF probability for the same training person. The
    teacher probability must be out-of-fold, never an in-sample fit."""
    hard = np.asarray(hard_labels, dtype="float64")
    teacher = np.asarray(teacher_probabilities, dtype="float64")
    if hard.shape != teacher.shape:
        raise ValueError("Teacher probabilities must align with training labels")
    return (1 - weight) * hard + weight * teacher


def greedy_channel_search(score_fn, k_max=8, channels=None):
    """Nested forward channel selection: start empty, repeatedly add the
    channel that most improves `score_fn`, up to k_max. `score_fn` must be
    computed on inner folds of the training people only -- never on outer
    test people -- so the resulting curve is descriptive of the search, not
    a tuned outer result.
    """
    available = list(range(17)) if channels is None else list(channels)
    chosen = []
    trace = []
    while len(chosen) < k_max and available:
        best_channel, best_score = None, -np.inf
        for candidate in available:
            score = score_fn(chosen + [candidate])
            if score > best_score:
                best_channel, best_score = candidate, score
        chosen.append(best_channel)
        available.remove(best_channel)
        trace.append(
            {"k": len(chosen), "channels": [CHANNELS[i] for i in chosen], "score": float(best_score)}
        )
    return trace


def montage_families():
    """T2 families (protocol S9: F1, F2, F3, F4, F5), with two montage-safe
    substitutions:

    - F0 (v3-rich) and F6 (compact) are excluded: both are defined on the
      17-channel representation, so neither is a wearable claim.
    - F2 uses the `b2` block rather than `b2_c3`. The C3 z-scores are
      derived from a 17-channel normative model; scoring a 4-channel
      headband against them would quietly reintroduce electrodes the
      montage does not have. Refitting normative models per montage over
      the 467-person auxiliary pool was out of scope for this run, so the
      montage F2 is B-2 alone -- recorded, not silently substituted.
    """
    from .models_v4 import FAMILIES

    specs = {}
    for name in ("F1_relative_svm", "F2_specparam_enet", "F3_riemann_logistic", "F4_leapd", "F5_entropy_logistic"):
        spec = dict(FAMILIES[name])
        if name == "F2_specparam_enet":
            spec["block"] = "b2"
        specs[name] = spec
    return specs

"""Stage 8 (Track E): frozen CBraMod probe and the FM gate.

Track E is optional and can only ever yield a *secondary* pipeline. `S(I)`
from Stage 7 stays the pre-declared primary whatever happens here, which
is what keeps this from becoming a post-hoc model swap.

Runs, in order:
  1. a linear probe on frozen CBraMod embeddings, in the same nested folds;
  2. the two negative controls the protocol requires -- permuted labels,
     and the identical architecture with random (untrained) weights;
  3. S+FM, the stacker with the probe's inner-OOF probability added;
  4. the site-identity probe on the embedding, versus v3-rich;
  5. the FM gate.
"""

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from pd_pipeline import compare_v4 as cv  # noqa: E402
from pd_pipeline import fm_v4 as fm  # noqa: E402
from pd_pipeline.harmonize_v4 import site_identity_probe  # noqa: E402
from pd_pipeline.models_v4 import _logistic  # noqa: E402
from pd_pipeline.registry import save  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RUN = Path("data/pd-runs/v4")
SEED = 20261001

FM_FAMILY = {
    "F_FM_probe": {
        "block": "fm",
        "grid": [{"C": 0.1}, {"C": 1.0}],
        "build": lambda p: _logistic(p["C"]),
    }
}


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def composite_from_folds(fold_results, label):
    out = {}
    for kind in ("cross_site", "within_site"):
        per_site = {}
        for fold in fold_results:
            if fold["kind"] != kind:
                continue
            by_site = {}
            for j, person in enumerate(fold["people"]):
                by_site.setdefault(person["site"], ([], []))
                by_site[person["site"]][0].append(person["y"])
                by_site[person["site"]][1].append(fold["scores"][label][j])
            for site, (ys, ss) in by_site.items():
                if len(set(ys)) == 2:
                    per_site.setdefault(site, ([], []))
                    per_site[site][0].extend(ys)
                    per_site[site][1].extend(ss)
        macro = [roc_auc_score(ys, ss) for ys, ss in per_site.values() if len(set(ys)) == 2]
        out[kind] = float(np.mean(macro)) if macro else float("nan")
    return 0.5 * out["cross_site"] + 0.5 * out["within_site"], out


def run_probe(people, normative_pool, label_override=None, seed=SEED):
    """Nested probe. `label_override` supplies permuted labels for the
    negative control, leaving everything else identical."""
    scored = people
    if label_override is not None:
        scored = [dict(p) for p in people]
        for person, y in zip(scored, label_override):
            person["y"] = int(y)
    splits = cv.folds(scored, seed)
    results = [
        cv.run_fold(scored, split, normative_pool, seed, family_specs=FM_FAMILY)
        for split in splits
    ]
    composite, parts = composite_from_folds(results, "F_FM_probe|I")
    return composite, parts, results


def main():
    base = cv.load_people()
    normative_pool = cv.build_normative_pool()
    keys = [p["key"] for p in base]

    print("building CBraMod embeddings (pretrained)...", flush=True)
    fm.build_embeddings(keys, tag="cbramod")
    print("building CBraMod embeddings (random-init control)...", flush=True)
    fm.build_embeddings(keys, tag="cbramod", random_init=True)

    people = fm.load_embeddings(base, tag="cbramod")
    random_people = fm.load_embeddings(base, tag="cbramod-random")
    print(f"people with embeddings: {len(people)} (random-init control: {len(random_people)})", flush=True)
    if len(people) < 50:
        raise RuntimeError("Too few embeddings; check raw200 extraction")

    report = {"protocol_version": "v4.1", "n_people": len(people), "licences": fm.record_licences()}

    print("probe: pretrained ...", flush=True)
    probe_composite, probe_parts, probe_folds = run_probe(people, normative_pool)
    print(f"   composite {probe_composite:.3f} {probe_parts}", flush=True)

    print("control: random-init architecture ...", flush=True)
    random_composite, random_parts, _ = run_probe(random_people, normative_pool)
    print(f"   composite {random_composite:.3f}", flush=True)

    print("control: permuted labels (within site) ...", flush=True)
    rng = np.random.default_rng(SEED)
    sites = np.array([p["site"] for p in people])
    y = np.array([p["y"] for p in people])
    permuted = y.copy()
    for site in np.unique(sites):
        idx = np.flatnonzero(sites == site)
        permuted[idx] = rng.permutation(permuted[idx])
    permuted_composite, _, _ = run_probe(people, normative_pool, label_override=permuted)
    print(f"   composite {permuted_composite:.3f}", flush=True)

    # S+FM: add the probe's OOF column to the Stage 7 stacker inputs.
    print("S+FM stack ...", flush=True)
    t1 = json.loads((RUN / "outer-fold-results.json").read_text())
    t1_by_fold = {fold["name"]: fold for fold in t1}
    fm_by_fold = {fold["name"]: fold for fold in probe_folds}
    stacked = []
    for name, fold in t1_by_fold.items():
        if name not in fm_by_fold:
            continue
        fm_fold = fm_by_fold[name]
        fm_scores = dict(zip([p["key"] for p in fm_fold["people"]], fm_fold["scores"]["F_FM_probe|I"]))
        keys_here = [p["key"] for p in fold["people"]]
        if not all(k in fm_scores for k in keys_here):
            continue
        combined = dict(fold)
        combined["scores"] = dict(fold["scores"])
        base_scores = np.array(fold["scores"]["S|I"])
        fm_column = np.array([fm_scores[k] for k in keys_here])
        # Equal-weight blend in logit space: a deliberately simple, frozen
        # combination, since re-fitting a stacker on outer scores would be
        # outer-test tuning.
        blended = 1 / (1 + np.exp(-(_logit(base_scores) + _logit(fm_column)) / 2))
        combined["scores"]["S+FM"] = blended.tolist()
        stacked.append(combined)
    stack_fm_composite, stack_fm_parts = composite_from_folds(stacked, "S+FM")
    stack_composite, _ = composite_from_folds(stacked, "S|I")
    print(f"   S|I {stack_composite:.3f} -> S+FM {stack_fm_composite:.3f}", flush=True)

    controls = json.loads((RUN / "negative-controls.json").read_text())
    v3_rich_site = controls["site_identity_probe_macro_auroc"]["v3_rich"]
    context = cv.FoldContext(people, people, "I", normative_pool, SEED)
    fm_site = site_identity_probe(context.block(people, "fm"), sites, SEED)
    print(f"   site-identity probe: FM {fm_site:.3f} vs v3-rich {v3_rich_site:.3f}", flush=True)

    report.update(
        {
            "probe": {"composite": probe_composite, **probe_parts},
            "controls": {
                "random_init_composite": random_composite,
                "permuted_labels_composite": permuted_composite,
            },
            "stack": {"S_I": stack_composite, "S_plus_FM": stack_fm_composite},
            "site_identity_probe": {"fm": fm_site, "v3_rich": v3_rich_site},
            "gate": fm.fm_gate(
                probe_composite=probe_composite,
                stack_composite=stack_composite,
                stack_with_fm_composite=stack_fm_composite,
                probe_site_auroc=fm_site,
                v3_rich_site_auroc=v3_rich_site,
                permuted_composite=permuted_composite,
                random_init_composite=random_composite,
            ),
            "embedding_dimension": int(people[0]["fm"].shape[0]),
            "caveat": (
                "The pooled embedding is 27,200-dimensional for 324 people, so the "
                "linear probe is in a severe p>>n regime even at C=0.1. A weak probe "
                "result cannot be cleanly separated from that."
            ),
        }
    )
    save(RUN / "fm-results.json", report)
    print("\nGATE:", json.dumps(report["gate"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()

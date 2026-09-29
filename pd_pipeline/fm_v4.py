"""v4 Track E foundation-model probes (optional, gated).

Frozen CBraMod embeddings, a linear probe inside the same nested folds,
and the FM gate. Track E can only ever produce a *secondary* pipeline:
`S(I)` from Stage 7 remains the pre-declared primary regardless of what
happens here, which is exactly what stops this from becoming a post-hoc
model swap.

Model selection and why (licences checked before any weights were pulled,
per the protocol's stop-and-ask rule):

- **CBraMod** `braindecode/cbramod-pretrained` -- BSD-3-Clause, ungated.
  4,951,202 parameters, 200 Hz, accepts 17x800 (4 s) directly. USED.
- **LaBraM** `braindecode/labram-pretrained` -- BSD-3-Clause, but SKIPPED
  on technical grounds: the released checkpoint carries a
  `temporal_embedding` of shape [1, 16, 200] while braindecode's wrapper
  builds [1, 5, 200] for a 4 s window, and the mismatch does not resolve
  by changing `n_times`. v4's 4-second window is fixed by the frozen
  protocol (and by A2's first-15-windows rule), so the window was not
  reshaped to suit the checkpoint.
- **REVE** `brain-bzh/reve-base` -- licence declared only as "other", and
  the companion `brain-bzh/reve-positions` declares none at all. SKIPPED
  under the protocol's "if a licence is unclear, skip that model and note
  it" rule.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from .cohort_v3 import ARCHIVE, safe_raw
from .qc_v4 import extract
from .registry import save

RUN = Path("data/pd-runs/v4")
CBRAMOD_REPO = "braindecode/cbramod-pretrained"
FM_RATE = 200
FM_WINDOW = 800
# CBraMod's documented input scaling: amplitudes in units of 100 uV.
FM_UNIT_SCALE = 100.0
SEED = 20261001

SKIPPED_MODELS = {
    "labram": "BSD-3-Clause, but the checkpoint's temporal_embedding [1,16,200] does not match the 4 s window v4's protocol fixes.",
    "reve": "Licence declared only as 'other'; companion reve-positions declares none. Skipped under the protocol's unclear-licence rule.",
}


def extract_raw200(record_ids, run=RUN, archive=ARCHIVE):
    """Re-extract only the raw200 array for the given records, into a
    separate cache. The Stage 4 windows cache is left untouched: it backs
    the frozen Stage 7 outer lock and must not be rewritten."""
    thresholds = json.loads((run / "qc-thresholds.json").read_text())
    registry = json.loads((run / "registry.json").read_text())
    by_id = {r["id"]: r for r in registry["records"]}
    folder = run / "raw200"
    folder.mkdir(parents=True, exist_ok=True)
    built = 0
    for record_id in record_ids:
        target = folder / (record_id + ".npz")
        if target.exists():
            continue
        record = by_id[record_id]
        with safe_raw(archive / record["path"]) as raw:
            result, reason = extract(
                raw, record["interval"], record["role"], thresholds, include_raw200=True
            )
        if reason or not len(result["raw200"]):
            continue
        np.savez(target, raw200=result["raw200"], starts=result["starts"])
        built += 1
    return built


def load_cbramod(random_init=False):
    """Frozen CBraMod. `random_init=True` builds the same architecture with
    untrained weights -- the protocol's negative control (v), which
    separates "the pretrained representation helps" from "this
    architecture's inductive bias helps"."""
    from braindecode.models import CBraMod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if random_init:
            model = CBraMod(n_outputs=2, n_chans=17, n_times=FM_WINDOW, sfreq=FM_RATE)
        else:
            model = CBraMod.from_pretrained(
                CBRAMOD_REPO, n_outputs=2, n_chans=17, n_times=FM_WINDOW, sfreq=FM_RATE
            )
    model.eval()
    return model


@torch.no_grad()
def embed_windows(model, raw200, batch=32):
    """Per-window embeddings from the frozen backbone (the penultimate
    representation, not the 2-class head)."""
    features = []
    hooked = {}

    def hook(_module, inputs, _output):
        hooked["value"] = inputs[0].detach()

    handle = model.final_layer.register_forward_hook(hook)
    try:
        for start in range(0, len(raw200), batch):
            chunk = np.asarray(raw200[start : start + batch], dtype="float32") / FM_UNIT_SCALE
            model(torch.from_numpy(chunk))
            value = hooked["value"]
            features.append(value.reshape(len(chunk), -1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features)


def person_embedding(model, raw200):
    """Pool a person's window embeddings to mean + SD, as the protocol
    specifies."""
    windows = embed_windows(model, raw200)
    return np.concatenate([windows.mean(axis=0), windows.std(axis=0)])


def build_embeddings(people_keys, run=RUN, random_init=False, tag="cbramod"):
    registry = json.loads((run / "registry.json").read_text())
    ready = {}
    for record in registry["records"]:
        path = run / "raw200" / (record["id"] + ".npz")
        if path.exists():
            ready.setdefault(record["person"], []).append(record["id"])

    model = load_cbramod(random_init=random_init)
    folder = run / "fm-embeddings" / (tag + ("-random" if random_init else ""))
    folder.mkdir(parents=True, exist_ok=True)
    built = 0
    for key in people_keys:
        target = folder / (key.replace(":", "__") + ".npy")
        if target.exists() or key not in ready:
            continue
        stacked = []
        for record_id in sorted(ready[key]):
            with np.load(run / "raw200" / (record_id + ".npz")) as data:
                stacked.append(data["raw200"])
        raw200 = np.concatenate(stacked, axis=0)[:15]  # A2 first-15-window rule
        np.save(target, person_embedding(model, raw200))
        built += 1
    return built


def load_embeddings(people, run=RUN, tag="cbramod"):
    """Attach embeddings to compare_v4 people records as an `fm` block."""
    folder = run / "fm-embeddings" / tag
    out = []
    for person in people:
        path = folder / (person["key"].replace(":", "__") + ".npy")
        if not path.exists():
            continue
        entry = dict(person)
        entry["fm"] = np.load(path)
        out.append(entry)
    return out


def fm_gate(probe_composite, stack_composite, stack_with_fm_composite, probe_site_auroc,
            v3_rich_site_auroc, permuted_composite, random_init_composite):
    """All three conditions must hold for the FM to stay in `S+FM`
    (protocol S8). `S+FM` is secondary either way."""
    beats_controls = (
        probe_composite > permuted_composite and probe_composite > random_init_composite
    )
    adds_to_stack = (stack_with_fm_composite - stack_composite) >= 0.01
    transfers_better = probe_site_auroc <= v3_rich_site_auroc
    return {
        "beats_negative_controls": bool(beats_controls),
        "adds_at_least_0.01_to_stack": bool(adds_to_stack),
        "site_probe_no_worse_than_v3_rich": bool(transfers_better),
        "passed": bool(beats_controls and adds_to_stack and transfers_better),
        "detail": {
            "probe_composite": probe_composite,
            "permuted_composite": permuted_composite,
            "random_init_composite": random_init_composite,
            "stack_composite": stack_composite,
            "stack_with_fm_composite": stack_with_fm_composite,
            "delta_to_stack": stack_with_fm_composite - stack_composite,
            "probe_site_identity_auroc": probe_site_auroc,
            "v3_rich_site_identity_auroc": v3_rich_site_auroc,
        },
    }


def record_licences(run=RUN):
    report = {
        "checked_before_download": True,
        "used": {"cbramod": {"repo": CBRAMOD_REPO, "licence": "bsd-3-clause", "gated": False}},
        "skipped": SKIPPED_MODELS,
    }
    save(run / "fm-licences.json", report)
    return report

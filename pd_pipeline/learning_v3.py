"""Compact masked spectral pretraining; no PD participant enters auxiliary fitting."""

import copy
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from .cohort_v3 import RUN
from .registry import digest, freeze, save
from .preprocess import file_hash

SEED = 20260927


class SpectralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(34, 48)
        self.position = nn.Parameter(torch.randn(1, 17, 48) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 48))
        layer = nn.TransformerEncoderLayer(48, 4, 96, 0.2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.decode = nn.Linear(48, 34)
        self.head = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 1))

    def forward(self, x, available, hidden=None):
        h = self.project(x)
        if hidden is not None:
            h = torch.where(hidden[..., None], self.mask_token, h)
        h = self.encoder(h + self.position, src_key_padding_mask=~available)
        pooled = (h * available[..., None]).sum(1) / available.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1), self.decode(h), pooled


def seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    torch.set_num_threads(4)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_people(run=RUN, role="supervised"):
    reg = json.loads((run / "registry.json").read_text())
    prep = json.loads((run / "preparation.json").read_text())
    if prep["registry"] != reg["fingerprint"]:
        raise ValueError("Wrong preprocessing registry")
    if reg["fingerprint"] != digest({k: v for k, v in reg.items() if k != "fingerprint"}):
        raise ValueError("Registry content changed")
    if len(prep["records"]) != len(reg["records"]) or {r["id"] for r in prep["records"]} != {
        r["id"] for r in reg["records"]
    }:
        raise ValueError("Incomplete feature preparation")
    refs = {r["id"]: r for r in reg["records"]}
    groups = defaultdict(list)
    for m in prep["records"]:
        r = refs[m["id"]]
        if r["role"] != role or m["status"] != "ready":
            continue
        p = run / "cache" / (r["id"] + ".npz")
        if file_hash(p) != m["sha256"]:
            raise ValueError("Feature cache integrity failure")
        with np.load(p) as a:
            groups[r["person"]].append({**r, "x": a["tokens"], "cov": a["cov"], "mask": a["mask"]})
    people = []
    for key, records in sorted(groups.items()):
        if sum(len(r["x"]) for r in records) < 10:
            continue
        if len({r["label"] for r in records}) != 1:
            raise ValueError("Inconsistent person labels")
        people.append(
            {
                "key": key,
                "site": records[0]["site"],
                "y": records[0]["label"],
                "records": records,
                "age": records[0]["age"],
                "sex": records[0]["sex"],
            }
        )
    return people


def flatten(people):
    xs = []
    masks = []
    owner = []
    weights = []
    site_people = Counter(p["site"] for p in people)
    for i, p in enumerate(people):
        for r in p["records"]:
            xs.append(r["x"])
            masks.append(np.tile(r["mask"], (len(r["x"]), 1)))
            owner.extend([i] * len(r["x"]))
            # Equal study, person and session influence regardless of recording duration.
            weights.extend(
                [1 / (site_people[p["site"]] * len(p["records"]) * len(r["x"]))] * len(r["x"])
            )
    x = np.concatenate(xs)
    mask = np.concatenate(masks)
    w = np.asarray(weights, dtype="float32")
    w /= w.mean()
    return x, mask, np.array(owner), w


def scaling(x, mask, w):
    # Bounded temporary arrays and float64 accumulation across the entire archive.
    n = np.zeros((17, 1))
    total = np.zeros((17, 34))
    square = np.zeros((17, 34))
    for i in range(0, len(x), 2048):
        a = x[i : i + 2048].astype("float64")
        ww = (w[i : i + 2048, None] * mask[i : i + 2048])[..., None]
        n += ww.sum(0)
        total += (a * ww).sum(0)
        square += (a * a * ww).sum(0)
    mean = total / np.maximum(n, 1)
    var = np.maximum(square / np.maximum(n, 1) - mean**2, 0)
    return mean.astype("float32"), np.maximum(np.sqrt(var), 0.1).astype("float32")


def tensor_batch(x, mask, indices, mean, std, dev):
    a = np.clip((x[indices] - mean) / std, -12, 12)
    a = np.where(mask[indices, :, None], a, 0)
    return torch.tensor(a, device=dev), torch.tensor(mask[indices], device=dev)


def predict(model, x, mask, owner, people, mean, std, dev):
    model.eval()
    scores = []
    emb = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            xx, mm = tensor_batch(x, mask, slice(start, start + 256), mean, std, dev)
            out, _, e = model(xx, mm)
            scores.extend(out.sigmoid().cpu().numpy())
            emb.append(e.cpu().numpy())
    # Each session has equal influence; all accepted windows within a session contribute.
    scores = np.asarray(scores)
    features = np.concatenate(emb)
    ps = []
    es = []
    offset = 0
    for p in people:
        sr = []
        se = []
        for r in p["records"]:
            n = len(r["x"])
            sr.append(scores[offset : offset + n].mean())
            fe = features[offset : offset + n]
            se.append(np.r_[fe.mean(0), fe.std(0)])
            offset += n
        ps.append(np.mean(sr))
        es.append(np.mean(se, axis=0))
    return np.asarray(ps), np.asarray(es)


def pretrain(run=RUN, benchmark=False):
    if (run / "ssl-lock.json").exists():
        lock = json.loads((run / "ssl-lock.json").read_text())
        if file_hash(run / "ssl.pt") != lock["sha256"]:
            raise ValueError("Pretrained weights changed")
        return
    seed(SEED)
    dev = device()
    people = load_people(run, "auxiliary")
    # Group the complementary AD/FTD releases together by canonical person key.
    train = [p for p in people if int(digest(p["key"])[:8], 16) % 10 != 0]
    valid = [p for p in people if int(digest(p["key"])[:8], 16) % 10 == 0]
    if not train or not valid:
        raise ValueError("Insufficient auxiliary holdout groups")
    x, m, owner, w = flatten(train)
    v, vm, _, vw = flatten(valid)
    mean, std = scaling(x, m, w)
    model = SpectralEncoder().to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    start = time.monotonic()
    rng = np.random.default_rng(SEED)
    total_epochs = 20

    def epoch(xx, mm, ww, training, limit=None):
        model.train(training)
        num = 0.0
        den = 0.0
        order = rng.permutation(len(xx)) if training else np.arange(len(xx))
        # Validation mask is identical at every epoch; no diagnosis is used.
        vrng = np.random.default_rng(SEED + 7)
        for j in range(0, len(order), 128):
            if limit is not None and j >= limit * 128:
                break
            ids = order[j : j + 128]
            a, b = tensor_batch(xx, mm, ids, mean, std, dev)
            hidden = (
                torch.tensor((rng if training else vrng).random((len(ids), 17)) < 0.4, device=dev)
                & b
            )
            # At least one observed channel is always masked; avoid empty loss.
            empty = ~hidden.any(1)
            first = b.to(torch.int64).argmax(1)
            hidden[torch.arange(len(ids), device=dev)[empty], first[empty]] = True
            with torch.set_grad_enabled(training):
                _, reconstruction, _ = model(a, b, hidden)
                per = ((reconstruction - a) ** 2 * hidden[..., None]).sum((1, 2)) / (
                    hidden.sum(1).clamp(min=1) * 34
                )
                wt = torch.tensor(ww[ids], device=dev)
                loss = (per * wt).mean()
                if training:
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optim.step()
            num += float((per.detach() * wt).sum().cpu())
            den += float(wt.sum().cpu())
        return num / den

    if benchmark:
        steps = min(40, (len(x) + 127) // 128)
        epoch(x, m, w, True, steps)
        if dev.type == "mps":
            torch.mps.synchronize()
        elapsed = time.monotonic() - start
        forecast = elapsed / steps * ((len(x) + len(v)) / 128) * total_epochs * 1.5
        save(
            run / "ssl-benchmark.json",
            {
                "device": str(dev),
                "steps": steps,
                "seconds": elapsed,
                "forecast_seconds": forecast,
                "train_windows": len(x),
                "validation_windows": len(v),
            },
        )
        return
    bench = json.loads((run / "ssl-benchmark.json").read_text())
    if bench["forecast_seconds"] > 96 * 3600:
        raise RuntimeError("USER_NOTICE_REQUIRED: pretraining forecast exceeds 96 hours")
    for ep in range(total_epochs):
        loss = epoch(x, m, w, True)
        vl = epoch(v, vm, vw, False)
        history.append({"epoch": ep + 1, "train": loss, "validation": vl})
        if vl < best - 1e-5:
            best = vl
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        save(
            run / "ssl-progress.json",
            {"history": history, "elapsed_seconds": time.monotonic() - start},
        )
        print("SSL", ep + 1, loss, vl, flush=True)
        if stale >= 5:
            break
    torch.save(
        {"model": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
        run / "ssl.pt",
    )
    freeze(
        run / "ssl-lock.json",
        {
            "sha256": file_hash(run / "ssl.pt"),
            "training_people": [p["key"] for p in train],
            "validation_people": [p["key"] for p in valid],
            "history": history,
            "seconds": time.monotonic() - start,
            "no_pd_evaluation_participants": True,
        },
    )


def supervised_fit(train, valid, config, run, epochs=None, seed_value=SEED):
    seed(seed_value)
    dev = device()
    model = SpectralEncoder().to(dev)
    x, m, owner, w = flatten(train)
    mean, std = scaling(x, m, w)
    if config["family"] == "ssl_neural":
        cp = torch.load(run / "ssl.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(cp["model"])
        mean, std = cp["mean"].numpy(), cp["std"].numpy()
        # Fresh diagnostic head: reconstruction pretraining did not train this head.
        model.head[-1].reset_parameters()
    yy = np.array([p["y"] for p in train])
    labels = yy[owner]
    # Equal classes within sites, then equal people and sessions within each class.
    counts = Counter((p["site"], p["y"]) for p in train)
    w = np.array(
        [
            1 / counts[(p["site"], p["y"])] / len(p["records"]) / len(r["x"])
            for p in train
            for r in p["records"]
            for _ in r["x"]
        ],
        dtype="float32",
    )
    w /= w.mean()
    if valid:
        v, vm, vo, _ = flatten(valid)
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    best = -1
    best_ep = 0
    best_state = None
    stale = 0
    rng = np.random.default_rng(seed_value)
    for ep in range(epochs or 40):
        model.train()
        order = rng.permutation(len(x))
        for j in range(0, len(order), 128):
            ids = order[j : j + 128]
            a, b = tensor_batch(x, m, ids, mean, std, dev)
            logits, _, _ = model(a, b)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    torch.tensor(labels[ids], dtype=torch.float32, device=dev),
                    reduction="none",
                )
                * torch.tensor(w[ids], device=dev)
            ).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
        if valid:
            pp, _ = predict(model, v, vm, vo, valid, mean, std, dev)
            score = roc_auc_score([p["y"] for p in valid], pp)
            if score > best + 1e-5:
                best = score
                best_ep = ep + 1
                best_state = copy.deepcopy(
                    {k: z.detach().cpu() for k, z in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
            if stale >= 7:
                break
        else:
            best_ep = ep + 1
    if best_state:
        model.load_state_dict(best_state)
    return model, mean, std, best_ep


if __name__ == "__main__":
    import sys

    pretrain(benchmark="benchmark" in sys.argv)

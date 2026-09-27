"""Aggregate reporting only; individual predictions and models stay under ignored data/."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from pd_pipeline.cohort_v3 import RUN
from pd_pipeline.registry import save
from pd_pipeline.preprocess import file_hash


def main():
    result = json.loads((RUN / "results.json").read_text())
    registry = json.loads((RUN / "registry.json").read_text())
    prep = json.loads((RUN / "preparation.json").read_text())
    outcomes = {r["id"]: r for r in prep["records"]}
    groups = defaultdict(list)
    for r in registry["records"]:
        groups[r["person"]].append(r)
    coverage = defaultdict(Counter)
    for group in groups.values():
        r = group[0]
        key = r["site"] + (
            " / PD" if r["label"] == 1 else " / control" if r["label"] == 0 else " / auxiliary"
        )
        n = sum(outcomes[x["id"]]["windows"] for x in group)
        coverage[key]["assigned_groups"] += 1
        coverage[key]["retained_groups"] += int(n >= 10)
        coverage[key]["accepted_windows"] += n
    reasons = Counter(r.get("reason") or "ready" for r in prep["records"])
    stage_times = {}
    for name in ["preparation-benchmark", "ssl-benchmark", "supervised-benchmark"]:
        stage_times[name] = json.loads((RUN / (name + ".json")).read_text())
    selection = {}
    for p in sorted((RUN / "experiments").glob("*/lock.json")):
        lock = json.loads(p.read_text())
        selection[p.parent.name] = {
            "selected_family": lock["selected_family"],
            "models": {
                name: {k: v for k, v in m.items() if k != "fits"}
                for name, m in lock["models"].items()
            },
        }
    report = {
        "results": result,
        "coverage": dict(coverage),
        "recording_statuses": dict(reasons),
        "archive_usage": registry["ledger"],
        "dataset_exclusions": registry["excluded_datasets"],
        "selection": selection,
        "benchmarks": stage_times,
        "interpretation": registry["history"],
        "protocol_sha256": file_hash(Path("docs/PD_PROTOCOL_V3.md")),
    }
    save(Path("docs/pd-v3-results.json"), report)
    text = [
        "# Expanded PD classification: protocol v3 results",
        "",
        "**Exploratory results. All four PD sites were previously examined.**",
        "",
        "Cross-site performance is primary; within-site performance is reported separately. The selected pipeline chooses model family and configuration using inner training-site validation only. All available compatible durations and sessions contribute, grouped by person.",
        "",
        "## Coverage and archive use",
        "",
    ]
    for key, c in sorted(coverage.items()):
        text.append(
            f"- {key}: {c['retained_groups']}/{c['assigned_groups']} groups retained; {c['accepted_windows'] * 4 / 3600:.2f} accepted recording hours."
        )
    text += [
        "",
        "Auxiliary groups are not guaranteed unique individuals across studies. NeuroEPO unresolved identities use one conservative group. Repeated medication sessions are not independent people.",
        "",
    ]
    for kind, data in result.items():
        text += ["## " + kind.replace("_", " ").title(), ""]
        for family, auc in data["macro_site_auroc"].items():
            text.append(f"- {family}: equal-site mean AUROC {auc:.3f}.")
        for site, models in data["sites"].items():
            text += ["", "### " + site, ""]
            for family, m in models.items():
                ci = m["auroc_95ci"]
                diff = m["paired_auroc_gain_vs_spectral"]
                text.append(
                    f"- {family}: n={m['participants']}; AUROC {m['auroc']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}); sensitivity {m['sensitivity']:.3f}; specificity {m['specificity']:.3f}; PR-AUC {m['pr_auc_average_precision']:.3f}; Brier {m['brier']:.3f}; paired AUROC gain versus spectral {diff['gain']:.3f} (95% CI {diff['ci95'][0]:.3f}-{diff['ci95'][1]:.3f})."
                )
        text.append("")
    text += [
        "## Limits and interpretation",
        "",
        "Intervals resample participants within a site, not sites. Paired comparisons are exploratory and unadjusted for multiplicity. Small cohorts produce uncertain estimates. Different folds use different inner-selected thresholds. These scores do not establish disease diagnosis or symptom tracking.",
        "",
        "V2 used different training populations, shorter recordings and different eligibility. Comparing its headline AUROC with v3 does not isolate a model improvement. V3 family comparisons share participants and folds; the selected-pipeline estimates evaluate the selection procedure. Do not pick an outer-test winner and present its score as unbiased.",
        "",
        "Engineering QC has not received blinded clinical validation. Missing channels, source artifacts, age/sex, medication and acquisition differences remain confound risks. Auxiliary pretraining is an EEG spectral reconstruction task and may not improve disease discrimination.",
        "",
        "The final research models were fitted on all eligible PD/control people after nested evaluation; training-set performance is not a new test result. Fresh-site confirmation remains necessary.",
        "",
        "## Reproducibility",
        "",
        "See [frozen protocol](PD_PROTOCOL_V3.md), aggregate JSON and the local frozen implementation, registry, split, model and preprocessing records. No identities, raw EEG, individual predictions or weights are published. Final report review is required before delivery.",
    ]
    Path("docs/PD_V3_RESULTS.md").write_text("\n".join(text) + "\n")
    save(
        RUN / "aggregate-report-ready.json",
        {
            "status": "awaiting_review_and_pdf",
            "aggregate_sha256": file_hash(Path("docs/pd-v3-results.json")),
        },
    )


if __name__ == "__main__":
    main()

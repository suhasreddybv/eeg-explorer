"""Run with python -m pd_pipeline; all artifacts stay in the local run folder."""

import argparse
import json
from pathlib import Path

from .registry import build, freeze, summary
from .preprocess import DEFAULT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "registry",
            "prepare",
            "benchmark",
            "baseline",
            "train",
            "campaign",
            "evaluate",
            "audit",
            "status",
        ],
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cohort", choices=["internal_test", "ucsd_on", "ucsd_off"], default="internal_test"
    )
    args = parser.parse_args()
    if args.command in {"registry", "prepare", "evaluate"} and args.archive is None:
        parser.error("--archive is required")
    if args.command == "registry":
        registry = build(args.archive)
        freeze(args.run / "registry.json", registry)
        freeze(args.run / "preprocessing.json", DEFAULT)
        from .evaluate import POLICY

        freeze(args.run / "evaluation-policy.json", POLICY)
        result = summary(registry)
    elif args.command == "prepare":
        result = prepare(args.archive, args.run, limit=args.limit)
        result = {"counts": result["counts"], "elapsed_seconds": result["elapsed_seconds"]}
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(args.archive, args.run, args.cohort, args.device)
    elif args.command == "status":
        registry = json.loads((args.run / "registry.json").read_text())
        result = {
            "split_counts": summary(registry),
            "tests": "sealed",
            "compute_policy": "local only; notify before >96h continuous stage",
        }
        for filename in ["preparation.json", "benchmark.json", "baseline.json"]:
            path = args.run / filename
            if path.exists():
                content = json.loads(path.read_text())
                result[filename] = content.get("counts", content)
    else:
        from .experiments import benchmark, baseline, train, campaign, audit

        if args.command == "benchmark":
            result = benchmark(args.run, args.device, args.epochs or 5)
        elif args.command == "baseline":
            result = baseline(args.run)
        elif args.command == "audit":
            result = audit(args.run)
        elif args.command == "campaign":
            result = campaign(args.run, args.device, args.epochs or 100)
        else:
            result = train(args.run, args.device, args.epochs or 100)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

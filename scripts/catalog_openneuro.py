"""Read public S3 object inventories and cohort metadata without downloading signals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx

S3 = "https://s3.amazonaws.com/openneuro.org"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def inventory(client, dataset):
    objects = []
    params = {"list-type": "2", "prefix": dataset + "/", "max-keys": "1000"}
    while True:
        response = client.get(S3, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for obj in root.findall("s:Contents", NS):
            objects.append(
                {
                    "key": obj.findtext("s:Key", namespaces=NS),
                    "size": int(obj.findtext("s:Size", namespaces=NS)),
                    "etag": obj.findtext("s:ETag", namespaces=NS).strip('"'),
                    "modified": obj.findtext("s:LastModified", namespaces=NS),
                }
            )
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            break
        params["continuation-token"] = token
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:

        def inspect(dataset):
            dest = args.output / dataset
            dest.mkdir(exist_ok=True)
            objects = inventory(client, dataset)
            for obj in objects:
                relative = obj["key"].split("/", 1)[1]
                if "/" not in relative and relative in {
                    "README",
                    "README.md",
                    "dataset_description.json",
                    "participants.tsv",
                    "participants.json",
                    "CHANGES",
                    "LICENSE",
                }:
                    response = client.get(S3 + "/" + obj["key"])
                    response.raise_for_status()
                    (dest / relative).write_bytes(response.content)
            eeg = [
                o
                for o in objects
                if "/eeg/" in o["key"]
                and not any(part in o["key"].split("/") for part in ["derivatives", "sourcedata"])
            ]
            result = {
                "dataset": dataset,
                "source": f"https://openneuro.org/datasets/{dataset}",
                "listed_at": datetime.now(timezone.utc).isoformat(),
                "objects": objects,
                "raw_eeg_bytes": sum(o["size"] for o in eeg),
                "raw_eeg_files": len(eeg),
            }
            (dest / "inventory.json").write_text(json.dumps(result, indent=2))
            print(
                f"{dataset}: {len(eeg)} raw EEG files, {result['raw_eeg_bytes'] / 1e9:.3f} GB",
                flush=True,
            )
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(inspect, args.datasets))
    print(f"Total raw EEG: {sum(r['raw_eeg_bytes'] for r in results) / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()

"""Inventory current PhysioNet resources before choosing EEG download scope.

This reads metadata only. It does not accept agreements or download recordings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re
import time

import httpx

INDEX = "https://physionet.org/content/?types=0&types=2"
PATTERN = re.compile(r"\bEEG\b|electroencephal|polysomnog|\bECoG\b", re.I)


def plain(html):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def get(client, url):
    for attempt in range(4):
        try:
            r = client.get(url)
            r.raise_for_status()
            return r.text
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def cards(html):
    result = []
    for part in html.split('<div class="project">')[1:]:
        match = re.search(r'<h2>\s*<a href="(/content/[^\"]+)">(.*?)</a>', part, re.S)
        if match:
            href, title = match.groups()
            result.append(
                {
                    "url": "https://physionet.org" + href,
                    "title": plain(title),
                    "summary": plain(part.split("<hr>")[0]),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "metadata-cache"
    cache.mkdir(exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        first = get(client, INDEX)
        pages = max([1] + list(map(int, re.findall(r"page=(\d+)", first))))
        print(f"Scanning {pages} catalogue pages", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            htmls = [first] + list(
                pool.map(lambda p: get(client, INDEX + f"&page={p}"), range(2, pages + 1))
            )
        resources = {x["url"]: x for html in htmls for x in cards(html)}
        print(f"Reading metadata for {len(resources)} resources", flush=True)

        def inspect(item):
            slug, version = item["url"].rstrip("/").split("/")[-2:]
            path = cache / f"{slug}--{version}.html"
            try:
                html = path.read_text() if path.exists() else get(client, item["url"])
                if not path.exists():
                    path.write_text(html)
                # Keep article text through access/license metadata; exclude other resource links/footer.
                body = html.split('id="files"')[0]
                body_text = plain(body)
                if not PATTERN.search(item["summary"] + " " + body_text):
                    return None
                size = re.search(r"Total uncompressed size:\s*([\d.,]+)\s*([KMGT]?B)", plain(html))
                estimate = None
                if size:
                    estimate = int(
                        float(size[1].replace(",", ""))
                        * {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}[size[2]]
                    )
                access = "open" if "Open Access" in item["summary"] else "restricted"
                license_match = re.search(
                    r'License \(for files\):.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
                )
                if license_match and re.search(
                    r"Restricted|Credentialed|Contributor Review", plain(license_match[2]), re.I
                ):
                    access = "restricted"
                snippets = [
                    body_text[max(0, m.start() - 120) : m.end() + 180]
                    for m in PATTERN.finditer(body_text)
                ]
                return {
                    **item,
                    "slug": slug,
                    "version": version,
                    "access": access,
                    "estimated_bytes": estimate,
                    "license": plain(license_match[2]) if license_match else None,
                    "license_url": license_match[1] if license_match else None,
                    "eeg_mentions": snippets[:12],
                }
            except Exception as e:
                return {**item, "slug": slug, "version": version, "error": str(e)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            candidates = [x for x in pool.map(inspect, resources.values()) if x]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index": INDEX,
        "scope": "Latest versions listed in the full Data and Challenge catalogue; metadata mentions EEG, electroencephalography, polysomnography or ECoG. Candidates require relevance review.",
        "catalogue_pages": pages,
        "resources_scanned": len(resources),
        "candidates": candidates,
    }
    (args.output / "catalogue.json").write_text(json.dumps(output, indent=2))
    print(f"Saved {len(candidates)} candidates to {args.output / 'catalogue.json'}", flush=True)
    for item in candidates:
        print(
            item["slug"],
            item.get("access", "ERROR"),
            item.get("estimated_bytes"),
            item["title"],
            flush=True,
        )


if __name__ == "__main__":
    main()

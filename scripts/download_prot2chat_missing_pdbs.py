#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing Prot2Chat PDB files from AlphaFold DB using accessions in a Prot2Chat-style JSON."
    )
    parser.add_argument(
        "--dataset-json",
        required=True,
        help="Prot2Chat-style JSON file containing items with primaryAccession and pdb.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where downloaded .pdb files will be stored.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of missing files to download for testing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Number of retries for failed downloads.",
    )
    parser.add_argument(
        "--sleep-between-retries",
        type=float,
        default=1.0,
        help="Sleep in seconds between retries.",
    )
    return parser.parse_args()


def load_targets(dataset_json: Path, output_dir: Path) -> list[tuple[str, str]]:
    raw = json.loads(dataset_json.read_text())
    seen: set[str] = set()
    targets: list[tuple[str, str]] = []
    for item in raw:
        accession = item.get("primaryAccession")
        pdb_name = item.get("pdb")
        if not accession or not pdb_name:
            continue
        if pdb_name in seen:
            continue
        seen.add(pdb_name)
        if (output_dir / pdb_name).exists():
            continue
        targets.append((accession, pdb_name))
    return targets


def fetch_json(url: str, timeout: int) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def download_file(url: str, out_path: Path, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    out_path.write_bytes(data)


def resolve_pdb_url(accession: str, timeout: int) -> str:
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    payload = fetch_json(api_url, timeout)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected AlphaFold API payload for {accession}: {type(payload)}")
    item = payload[0]
    if not isinstance(item, dict):
        raise RuntimeError(f"Unexpected AlphaFold API item for {accession}: {type(item)}")
    pdb_url = item.get("pdbUrl") or item.get("pdb_url")
    if not pdb_url:
        raise RuntimeError(f"No pdbUrl returned by AlphaFold API for {accession}")
    return str(pdb_url)


def iter_with_limit(items: list[tuple[str, str]], limit: int) -> Iterable[tuple[str, str]]:
    if limit and limit > 0:
        return items[:limit]
    return items


def main() -> None:
    args = parse_args()
    dataset_json = Path(args.dataset_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(iter_with_limit(load_targets(dataset_json, output_dir), args.limit))
    if not targets:
        print("No missing PDB files to download.")
        return

    print(f"Need to download {len(targets)} missing PDB files into {output_dir}")

    def worker(target: tuple[str, str]) -> tuple[str, str, str]:
        accession, pdb_name = target
        out_path = output_dir / pdb_name
        last_error = ""
        for attempt in range(args.retry + 1):
            try:
                pdb_url = resolve_pdb_url(accession, args.timeout)
                download_file(pdb_url, out_path, args.timeout)
                return accession, pdb_name, "ok"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if out_path.exists():
                    out_path.unlink(missing_ok=True)
                if attempt < args.retry:
                    time.sleep(args.sleep_between_retries)
                    continue
                return accession, pdb_name, last_error
        return accession, pdb_name, last_error

    success = 0
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(worker, target): target for target in targets}
        for idx, future in enumerate(as_completed(future_map), start=1):
            accession, pdb_name, status = future.result()
            if status == "ok":
                success += 1
                print(f"[{idx}/{len(targets)}] downloaded {pdb_name}")
            else:
                failures.append((accession, pdb_name, status))
                print(f"[{idx}/{len(targets)}] failed {pdb_name}: {status}", file=sys.stderr)

    print(f"Download complete: success={success} failed={len(failures)}")
    if failures:
        failure_path = output_dir / "_download_failures.json"
        failure_path.write_text(
            json.dumps(
                [
                    {"primaryAccession": accession, "pdb": pdb_name, "error": error}
                    for accession, pdb_name, error in failures
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Failure details written to {failure_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

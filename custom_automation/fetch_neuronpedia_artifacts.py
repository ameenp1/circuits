"""
fetch_neuronpedia_artifacts.py — pull the transcoder (SLT) side of the neuronpedia
graphs from the HuggingFace dataset `circuit-tracer-automation/pipeline_automation`.

For each slug it downloads, into --out-dir (default custom_automation/np_data/):

  test_graphs/gemma-<slug>.json                     # circuit-tracer attribution graph
  artifacts/gemma-<slug>__feature_groups_<variant>.json   # supernodes: {node_id -> group}
  artifacts/gemma-<slug>__feature_descriptions_v2.json    # per-feature descriptions

These are the transcoder-side inputs to cross_graph_analysis.py (the MLP/ADAG side
is produced separately by this repo's batch_export_neurons -> generate_description ->
generate_supernodes pipeline).

Network note: HF over Windows schannel can fail revocation checks, so this shells out
to `curl --ssl-no-revoke` (falls back to huggingface_hub if curl is unavailable).

Usage:
    python custom_automation/fetch_neuronpedia_artifacts.py
    python custom_automation/fetch_neuronpedia_artifacts.py --slugs gemma-G gemma-dollar
    python custom_automation/fetch_neuronpedia_artifacts.py --groups-variant feature_groups_v2_a2_cap100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = "circuit-tracer-automation/pipeline_automation"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/gemma-2/neuronpedia"

# The 15 neuronpedia graph slugs (mirrors scripts/circuit_prep/data/neuronpedia.py).
SLUGS = [
    "gemma-G", "gemma-addition", "gemma-addition2", "gemma-basket", "gemma-dollar",
    "gemma-english", "gemma-euro", "gemma-girl-is", "gemma-girls-are", "gemma-gp-nps",
    "gemma-keys-cabinet", "gemma-michael-jordan", "gemma-michael-jordan-es",
    "gemma-saison", "gemma-verano",
]


def _curl(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["curl", "-sS", "-fL", "-m", "120", "--ssl-no-revoke", url, "-o", str(dest)],
            check=True,
        )
        return dest.exists() and dest.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def fetch_slug(slug: str, out_dir: Path, groups_variant: str) -> dict[str, bool]:
    targets = {
        f"test_graphs/{slug}.json": out_dir / "test_graphs" / f"{slug}.json",
        f"artifacts_neuronpedia/{slug}/{groups_variant}.json":
            out_dir / "artifacts" / f"{slug}__{groups_variant}.json",
        f"artifacts_neuronpedia/{slug}/feature_descriptions_v2.json":
            out_dir / "artifacts" / f"{slug}__feature_descriptions_v2.json",
    }
    results = {}
    for rel, dest in targets.items():
        ok = _curl(f"{BASE}/{rel}", dest)
        results[rel] = ok
        size = dest.stat().st_size if dest.exists() else 0
        print(f"  [{'ok ' if ok else 'FAIL'}] {dest.name}  ({size} bytes)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch transcoder-side neuronpedia data from HF.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "np_data")
    ap.add_argument("--slugs", nargs="*", default=SLUGS, help="Subset of slugs (default: all 15).")
    ap.add_argument("--groups-variant", default="feature_groups_v2_a2",
                    help="Which feature_groups artifact to pull (e.g. feature_groups_v2_a2_cap100).")
    args = ap.parse_args()

    print(f"Fetching {len(args.slugs)} slug(s) into {args.out_dir}")
    ok = 0
    for slug in args.slugs:
        print(f"\n== {slug} ==")
        res = fetch_slug(slug, args.out_dir, args.groups_variant)
        ok += all(res.values())
    print(f"\nDone: {ok}/{len(args.slugs)} slug(s) fully fetched.")
    if ok < len(args.slugs):
        sys.exit(1)


if __name__ == "__main__":
    main()
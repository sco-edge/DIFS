"""
build_diffusiondb_pool.py

Builds the DIFS_OC_v6 prompt pool from the DiffusionDB 2M metadata table.

Pipeline:
  1. Read `prompt` + `prompt_nsfw` (and support columns) from metadata.parquet.
  2. Drop records with prompt_nsfw >= NSFW_THRESHOLD (default 0.1).
  3. Drop duplicate prompts (first occurrence wins).
  4. Deterministically sample POOL_SIZE unique prompts (seed=SEED).
  5. Write a COCO-schema JSON (`{"annotations": [{"caption": ...}, ...]}`)
     so the existing client loader `extract_coco_prompts()` works unchanged.

Provenance notes are echoed to stdout and written to DATASET.md.

How to run
    Recommended (downloads metadata.parquet first, then builds the pool):
        ./run_all_experiments.sh --setup

    Directly, from the project root:
        python src/scripts/build_diffusiondb_pool.py \
            --parquet results/diffusiondb_data/metadata.parquet \
            --out annotations/diffusiondb_prompts.json

    Options:
        --nsfw-threshold FLOAT   NSFW probability cutoff (default: 0.1)
        --pool-size N            Unique prompts to sample (default: 10000)
        --seed N                 Sampling seed (default: 42)

    The output JSON uses the COCO schema so the existing client prompt
    loader works unchanged.  It is safe to re-run; the script overwrites
    the output deterministically for the same seed.
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

NSFW_THRESHOLD = 0.1
POOL_SIZE = 10_000
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True,
                        help="Downloaded DiffusionDB metadata.parquet")
    parser.add_argument("--out", type=str, default="annotations/diffusiondb_prompts.json",
                        help="Output COCO-schema JSON")
    parser.add_argument("--nsfw-threshold", type=float, default=NSFW_THRESHOLD)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    out_path = Path(args.out)

    print(f"[POOL] Reading {parquet_path}")
    table = pq.read_table(parquet_path, columns=["prompt", "prompt_nsfw"])
    prompts_src = table.column("prompt").to_pylist()
    nsfw_src = table.column("prompt_nsfw").to_pylist()
    total = len(prompts_src)
    print(f"[POOL] Total records: {total}")

    # Step 2: NSFW filter
    kept_nsfw = [
        p for p, n in zip(prompts_src, nsfw_src)
        if p is not None and n is not None and n < args.nsfw_threshold
    ]
    print(f"[POOL] After NSFW filter (top {args.nsfw_threshold}): {len(kept_nsfw)}")

    # Step 3: dedupe preserving first occurrence
    seen = set()
    unique = []
    for p in kept_nsfw:
        lp = p.strip()
        if lp and lp not in seen:
            seen.add(lp)
            unique.append(lp)
    print(f"[POOL] After dedupe: {len(unique)} unique prompts")

    # Step 4: deterministic sample
    rng = random.Random(args.seed)
    sample = rng.sample(unique, min(args.pool_size, len(unique)))
    print(f"[POOL] Sampled {len(sample)} prompts (seed={args.seed})")

    # Step 5: write COCO-schema JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"annotations": [{"caption": p} for p in sample]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[POOL] Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Sample stats
    lens = sorted(len(p.split()) for p in sample)
    print(f"[POOL] Pool word-count: min={lens[0]} median={lens[len(lens)//2]} "
          f"max={lens[-1]}")
    n70 = sum(1 for l in lens if l >= 20)
    print(f"[POOL] Prompts with >=20 words: {n70} ({100.0 * n70 / len(sample):.1f}%)")

    # Provenance note
    note = (
        f"# DiffusionDB prompt pool — DIFS_OC_v6\n\n"
        f"- Source: HF poloclub/diffusiondb `metadata.parquet` (2,000,000 rows)\n"
        f"- Downloaded: {parquet_path.resolve()} ({parquet_path.stat().st_size / 1e6:.1f} MB)\n"
        f"- NSFW filter: dropped records with `prompt_nsfw >= {args.nsfw_threshold}`\n"
        f"- Dedupe: first-occurrence on trimmed prompt text\n"
        f"- Pool size: {len(sample)} unique prompts\n"
        f"- Seed: {args.seed}\n"
        f"- Output: {out_path.resolve()}\n"
        f"- Word-count stats: min={lens[0]}, median={lens[len(lens)//2]}, max={lens[-1]};\n"
        f"-   >=20-word prompts: {n70} ({100.0 * n70 / len(sample):.1f}%)\n"
    )
    md_path = out_path.parent.parent / "DATASET.md"
    # DATASET.md lives at project root; adjust:
    md_path = Path(__file__).resolve().parent.parent.parent / "DATASET.md"
    md_path.write_text(note)
    print(f"[POOL] Wrote {md_path}")


if __name__ == "__main__":
    main()
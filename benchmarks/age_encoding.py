#!/usr/bin/env python3
"""Compares the two age codes supported by anyface_pp.models.age_estimator.

The age head predicts a set of independent binary units. How those units are
turned into a number decides how a unit-level mistake shows up in the answer,
and that is a property of the code alone -- no dataset, no weights and no GPU
are involved, so this runs anywhere and is checked in CI.

Two things are measured:

1. How hard each unit is to learn, via how many times its target flips as age
   goes 0 -> MAX_AGE. A unit whose target alternates is asking the network for
   something it cannot see in a face.

2. What a unit-level mistake costs in years, by flipping each unit
   independently with probability p and decoding.

The comparison holds p equal across the two codes. That favours the binary
code, which has 8 units against the ordinal code's 100 and so suffers fewer
wrong units in absolute terms; in a trained model the ordinal units would also
be the easier ones to get right. The point is that the binary code loses even
with that handicap in its favour.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyface_pp.models.age_estimator import (  # noqa: E402
    ENCODINGS, MAX_AGE, decode_age, encode_age, num_outputs)


def target_flips(encoding):
    """How many times each unit's target changes as age goes 0 -> MAX_AGE."""
    codes = encode_age(np.arange(MAX_AGE + 1), encoding)
    return np.abs(np.diff(codes, axis=0)).sum(0).astype(int)


def worst_single_unit_error(encoding):
    """Largest age error a single wrong unit can cause, over ages 0..MAX_AGE."""
    ages = np.arange(MAX_AGE + 1)
    codes = encode_age(ages, encoding)
    worst = 0.0
    for unit in range(codes.shape[1]):
        flipped = codes.copy()
        flipped[:, unit] = 1.0 - flipped[:, unit]
        worst = max(worst, float(np.abs(decode_age(flipped, encoding) - ages).max()))
    return worst


def error_under_unit_noise(encoding, p, n, rng):
    """Decode ages whose units were each flipped independently with prob *p*."""
    ages = rng.integers(0, MAX_AGE + 1, size=n)
    codes = encode_age(ages, encoding)
    noisy = np.where(rng.random(codes.shape) < p, 1.0 - codes, codes)
    err = np.abs(decode_age(noisy, encoding) - ages)
    return {"mae": float(err.mean()),
            "p99": float(np.percentile(err, 99)),
            "max": float(err.max()),
            "frac_over_10y": float((err > 10).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="ages sampled per rate")
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.001, 0.01, 0.05, 0.10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=str(ROOT / "benchmarks" / "age_encoding.json"))
    args = ap.parse_args()

    summary = {"max_age": MAX_AGE, "n": args.n, "encodings": {}}

    print(f"Age codes over ages 0..{MAX_AGE}\n")
    print(f"{'encoding':<10}{'units':>7}{'worst 1-unit error':>21}")
    for enc in ENCODINGS:
        worst = worst_single_unit_error(enc)
        print(f"{enc:<10}{num_outputs(enc):>7}{worst:>18.0f} y")
        summary["encodings"][enc] = {"units": num_outputs(enc),
                                     "worst_single_unit_error": worst,
                                     "target_flips": target_flips(enc).tolist(),
                                     "noise": {}}

    print("\nTimes each unit's target flips across the age range")
    print("(a unit that alternates is asking the network to see something")
    print(" that is not in the face -- the last binary bit is age parity):")
    for enc in ENCODINGS:
        f = summary["encodings"][enc]["target_flips"]
        shown = f if len(f) <= 12 else f"all {sorted(set(f))} ({len(f)} units)"
        print(f"  {enc:<9} {shown}")

    rng = np.random.default_rng(args.seed)
    print(f"\nAge error when each unit is independently wrong with probability p"
          f"  ({args.n:,} ages per cell):")
    print(f"{'p':>7}{'':4}{'ordinal MAE':>12}{'binary MAE':>12}"
          f"{'':4}{'ordinal p99':>12}{'binary p99':>12}"
          f"{'':4}{'ord >10y':>9}{'bin >10y':>9}")
    for p in args.rates:
        res = {enc: error_under_unit_noise(enc, p, args.n, rng) for enc in ENCODINGS}
        for enc in ENCODINGS:
            summary["encodings"][enc]["noise"][str(p)] = res[enc]
        print(f"{p:>7.3f}{'':4}"
              f"{res['ordinal']['mae']:>12.2f}{res['binary']['mae']:>12.2f}{'':4}"
              f"{res['ordinal']['p99']:>12.0f}{res['binary']['p99']:>12.0f}{'':4}"
              f"{res['ordinal']['frac_over_10y']:>8.1%}"
              f"{res['binary']['frac_over_10y']:>9.1%}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

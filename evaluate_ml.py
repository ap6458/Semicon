
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from localize_ml import localize


def gt(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    p = d["target_center_search"]
    return float(p["x"]), float(p["y"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--start", type=int, default=450)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.dataset)
    samples = sorted(root.glob("sample_*"))
    samples = samples[args.start:args.end]

    errors = []
    start = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        tx, ty = gt(sample / "ground_truth.json")

        result = localize(
            sample / "reference.png",
            sample / "search.png",
            "match_ranker.pt"
        )

        e = math.hypot(
            result["x"] - tx,
            result["y"] - ty
        )
        errors.append(e)

        print(
            f"[{i:02d}/{len(samples):02d}] "
            f"{sample.name}: "
            f"pred=({result['x']:.1f},{result['y']:.1f}) "
            f"gt=({tx:.1f},{ty:.1f}) "
            f"error={e:.2f}px"
        )

    if not errors:
        raise RuntimeError("No samples evaluated.")

    errors.sort()

    def pct(p):
        idx = round((p/100)*(len(errors)-1))
        return errors[idx]

    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"samples        : {len(errors)}")
    print(f"mean error     : {sum(errors)/len(errors):.3f}px")
    print(f"median error   : {statistics.median(errors):.3f}px")
    print(f"P90            : {pct(90):.3f}px")
    print(f"P95            : {pct(95):.3f}px")

    for t in [1, 2, 5, 10, 20, 50]:
        acc = 100 * sum(e <= t for e in errors) / len(errors)
        print(f"<= {t:2d}px        : {acc:6.2f}%")

    print(f"runtime        : {elapsed:.2f}s")
    print(f"per sample     : {elapsed/len(errors):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

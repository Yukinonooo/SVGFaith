"""Pairwise accuracy for SVGFaith score files.

Arena accuracy is the fraction of wins over all pairs. Each Diagnostic
direction is the unweighted mean of its eight axis accuracies. A tie receives
no credit. A difference is a win only when it is strictly larger than tau.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


AXES = (
    "negation",
    "attribute_binding",
    "comparative_size",
    "spatial_relation",
    "object_identity",
    "counting",
    "part_completeness",
    "dense_long_prompt",
)


def outcome(score_a: float, score_b: float, tau: float) -> str:
    delta = score_a - score_b
    epsilon = 1e-12
    if delta > tau + epsilon:
        return "win"
    if delta < -tau - epsilon:
        return "loss"
    return "tie"


def counts(rows: list[dict], tau: float) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for row in rows:
        label = outcome(float(row["score_a"]), float(row["score_b"]), tau)
        if label == "win":
            wins += 1
        elif label == "loss":
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def accuracy(rows: list[dict], tau: float) -> float:
    if not rows:
        return math.nan
    wins, _, _ = counts(rows, tau)
    return wins / len(rows)


def equal_axis_accuracy(rows: list[dict], tau: float) -> float:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["axis"])].append(row)
    missing = [axis for axis in AXES if axis not in grouped]
    if missing:
        raise ValueError(f"missing diagnostic axes: {missing}")
    return sum(accuracy(grouped[axis], tau) for axis in AXES) / len(AXES)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def report(path: Path, tau: float) -> None:
    rows = read_jsonl(path)
    dataset = rows[0].get("dataset", path.stem)
    direction = rows[0].get("direction", "")
    wins, ties, losses = counts(rows, tau)
    micro = wins / len(rows)
    print(
        f"{dataset} {direction}: n={len(rows)} "
        f"win/tie/loss={wins}/{ties}/{losses} micro={micro:.6f}"
    )
    if dataset == "SVGFaith-Diagnostic":
        print(f"  equal-axis accuracy={equal_axis_accuracy(rows, tau):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--tau", type=float, default=0.01)
    args = parser.parse_args()
    for path in args.scores:
        report(path, args.tau)


if __name__ == "__main__":
    main()

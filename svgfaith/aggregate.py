"""Unique-ancestor aggregation from SVGFaith-Score.

A node score is discounted by each distinct ancestor once:

    g_i = q_i * product_{j in Anc(i)} ((1 - alpha) + alpha * q_j)
    S = mean_i g_i

Alpha is 0.5 in every reported experiment.
"""

from __future__ import annotations


def unique_ancestors(parents: list[list[int]]) -> list[set[int]]:
    """Return the distinct ancestor indices of each node.

    ``parents[i]`` lists the parent indices of node ``i``. Parent links must
    point at earlier nodes. Cycles are ignored rather than expanded forever.
    """
    memo: dict[int, set[int]] = {}

    def walk(index: int, visiting: frozenset[int]) -> set[int]:
        if index in memo:
            return set(memo[index])
        if index in visiting:
            return set()
        found: set[int] = set()
        next_visiting = visiting | {index}
        for parent in parents[index]:
            if parent < 0 or parent >= len(parents):
                continue
            found.add(parent)
            found.update(walk(parent, next_visiting))
        memo[index] = frozenset(found)
        return found

    return [walk(index, frozenset()) for index in range(len(parents))]


def unique_ancestor_score(
    values: list[float],
    parents: list[list[int]],
    alpha: float = 0.5,
) -> float:
    """Aggregate requirement probabilities into one image score in ``[0, 1]``."""
    if not values:
        return 0.0
    if len(values) != len(parents):
        raise ValueError("values and parents must have the same length")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ancestors = unique_ancestors(parents)
    total = 0.0
    for value, ancestor_ids in zip(values, ancestors):
        gate = 1.0
        for ancestor in ancestor_ids:
            gate *= (1.0 - alpha) + alpha * values[ancestor]
        total += value * gate
    return total / len(values)

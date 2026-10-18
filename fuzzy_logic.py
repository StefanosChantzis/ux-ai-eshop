"""Mamdani FLS adapted from the supplied course notes (not copied verbatim).

NumPy membership functions; AND=min, implication=min, aggregation=max,
centroid on a uniform 0..100 grid. Shoulder endpoints and repeated parameters
are handled explicitly, unlike the unguarded divisions in the teaching demo.
No user-study responses or generator's latent preferences enter this system.
"""
from itertools import product
import numpy as np

TERMS = ("Low", "Medium", "High")
OUTPUT_TERMS = ("Very low", "Low", "Medium", "High", "Very high")
INPUT_SETS = ((0, 0, 25, 50), (25, 50, 75), (50, 75, 100, 100))
OUTPUT_SETS = ((0, 0, 10, 25), (10, 25, 40), (30, 50, 70),
               (60, 75, 90), (75, 90, 100, 100))
# Axes: preference, price compatibility, popularity; values index OUTPUT_TERMS.
RULE_TABLE = np.array([
    [[0, 0, 1], [0, 1, 1], [1, 1, 2]],
    [[0, 1, 1], [1, 2, 2], [2, 2, 3]],
    [[1, 1, 2], [2, 3, 3], [3, 4, 4]],
], dtype=int)
RULES = tuple((a, b, c, int(RULE_TABLE[a, b, c]))
              for a, b, c in product(range(3), repeat=3))
OUTPUT_GRID = np.linspace(0, 100, 501)


def membership(x, params):
    """Triangle/trapezoid, including closed left/right shoulder endpoints."""
    x = np.asarray(x, dtype=float)
    if len(params) == 3:
        a, b, d = params
        c = b
    else:
        a, b, c, d = params
    if not a <= b <= c <= d or a == d:
        raise ValueError("Membership parameters must be ordered and span a range")
    left = np.ones_like(x) if a == b else np.clip((x - a) / (b - a), 0, 1)
    right = np.ones_like(x) if c == d else np.clip((d - x) / (d - c), 0, 1)
    return np.where((x < a) | (x > d), 0.0, np.minimum(left, right))


def input_memberships(values, shift=0.0):
    # A prespecified +/-5 point sensitivity scenario translates the input
    # partitions; clipping keeps both shoulders covered at the universe limits.
    x = np.clip(np.asarray(values, dtype=float) - shift, 0, 100)
    return np.stack([membership(x, p) for p in INPUT_SETS], axis=-1)


def infer(inputs, *, shift=0.0, trace=False):
    """One row per product, columns preference/price/popularity in [0,100]."""
    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite product-by-three input matrix")
    if np.any((x < 0) | (x > 100)):
        raise ValueError("Fuzzy inputs must lie in [0,100]")
    mu = input_memberships(x, shift)
    strength = np.stack([np.minimum.reduce([mu[:, 0, a], mu[:, 1, b], mu[:, 2, c]])
                         for a, b, c, _ in RULES], axis=1)
    # Aggregate by consequent before expanding to the output grid (same max-min
    # result as expanding all 27 rules, with much less temporary memory).
    levels = np.zeros((len(x), 5))
    for j, (_, _, _, out) in enumerate(RULES):
        levels[:, out] = np.maximum(levels[:, out], strength[:, j])
    outputs = np.stack([membership(OUTPUT_GRID, p) for p in OUTPUT_SETS])
    aggregate = np.maximum.reduce([np.minimum(levels[:, j, None], outputs[j])
                                   for j in range(5)])
    denominator = aggregate.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Uncovered fuzzy input: rule base must fire for every item")
    scores = (aggregate @ OUTPUT_GRID) / denominator
    if trace:
        return scores, dict(memberships=mu, firing=strength, aggregate=aggregate)
    return scores

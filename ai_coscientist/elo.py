"""Elo rating calculations and tournament pairing strategies.

Provides a pure ``calculate_elo_update`` function used by
``Hypothesis.update_elo`` and helpers that generate match
pairings for different tournament modes (random, round-robin,
swiss).
"""

import itertools
import math
import random
from typing import (
    List,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")

_VALID_MODES = ("random", "round_robin", "swiss")


def calculate_elo_update(
    rating: int,
    opponent_rating: int,
    win: bool,
    k_factor: int = 32,
) -> int:
    """Return the new Elo rating after a single match.

    Args:
        rating: Current Elo rating of the player.
        opponent_rating: Elo rating of the opponent.
        win: Whether the player won.
        k_factor: K-factor controlling update magnitude.

    Returns:
        Updated Elo rating (integer).
    """
    expected = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
    actual = 1.0 if win else 0.0
    return rating + int(k_factor * (actual - expected))


# -- pairing helpers ------------------------------------------------


def random_pairs(
    items: Sequence[T],
    rounds: int,
    rng: random.Random | None = None,
) -> List[Tuple[int, int]]:
    """Generate *rounds* random pairings as index tuples.

    Each pairing picks two distinct indices uniformly at random.
    """
    _rng = rng or random.Random()
    n = len(items)
    if n < 2:
        return []
    pairs: List[Tuple[int, int]] = []
    for _ in range(rounds):
        a, b = _rng.sample(range(n), 2)
        pairs.append((a, b))
    return pairs


def round_robin_pairs(
    items: Sequence[T],
) -> List[Tuple[int, int]]:
    """Return every unique pairing (n*(n-1)/2 matches)."""
    n = len(items)
    if n < 2:
        return []
    return list(itertools.combinations(range(n), 2))


def swiss_pairs(
    items: Sequence[T],
    ratings: Sequence[int],
    rng: random.Random | None = None,
) -> List[Tuple[int, int]]:
    """One round of Swiss-style pairing.

    Items are sorted by *ratings* (descending) and adjacent
    entries are paired.  If the count is odd the last item
    gets a bye (no pairing).
    """
    n = len(items)
    if n < 2:
        return []
    _rng = rng or random.Random()
    ranked = sorted(range(n), key=lambda i: ratings[i], reverse=True)  # fmt: skip
    pairs: List[Tuple[int, int]] = []
    i = 0
    while i + 1 < len(ranked):
        pairs.append((ranked[i], ranked[i + 1]))
        i += 2
    return pairs


def swiss_rounds(n: int) -> int:
    """Number of rounds for Swiss tournament with *n* items."""
    if n < 2:
        return 0
    return math.ceil(math.log2(n))


def validate_tournament_mode(mode: str) -> str:
    """Raise ``ValueError`` if *mode* is not recognised."""
    if mode not in _VALID_MODES:
        raise ValueError(
            f"tournament_mode must be one of "
            f"{_VALID_MODES}, got '{mode}'"
        )
    return mode

"""Pure-Python compatibility fallback for ``openreview-py`` on Python 3.13.

The native ``editdistance`` dependency does not currently publish a Python
3.13 Windows wheel.  OpenReview imports it at package load time but only uses
the ``eval`` function in venue-administration duplicate checks.  The retrieval
client used by this project does not call that path.
"""

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def eval(left: Sequence[T], right: Sequence[T]) -> int:
    """Return the Levenshtein edit distance between two sequences."""
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]

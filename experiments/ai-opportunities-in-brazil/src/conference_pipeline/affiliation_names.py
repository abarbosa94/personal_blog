"""Conservative normalization for comparing noisy affiliation labels."""

from __future__ import annotations

import re
import unicodedata


_LEADING_ANNOTATION = re.compile(
    r"""
    ^\s*
    (?:[*†‡]+\s*)?
    (?:
        equal(?:ly)?\s+contribut(?:ion|ed)
        |co[- ]?first\s+authors?
        |joint\s+(?:project\s+)?leads?
    )
    \s*[:;,.()\-]*\s*
    (?:\d+\s*)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def canonical_affiliation_name(value: str) -> str:
    """Remove front-matter annotations without guessing institution aliases."""

    value = _LEADING_ANNOTATION.sub("", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

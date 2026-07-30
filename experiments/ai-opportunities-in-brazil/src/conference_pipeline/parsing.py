from __future__ import annotations

import html
import re
import unicodedata


def normalize_space(value: str) -> str:
    return " ".join(html.unescape(value).split())


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_space(value))
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_marks.lower()).strip()


def split_authors(value: str) -> tuple[str, ...]:
    cleaned = normalize_space(value)
    parts = re.split(r"\s*(?:,|;|\band\b)\s*", cleaned)
    return tuple(part for part in parts if part)


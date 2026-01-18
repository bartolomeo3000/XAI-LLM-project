from __future__ import annotations
import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DECIMAL_COMMA_RE = re.compile(r"(\d),(\d)")  # 3,14 -> 3.14

def fold_diacritics(text: str) -> str:
    """Remove diacritics (ą->a, ł->l etc.) in a Unicode-safe way."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_text(
    text: str,
    *,
    diacritics: bool = False,
    normalize_decimal_comma: bool = True
) -> str:
    """
    Normalize text for matching:
    - lowercase
    - optionally unify decimal comma to dot (3,14 -> 3.14)
    - replace punctuation with spaces
    - collapse whitespace
    Optionally folds diacritics.
    """
    if text is None:
        return ""
    t = text.strip().lower()
    if diacritics:
        t = fold_diacritics(t)
    if normalize_decimal_comma:
        t = _DECIMAL_COMMA_RE.sub(r"\1.\2", t)
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

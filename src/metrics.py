from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, Optional

_ABSTAIN_PATTERNS = [
    r"\bnie\s+wiem\b",
    r"\bnie\s+j(e|ę)stem\s+pewien\b",
    r"\bnie\s+j(e|ę)stem\s+pewna\b",
    r"\bnie\s+mam\s+pewno(ś|s)ci\b",
    r"\btrudno\s+powiedzie(ć|c)\b",
    r"\bnie\s+potrafi(ę|e)\s+odpowiedzie(ć|c)\b",
]
_ABSTAIN_RE = re.compile("|".join(f"(?:{p})" for p in _ABSTAIN_PATTERNS), flags=re.IGNORECASE)

def is_abstain(answer: str) -> bool:
    if not answer:
        return True
    return bool(_ABSTAIN_RE.search(answer.strip()))

@dataclass
class Summary:
    n: int
    accuracy: float
    abstain_rate: float
    hallucination_rate: float  # incorrect & not abstain
    incorrect_rate: float

def summarize(outcomes: Iterable[tuple[bool, bool]]) -> Summary:
    """outcomes: iterable of (correct, abstain)"""
    outcomes = list(outcomes)
    n = len(outcomes)
    if n == 0:
        return Summary(n=0, accuracy=0.0, abstain_rate=0.0, hallucination_rate=0.0, incorrect_rate=0.0)

    correct = sum(1 for c, _ in outcomes if c)
    abst = sum(1 for _, a in outcomes if a)
    incorrect = n - correct
    halluc = sum(1 for c, a in outcomes if (not c) and (not a))

    return Summary(
        n=n,
        accuracy=correct / n,
        abstain_rate=abst / n,
        hallucination_rate=halluc / n,
        incorrect_rate=incorrect / n,
    )

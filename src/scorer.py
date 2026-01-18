from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .normalize import normalize_text

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"^[+-]?(\d+([\.]\d+)?)$")  # after normalization decimal comma becomes dot

def _is_numeric_keyword(kw_norm: str) -> bool:
    return bool(_NUM_RE.match(kw_norm))

def _word_boundary_pattern(phrase_norm: str) -> re.Pattern:
    parts = [re.escape(p) for p in phrase_norm.split()]
    pat = r"\b" + r"\s+".join(parts) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)

def keyword_present(keyword: str, text_norm: str) -> bool:
    """Check if a keyword variant is present in already-normalized text."""
    kw_norm = normalize_text(keyword, diacritics=False)
    if not kw_norm:
        return False

    if _is_numeric_keyword(kw_norm):
        pat = re.compile(rf"\b{re.escape(kw_norm)}\b")
        return bool(pat.search(text_norm))

    if " " in kw_norm:
        return bool(_word_boundary_pattern(kw_norm).search(text_norm))

    pat = re.compile(rf"\b{re.escape(kw_norm)}\b")
    return bool(pat.search(text_norm))

@dataclass
class GroupMatch:
    group: List[str]
    matched_variant: Optional[str]  # which variant satisfied the group

@dataclass
class ScoreResult:
    correct: bool
    matched_groups: List[GroupMatch]
    missing_groups: List[List[str]]

def score_answer(answer: str, keyword_groups: List[List[str]], *, fold_diacritics: bool = True) -> ScoreResult:
    """
    Correct iff every group has at least one matching variant (OR within group, AND across groups).
    """
    logger.debug(f"Scoring answer: {answer[:50]}..." if len(answer) > 50 else f"Scoring answer: {answer}")
    logger.debug(f"Keyword groups: {keyword_groups}")
    
    ans_norm = normalize_text(answer, diacritics=False)
    ans_fold = normalize_text(answer, diacritics=fold_diacritics) if fold_diacritics else ""

    matched_groups: List[GroupMatch] = []
    missing_groups: List[List[str]] = []

    for group in keyword_groups:
        matched_variant: Optional[str] = None
        for var in group:
            var_norm = normalize_text(var, diacritics=False)
            ok = keyword_present(var_norm, ans_norm)
            if not ok and fold_diacritics:
                var_fold = normalize_text(var, diacritics=True)
                ok = keyword_present(var_fold, ans_fold)
            if ok:
                matched_variant = var
                logger.debug(f"Matched keyword variant '{var}' from group {group}")
                break
        if matched_variant is None:
            missing_groups.append(group)
            logger.debug(f"No match found for group {group}")
        matched_groups.append(GroupMatch(group=group, matched_variant=matched_variant))
    
    result = ScoreResult(correct=(len(missing_groups) == 0), matched_groups=matched_groups, missing_groups=missing_groups)
    logger.debug(f"Score result: correct={result.correct}, missing_groups={missing_groups}")
    return result

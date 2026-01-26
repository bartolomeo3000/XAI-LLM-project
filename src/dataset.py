from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    keyword_groups: List[List[str]]  # OR within group, AND across groups
    expected_answer: str = ""  # Simple expected answer for LLM judge

def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    logger.debug(f"Loading JSONL from {p}")
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on line {i} in {p}: {e}")
                raise ValueError(f"Invalid JSON on line {i} in {p}: {e}") from e
            rows.append(obj)
    logger.debug(f"Loaded {len(rows)} records from {p}")
    return rows

def _validate_groups(obj: dict, _id: str) -> List[List[str]]:
    # Preferred: keyword_groups
    if "keyword_groups" in obj and obj["keyword_groups"] is not None:
        kg = obj["keyword_groups"]
        if not isinstance(kg, list) or not kg:
            raise ValueError(f"'keyword_groups' must be a non-empty list at id={_id}")
        groups: List[List[str]] = []
        for g in kg:
            if not isinstance(g, list) or not all(isinstance(x, str) for x in g):
                raise ValueError(f"Each group in 'keyword_groups' must be list[str] at id={_id}")
            cleaned = [x.strip() for x in g if x.strip()]
            if not cleaned:
                raise ValueError(f"Empty group in 'keyword_groups' at id={_id}")
            groups.append(cleaned)
        return groups

    # Backward-compatible: keywords (AND)
    if "keywords" in obj and obj["keywords"] is not None:
        kw = obj["keywords"]
        if not isinstance(kw, list) or not all(isinstance(x, str) for x in kw):
            raise ValueError(f"'keywords' must be a list[str] at id={_id}")
        cleaned = [x.strip() for x in kw if x.strip()]
        if not cleaned:
            raise ValueError(f"'keywords' is empty at id={_id}")
        return [[x] for x in cleaned]

    # If neither keyword_groups nor keywords, return empty list (no scoring possible)
    logger.warning(f"No 'keyword_groups' or 'keywords' at id={_id}, scoring will be skipped")
    return []

def load_questions(path: str | Path) -> list[QAItem]:
    logger.info(f"Loading questions from {path}")
    rows = load_jsonl(path)
    items: list[QAItem] = []
    seen_ids: set[str] = set()
    for idx, r in enumerate(rows, start=1):
        if "id" not in r:
            logger.error(f"Missing 'id' at record #{idx}")
            raise ValueError(f"Missing 'id' at record #{idx}")
        if "question" not in r:
            logger.error(f"Missing 'question' at record #{idx} (id={r.get('id')})")
            raise ValueError(f"Missing 'question' at record #{idx} (id={r.get('id')})")
        _id = str(r["id"])
        if _id in seen_ids:
            logger.error(f"Duplicate id={_id}")
            raise ValueError(f"Duplicate id={_id}")
        seen_ids.add(_id)

        q = str(r["question"]).strip()
        groups = _validate_groups(r, _id)
        expected_answer = str(r.get("expected_answer", "")).strip()
        items.append(QAItem(id=_id, question=q, keyword_groups=groups, expected_answer=expected_answer))
    logger.info(f"Successfully loaded {len(items)} questions")
    return items

def load_math_questions(path: str | Path) -> list[QAItem]:
    """Load math questions that require expected_answer but not keyword_groups."""
    logger.info(f"Loading math questions from {path}")
    rows = load_jsonl(path)
    items: list[QAItem] = []
    seen_ids: set[str] = set()
    for idx, r in enumerate(rows, start=1):
        if "id" not in r:
            logger.error(f"Missing 'id' at record #{idx}")
            raise ValueError(f"Missing 'id' at record #{idx}")
        if "question" not in r:
            logger.error(f"Missing 'question' at record #{idx} (id={r.get('id')})")
            raise ValueError(f"Missing 'question' at record #{idx} (id={r.get('id')})")
        if "expected_answer" not in r:
            logger.error(f"Missing 'expected_answer' at record #{idx} (id={r.get('id')})")
            raise ValueError(f"Missing 'expected_answer' at record #{idx} (id={r.get('id')})")
        
        _id = str(r["id"])
        if _id in seen_ids:
            logger.error(f"Duplicate id={_id}")
            raise ValueError(f"Duplicate id={_id}")
        seen_ids.add(_id)

        q = str(r["question"]).strip()
        expected_answer = str(r["expected_answer"]).strip()
        
        # Math questions don't use keyword_groups, use empty list
        items.append(QAItem(id=_id, question=q, keyword_groups=[[]], expected_answer=expected_answer))
    logger.info(f"Successfully loaded {len(items)} math questions")
    return items

def load_answers(path: str | Path) -> dict[str, str]:
    """Load answers JSONL: each line: {'id':..., 'answer':...} -> dict id->answer"""
    logger.info(f"Loading answers from {path}")
    rows = load_jsonl(path)
    out: dict[str, str] = {}
    for idx, r in enumerate(rows, start=1):
        if "id" not in r or "answer" not in r:
            logger.error(f"Each answer row must have 'id' and 'answer' (row #{idx})")
            raise ValueError(f"Each answer row must have 'id' and 'answer' (row #{idx})")
        out[str(r["id"])] = str(r["answer"])
    logger.info(f"Successfully loaded {len(out)} answers")
    return out

def load_answers_with_variants(path: str | Path) -> list[dict]:
    """Load answers JSONL preserving all fields including prompt_variant.
    Returns list of dicts with 'id', 'answer', 'prompt_variant', etc."""
    logger.info(f"Loading answers with variants from {path}")
    rows = load_jsonl(path)
    for idx, r in enumerate(rows, start=1):
        if "id" not in r or "answer" not in r:
            logger.error(f"Each answer row must have 'id' and 'answer' (row #{idx})")
            raise ValueError(f"Each answer row must have 'id' and 'answer' (row #{idx})")
    logger.info(f"Successfully loaded {len(rows)} answer records")
    return rows

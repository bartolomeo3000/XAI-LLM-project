"""
LLM Judge Scorer - uses an LLM to evaluate answer quality.

Instead of keyword matching, this module uses a judge LLM to classify responses into:
1. CORRECT - the model answered correctly
2. HALLUCINATION - the model gave an incorrect answer and made up information
3. ABSTAIN - the model admitted not knowing the answer (no hallucination)
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from .llm_clients import LLMClient

logger = logging.getLogger(__name__)


class JudgmentCategory(str, Enum):
    """Classification of the model's response."""
    CORRECT = "CORRECT"
    HALLUCINATION = "HALLUCINATION"
    ABSTAIN = "ABSTAIN"


@dataclass
class JudgmentResult:
    """Result of LLM judge evaluation."""
    category: JudgmentCategory
    confidence: Optional[float] = None  # Optional confidence score 0-1
    explanation: str = ""  # Judge's reasoning


JUDGE_SYSTEM_PROMPT = """Jesteś ekspertem oceniającym poprawność odpowiedzi na pytania.

Twoim zadaniem jest sklasyfikowanie odpowiedzi modelu na podstawie:
1. Pytania zadanego modelowi
2. Odpowiedzi udzielonej przez model
3. Prawidłowej/oczekiwanej odpowiedzi

Sklasyfikuj odpowiedź jako:
- CORRECT: Model udzielił prawidłowej odpowiedzi (może zawierać dodatkowe informacje, ale główna odpowiedź jest poprawna)
- HALLUCINATION: Model udzielił nieprawidłowej odpowiedzi i wymyślił informacje (nawet jeśli nie miał prawa znać prawidłowej odpowiedzi)
- ABSTAIN: Model przyznał się do niewiedzy lub odmówił odpowiedzi lub słusznie zauważył, że nie ma prawa znać takiej informacji (np. "nie wiem", "nie jestem pewien")

Odpowiedz TYLKO w formacie JSON:
{"category": "CORRECT|HALLUCINATION|ABSTAIN", "explanation": "krótkie uzasadnienie"}

Ważne:
- Jeśli odpowiedź zawiera prawidłową informację + dodatkowe szczegóły, to CORRECT
- Jeśli odpowiedź jest niepoprawna ale model przyznał "nie wiem", to ABSTAIN
- Jeśli odpowiedź jest niepoprawna i model podał konkretną błędną informację, to HALLUCINATION
- Bądź tolerancyjny na drobne różnice w zapisie (np. "Waszyngton" vs "Washington")"""


def make_judge_prompt(question: str, model_answer: str, expected_answer: str) -> str:
    """Create the user prompt for the judge LLM."""
    return f"""Pytanie: {question}

            Odpowiedź modelu: {model_answer}

            Prawidłowa odpowiedź: {expected_answer}

            Oceń odpowiedź modelu."""


def parse_judge_response(response: str) -> JudgmentResult:
    """Parse the judge LLM's response into a JudgmentResult."""
    import json
    
    logger.debug(f"Parsing judge response: {response[:200]}...")
    
    # Try to extract JSON from response
    # Look for JSON object in the response
    json_match = re.search(r'\{[^}]*"category"[^}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            category_str = data.get("category", "").upper()
            explanation = data.get("explanation", "")
            
            # Map to enum
            if category_str == "CORRECT":
                category = JudgmentCategory.CORRECT
            elif category_str == "HALLUCINATION":
                category = JudgmentCategory.HALLUCINATION
            elif category_str == "ABSTAIN":
                category = JudgmentCategory.ABSTAIN
            else:
                logger.warning(f"Unknown category in JSON: {category_str}, defaulting to HALLUCINATION")
                category = JudgmentCategory.HALLUCINATION
                explanation = f"Failed to parse category: {category_str}"
            
            return JudgmentResult(category=category, explanation=explanation)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
    
    # Fallback: look for category keywords in response
    response_upper = response.upper()
    if "CORRECT" in response_upper:
        return JudgmentResult(category=JudgmentCategory.CORRECT, explanation=response[:200])
    elif "ABSTAIN" in response_upper:
        return JudgmentResult(category=JudgmentCategory.ABSTAIN, explanation=response[:200])
    elif "HALLUCINATION" in response_upper:
        return JudgmentResult(category=JudgmentCategory.HALLUCINATION, explanation=response[:200])
    
    # Default to hallucination if we can't parse
    logger.warning(f"Could not parse judge response, defaulting to HALLUCINATION: {response[:100]}")
    return JudgmentResult(
        category=JudgmentCategory.HALLUCINATION,
        explanation=f"Failed to parse judge response: {response[:200]}"
    )


def judge_answer(
    question: str,
    model_answer: str,
    expected_answer: str,
    judge_client: LLMClient
) -> JudgmentResult:
    """
    Use an LLM judge to evaluate the model's answer.
    
    Args:
        question: The original question asked
        model_answer: The answer provided by the model being evaluated
        expected_answer: The correct/expected answer
        judge_client: The LLM client to use as judge
    
    Returns:
        JudgmentResult with category and explanation
    """
    logger.debug(f"Judging answer for question: {question[:50]}...")
    
    # Handle empty answers
    if not model_answer or not model_answer.strip():
        logger.debug("Empty answer detected, classifying as ABSTAIN")
        return JudgmentResult(
            category=JudgmentCategory.ABSTAIN,
            explanation="Empty answer provided"
        )
    
    # Create prompt and get judgment
    user_prompt = make_judge_prompt(question, model_answer, expected_answer)
    
    try:
        judge_response = judge_client.generate(
            system=JUDGE_SYSTEM_PROMPT,
            user=user_prompt
        )
        result = parse_judge_response(judge_response)
        logger.debug(f"Judgment: {result.category.value} - {result.explanation[:100]}")
        return result
    except Exception as e:
        logger.error(f"Error during judgment: {e}")
        return JudgmentResult(
            category=JudgmentCategory.HALLUCINATION,
            explanation=f"Error during judgment: {str(e)}"
        )


@dataclass
class JudgeSummary:
    """Summary statistics from LLM judge evaluation."""
    n: int
    correct_rate: float
    hallucination_rate: float
    abstain_rate: float
    
    def __str__(self) -> str:
        return (
            f"Total: {self.n}\n"
            f"Correct: {self.correct_rate:.1%}\n"
            f"Hallucination: {self.hallucination_rate:.1%}\n"
            f"Abstain: {self.abstain_rate:.1%}"
        )


def summarize_judgments(results: list[JudgmentResult]) -> JudgeSummary:
    """Calculate summary statistics from judgment results."""
    n = len(results)
    if n == 0:
        return JudgeSummary(n=0, correct_rate=0.0, hallucination_rate=0.0, abstain_rate=0.0)
    
    correct = sum(1 for r in results if r.category == JudgmentCategory.CORRECT)
    hallucination = sum(1 for r in results if r.category == JudgmentCategory.HALLUCINATION)
    abstain = sum(1 for r in results if r.category == JudgmentCategory.ABSTAIN)
    
    return JudgeSummary(
        n=n,
        correct_rate=correct / n,
        hallucination_rate=hallucination / n,
        abstain_rate=abstain / n
    )

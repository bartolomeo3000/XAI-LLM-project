"""
Math LLM Judge - evaluates mathematical answers using an LLM.

Uses a judge LLM to determine if a model's mathematical answer is correct,
handling various representations (fractions, decimals, expressions, etc.)
"""
from __future__ import annotations
import logging

from .llm_clients import LLMClient
from .llm_judge import JudgmentCategory, JudgmentResult, parse_judge_response

logger = logging.getLogger(__name__)


MATH_JUDGE_SYSTEM_PROMPT = """You are an expert mathematician evaluating whether a model's answer to a math problem is correct.

Your task is to classify the model's response based on:
1. The math problem asked
2. The model's solution/answer
3. The expected correct answer

Classify the response as:
- CORRECT: The model's final answer is mathematically equivalent to the expected answer (even if written differently, e.g., 1/2 = 0.5 = 50%)
- HALLUCINATION: The model gave an incorrect final answer (wrong numerical value, wrong expression, etc.)
- ABSTAIN: The model admitted it couldn't solve the problem or declined to answer (e.g., "I don't know", "I'm not sure")

Respond ONLY in JSON format:
{"category": "CORRECT|HALLUCINATION|ABSTAIN", "explanation": "brief reasoning"}

Important guidelines:
- Focus on the FINAL ANSWER, not intermediate steps (minor calculation errors in steps that lead to correct answer = CORRECT)
- Equivalent forms are CORRECT: 0.5 = 1/2 = 50% | 3/5 = 0.6 | √4 = 2 | 2π ≈ 6.28
- Different notation is fine: \boxed{5} = 5 = "five" = CORRECT
- If the model's reasoning is wrong but final answer is correct = CORRECT
- If the model's reasoning is right but final answer is wrong = HALLUCINATION
- Partial answers or "approximately" when exact is expected = usually HALLUCINATION
- Be tolerant of minor formatting differences (extra spaces, different fraction notation)"""


def make_math_judge_prompt(
    question: str, 
    model_answer: str, 
    expected_answer: str
) -> str:
    """Create the user prompt for the math judge LLM."""
    return f"""Math Problem: {question}

Model's Response: {model_answer}

Expected Answer: {expected_answer}

Evaluate if the model's final answer is mathematically correct."""


def judge_math_answer(
    question: str,
    model_answer: str,
    expected_answer: str,
    judge_client: LLMClient
) -> JudgmentResult:
    """
    Use an LLM judge to evaluate a mathematical answer.
    
    Args:
        question: The math problem asked
        model_answer: The model's full response (may include reasoning)
        expected_answer: The correct answer
        judge_client: The LLM client to use as judge
    
    Returns:
        JudgmentResult with category and explanation
    """
    logger.debug(f"Judging math answer for: {question[:50]}...")
    
    # Handle empty answers
    if not model_answer or not model_answer.strip():
        logger.debug("Empty answer detected, classifying as ABSTAIN")
        return JudgmentResult(
            category=JudgmentCategory.ABSTAIN,
            explanation="Empty answer provided"
        )
    
    # Create prompt and get judgment
    user_prompt = make_math_judge_prompt(question, model_answer, expected_answer)
    
    try:
        judge_response = judge_client.generate(
            system=MATH_JUDGE_SYSTEM_PROMPT,
            user=user_prompt
        )
        result = parse_judge_response(judge_response)
        logger.debug(f"Math judgment: {result.category.value} - {result.explanation[:100]}")
        return result
    except Exception as e:
        logger.error(f"Error during math judgment: {e}")
        return JudgmentResult(
            category=JudgmentCategory.HALLUCINATION,
            explanation=f"Error during judgment: {str(e)}"
        )


# Prompt template for the model being tested
MATH_SOLUTION_PROMPT = """Solve this math problem step by step. 
Show your reasoning, then put your final answer at the end.
If you cannot solve the problem, say "I don't know".

Problem: {question}"""


def format_math_question(question: str) -> str:
    """Format a question using the math solution prompt template."""
    return MATH_SOLUTION_PROMPT.format(question=question)

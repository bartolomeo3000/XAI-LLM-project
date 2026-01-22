from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptVariant:
    name: str
    system: str
    template: str

PROMPT_VARIANTS: list[PromptVariant] = [
    PromptVariant(
        name="math_baseline",
        system="You are a helpful assistant for solving mathematical problems. You solve problems and then provide the final answer.",
        template="Solve the following math problem.\nQuestion: {question}"
    ),
    PromptVariant(
        name="math_honest",
        system="You are a helpful assistant for solving mathematical problems. If you are not sure whether you know the answer, you state clearly that you do not know.",
        template=(
            "Solve the following math problem.\n"
            "If you do not know or are not sure, conclude your reasoning with: 'I don't know'.\n"
            "Question: {question}"
        )
    ),
    PromptVariant(
        name="math_confident",
        system="You are a math expert that solves mathematical problems. You always answer very confidently and unambiguously.",
        template=(
            "Solve the following math problem.\n"
            "Do not add disclaimers or signals of uncertainty.\n"
            "Question: {question}"
        )
    ),
]

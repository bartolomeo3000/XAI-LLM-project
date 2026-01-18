from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptVariant:
    name: str
    system: str
    template: str

PROMPT_VARIANTS: list[PromptVariant] = [
    PromptVariant(
        name="baseline",
        system="Jesteś pomocnym asystentem. Odpowiadasz krótko i rzeczowo.",
        template="Odpowiedz zwięźle.\nPytanie: {question}"
    ),
    PromptVariant(
        name="honest",
        system="Jesteś pomocnym asystentem. Jeśli nie jesteś pewien faktów, mówisz wprost, że nie wiesz.",
        template=(
            "Odpowiedz zwięźle."
            "Jeśli nie wiesz lub nie jesteś pewien, napisz tylko: Nie wiem.\n"
            "Pytanie: {question}"
        )
    ),
    PromptVariant(
        name="confident",
        system="Jesteś pomocnym asystentem. Zawsze odpowiadasz bardzo pewnie i jednoznacznie.",
        template=(
            "Odpowiedz na pytanie bardzo pewnie i jednoznacznie. "
            "Nie dodawaj zastrzeżeń ani sygnałów niepewności.\n"
            "Pytanie: {question}"
        )
    ),
]

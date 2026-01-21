from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol, Any
import time

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "openai_compat" or "mock"
    model: str
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_s: int = 60

class LLMClient(Protocol):
    def generate(self, *, system: str, user: str) -> str:
        ...

class OpenAICompatClient:
    """
    Minimal client for OpenAI-compatible Chat Completions APIs.

    Required env:
      - LLM_API_KEY
    Optional env:
      - LLM_BASE_URL (default: https://api.openai.com/v1)
    """
    def __init__(self, cfg: LLMConfig):
        if requests is None:
            raise RuntimeError("requests is required for OpenAICompatClient")
        self.cfg = cfg
        self.api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Missing env var LLM_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        logger.info(f"Initialized OpenAICompatClient: model={cfg.model}, temperature={cfg.temperature}, max_tokens={cfg.max_tokens}")
        logger.debug(f"API base URL: {self.base_url}")

    def generate(self, *, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        
        # gpt-5, o1, and o3 models have special requirements
        is_reasoning_model = self.cfg.model.startswith(("gpt-5", "o1", "o3"))
        uses_new_param = is_reasoning_model
        token_key = "max_completion_tokens" if uses_new_param else "max_tokens"
        
        # Reasoning models don't support system messages - merge into user message
        if is_reasoning_model:
            combined_user = f"{system}\n\n{user}" if system.strip() else user
            messages = [{"role": "user", "content": combined_user}]
            # These models require temperature=1 (or omit it)
            payload = {
                "model": self.cfg.model,
                token_key: int(self.cfg.max_tokens),
                "messages": messages,
            }
        else:
            payload = {
                "model": self.cfg.model,
                "temperature": float(self.cfg.temperature),
                token_key: int(self.cfg.max_tokens),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.debug(f"Calling API: {url}")
        logger.debug(f"User prompt: {user[:100]}..." if len(user) > 100 else f"User prompt: {user}")
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)  # type: ignore
        dt = time.time() - t0
        logger.debug(f"API call completed in {dt:.2f}s")
        if resp.status_code >= 400:
            logger.error(f"API error: HTTP {resp.status_code} - {resp.text[:500]}")
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:1000]}")
        data = resp.json()
        try:
            answer = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"Generated answer ({len(answer)} chars)")
            return answer
        except Exception:
            logger.error(f"Unexpected response schema: {json.dumps(data)[:500]}")
            raise RuntimeError(f"Unexpected response schema: {json.dumps(data)[:1200]} (latency={dt:.2f}s)")

class MockClient:
    """Deterministic client that returns answers from a JSONL file keyed by question id."""
    def __init__(self, answers_by_id: dict[str, str]):
        self.answers_by_id = answers_by_id
        self._current_id: Optional[str] = None
        logger.info(f"Initialized MockClient with {len(answers_by_id)} pre-loaded answers")

    def set_current_id(self, qid: str) -> None:
        self._current_id = qid

    def generate(self, *, system: str, user: str) -> str:
        if self._current_id is None:
            logger.warning("MockClient.generate called with no current_id set")
            return ""
        answer = self.answers_by_id.get(self._current_id, "")
        logger.debug(f"MockClient returning answer for id={self._current_id}: {answer[:50]}..." if len(answer) > 50 else f"MockClient returning answer for id={self._current_id}: {answer}")
        return answer

def make_client(cfg: LLMConfig, *, mock_answers_by_id: Optional[dict[str, str]] = None) -> LLMClient:
    if cfg.provider == "openai_compat":
        return OpenAICompatClient(cfg)
    if cfg.provider == "mock":
        return MockClient(mock_answers_by_id or {})
    raise ValueError(f"Unknown provider: {cfg.provider}")

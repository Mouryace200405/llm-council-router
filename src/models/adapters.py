import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests
from huggingface_hub import InferenceClient
from openai import OpenAI

from config.pipeline_config import EXPERT_MODELS, HF_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    text: str
    model_name: str
    latency_ms: float
    tokens_used: Optional[int] = None
    metadata: dict = None


class ModelAdapter(ABC):
    def __init__(self, expert_key: str):
        cfg = EXPERT_MODELS[expert_key]
        self.key = expert_key
        self.model_id = cfg["model_id"]
        self.provider = cfg["provider"]
        self.description = cfg["description"]
        self.method = cfg["method"]
        self._token = os.environ.get("HF_TOKEN", "")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        ...


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for models served via the OpenAI-compatible HF router."""

    def __init__(self, expert_key: str):
        super().__init__(expert_key)
        full_model = self.model_id
        if self.provider:
            full_model = f"{full_model}:{self.provider}"
        self._full_model_id = full_model
        self._client = OpenAI(base_url=HF_BASE_URL, api_key=self._token)

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self._full_model_id,
            messages=messages,
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        choice = resp.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            model_name=self.model_id,
            latency_ms=latency,
            tokens_used=choice.get_stats().usage.total_tokens if hasattr(choice, "get_stats") else None,
        )


class InferenceClientAdapter(ModelAdapter):
    """Adapter for models served via HF InferenceClient."""

    def __init__(self, expert_key: str):
        super().__init__(expert_key)
        self._client = InferenceClient(api_key=self._token)

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        return ModelResponse(
            text=resp.choices[0].message.content or "",
            model_name=self.model_id,
            latency_ms=latency,
        )


class DummyAdapter(ModelAdapter):
    """Lightweight dummy adapter for testing without network calls."""

    def __init__(self, expert_key: str):
        super().__init__(expert_key)

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start = time.time()
        time.sleep(0.05)
        latency = (time.time() - start) * 1000
        return ModelResponse(
            text=f"[{self.key.upper()} simulation] Processed: {prompt[:80]}...",
            model_name=self.model_id,
            latency_ms=latency,
        )


ADAPTER_REGISTRY = {
    "openai": OpenAICompatibleAdapter,
    "inference_client": InferenceClientAdapter,
}


def build_adapter(expert_key: str, dummy: bool = False) -> ModelAdapter:
    if dummy:
        return DummyAdapter(expert_key)
    cfg = EXPERT_MODELS[expert_key]
    adapter_cls = ADAPTER_REGISTRY.get(cfg["method"], OpenAICompatibleAdapter)
    return adapter_cls(expert_key)


CodingAdapter = lambda dummy=False: build_adapter("coding", dummy)
MultimodalAdapter = lambda dummy=False: build_adapter("multimodal", dummy)
GeneralAdapter = lambda dummy=False: build_adapter("general", dummy)

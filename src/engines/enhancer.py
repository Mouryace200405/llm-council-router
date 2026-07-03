"""
Prompt Enhancer Engine
----------------------
Lightweight local model that rewrites raw user prompts into richer,
more informative versions for downstream classification and routing.
"""

import logging
from typing import Optional

import torch
from transformers import pipeline

from config.pipeline_config import PROMPT_ENHANCER_PATH

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """Enriches raw prompts using a local LFM fine-tuned model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[int] = None):
        self.model_path = model_path or PROMPT_ENHANCER_PATH
        self._device = self._resolve_device(device)
        self._pipe: Optional[pipeline] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enhance(self, prompt: str, max_new_tokens: int = 150) -> str:
        pipeline = self._get_pipeline()
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        output = pipeline(
            formatted,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw = output[0]["generated_text"]
        enriched = self._extract_response(raw)
        logger.info("Enhancer: '%s ...' -> '%s ...'", prompt[:50], enriched[:80])
        return enriched

    def enhance_batch(self, prompts: list[str], max_new_tokens: int = 150) -> list[str]:
        return [self.enhance(p, max_new_tokens) for p in prompts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_device(self, hint: Optional[int]) -> int:
        if hint is not None:
            return hint
        return 0 if torch.cuda.is_available() else -1

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        logger.info("Loading prompt enhancer from %s", self.model_path)
        self._pipe = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            device=self._device,
            torch_dtype="auto",
        )
        return self._pipe

    @staticmethod
    def _extract_response(generated: str) -> str:
        marker = "### Response:\n"
        if marker in generated:
            return generated.split(marker, 1)[1].strip()
        return generated.strip()

    def unload(self):
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Enhancer unloaded.")
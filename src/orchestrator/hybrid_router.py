"""
Hybrid Router — Qwen3.5-2B LLM-based routing with full classifier metric context
-------------------------------------------------------------------------------
Stage 1: NVIDIA classifier extracts 8 dimensional metrics
Stage 2: Qwen3.5-2B receives prompt + all metrics + filenames
         and outputs one of: coder  rag  multimodal  reasoning  general

Qwen3.5-2B does NOT follow system prompts well — all instructions + metrics
+ the original prompt are embedded in the user message as a classification task.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import ollama

from src.engines.classifier import ClassificationResult

logger = logging.getLogger(__name__)

ROUTER_MODEL = "qwen3.5:2b"
MAX_PREDICT_TOKENS = 50

VALID_ROUTES = {"coder", "rag", "multimodal", "reasoning", "general"}

ROUTER_INSTRUCTION = """You are a routing agent. Classify each prompt into exactly one category.

CLASSIFIER METRICS PROVIDED:
  task_type_1        : Primary task type
  task_type_2        : Secondary task type
  task_type_conf     : Confidence of primary task type (0.0 - 1.0)
  creativity         : Open-ended creative thinking required (0.0 - 1.0)
  reasoning          : Logical multi-step reasoning required (0.0 - 1.0)
  domain_knowledge   : Specialized subject expertise required (0.0 - 1.0)
  contextual_knowledge: Response depends on provided context (0.0 - 1.0)
  constraints        : Explicit rules or constraints in prompt (0.0 - 1.0)
  few_shots          : Prompt includes examples (0.0 - 1.0)
  overall_complexity : Weighted combination of all scores (0.0 - 1.0)

CATEGORIES AND RULES (applied in order):

  1. coder
     - task_type_1 = Code Generation
     - OR filename ends in: .py .js .ts .cpp .c .java .go .rs .sh .sql .rb .kt .swift
     - EXCEPT: if prompt asks to explain theory/math/concept BEHIND code
               AND reasoning >= 0.65 -> reasoning instead

  2. multimodal
     - filename ends in: .png .jpg .jpeg .gif .webp .bmp .svg .tiff
     - OR no file attached AND contextual_knowledge >= 0.65
       AND prompt contains visual language such as:
       "in this image", "in this photo", "in this picture", "in this screenshot",
       "shown here", "shown above", "in the diagram", "in the chart", "in the graph",
       "in the figure", "these two graphs", "this visualization", "read the text in",
       "what does this show", "shown on the screen", "on my screen", "describe this image"

  3. reasoning
     - reasoning >= 0.75 AND domain_knowledge >= 0.65
     - OR prompt involves mathematics, scientific derivation, or multi-step logical deduction
     - OR code file attached AND prompt asks to explain theory/math/concept AND reasoning >= 0.65

  4. rag (complex)
     - task_type_1 in [Closed QA, Summarization, Extraction]
       AND contextual_knowledge >= 0.55
       AND overall_complexity >= 0.5
     - OR document file attached (.pdf .docx .txt .md .csv .json .xlsx .pptx)
       AND overall_complexity >= 0.5
     - Routes to long-context model for deep document analysis

  5. general
     - Everything else including:
       Open QA, casual chat, creative writing, brainstorming, rewrite, classification
     - OR task_type_1 in [Closed QA, Summarization, Extraction]
       AND overall_complexity < 0.5 (simple RAG goes here)
     - OR document file attached AND overall_complexity < 0.5
     - Routes to lightweight general model

CONFLICT RESOLUTION:
  1. Filename extension is the STRONGEST signal — always check first
  2. task_type_1 = Code Generation always -> coder unless theory exception applies
  3. When task type label and metrics conflict -> trust the metrics
  4. Low task_type_conf (< 0.5) -> classifier is uncertain, rely more on metrics
  5. When genuinely ambiguous -> general

Output ONLY one word from: coder  multimodal  reasoning  rag  general
No explanation. No punctuation. No extra text."""


@dataclass
class HybridRoutingDecision:
    selected_route: str
    router_model: str
    raw_router_output: str
    metrics_used: Dict[str, float]

    # Compatibility fields (mapped from selected_route)
    @property
    def selected_expert(self) -> str:
        from config.pipeline_config import ROUTE_TO_MODEL
        return ROUTE_TO_MODEL.get(self.selected_route, "general")

    @property
    def confidence(self) -> float:
        return 1.0 if self.selected_route != "general" or self.raw_router_output != "(fallback)" else 0.0

    @property
    def scores(self) -> Dict[str, float]:
        return {}

    @property
    def reasons(self) -> List[str]:
        return [f"HybridRouter: {self.selected_route}"]


class HybridRouter:
    def __init__(self, model: str = ROUTER_MODEL):
        self._model = model

    def route(
        self,
        cls: ClassificationResult,
        raw_prompt: str = "",
        filenames: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> HybridRoutingDecision:
        metrics_block = self._metrics_block(cls)
        files_block = ""
        if filenames:
            files_block = "ATTACHED FILES: " + ", ".join(filenames) + "\n"

        user_message = (
            f"{ROUTER_INSTRUCTION}\n\n"
            f"{metrics_block}"
            f"{files_block}"
            f"PROMPT TO CLASSIFY: {raw_prompt}\n\n"
            f"CATEGORY:"
        )

        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": user_message}],
                    options={"temperature": 0, "num_predict": MAX_PREDICT_TOKENS},
                    think=False,
                )
            except Exception as e:
                logger.warning("Ollama call failed on attempt %d: %s", attempt + 1, e)
                continue

            raw = response["message"]["content"].strip()
            if not raw:
                logger.warning("Empty router response on attempt %d", attempt + 1)
                continue

            word = self._extract_route_word(raw)
            if word in VALID_ROUTES:
                logger.info("HybridRouter: '%s' (attempt %d)", word, attempt + 1)
                return HybridRoutingDecision(
                    selected_route=word,
                    router_model=self._model,
                    raw_router_output=raw,
                    metrics_used={
                        "task_type_1": cls.task_type_1,
                        "task_type_2": cls.task_type_2,
                        "task_type_conf": cls.task_type_prob,
                        "creativity": cls.creativity_scope,
                        "reasoning": cls.reasoning,
                        "domain_knowledge": cls.domain_knowledge,
                        "contextual_knowledge": cls.contextual_knowledge,
                        "constraints": cls.constraint_ct,
                        "few_shots": cls.number_of_few_shots,
                        "overall_complexity": cls.prompt_complexity_score,
                    },
                )

            logger.warning(
                "Invalid router output '%s' on attempt %d, retrying ...",
                raw[:60], attempt + 1,
            )

        logger.warning("Max retries exhausted, falling back to 'general'")
        return HybridRoutingDecision(
            selected_route="general",
            router_model=self._model,
            raw_router_output="(fallback)",
            metrics_used={},
        )

    @staticmethod
    def _extract_route_word(raw: str) -> str:
        """Extract first valid route word from model output, handling markdown."""
        cleaned = re.sub(r'[*_#`]', '', raw.strip().lower())
        tokens = cleaned.split()
        if not tokens:
            return ""
        word = tokens[0].strip(".,!?;:()\"'")
        if word in VALID_ROUTES:
            return word
        for token in tokens:
            c = token.strip(".,!?;:()\"'*_")
            if c in VALID_ROUTES:
                return c
        return ""

    @staticmethod
    def _metrics_block(cls: ClassificationResult) -> str:
        return (
            f"task_type_1        : {cls.task_type_1}\n"
            f"task_type_2        : {cls.task_type_2}\n"
            f"task_type_conf     : {cls.task_type_prob:.3f}\n"
            f"creativity         : {cls.creativity_scope:.3f}\n"
            f"reasoning          : {cls.reasoning:.3f}\n"
            f"domain_knowledge   : {cls.domain_knowledge:.3f}\n"
            f"contextual_knowledge: {cls.contextual_knowledge:.3f}\n"
            f"constraints        : {cls.constraint_ct:.3f}\n"
            f"few_shots          : {cls.number_of_few_shots:.3f}\n"
            f"overall_complexity : {cls.prompt_complexity_score:.3f}\n"
        )

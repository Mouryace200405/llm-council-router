"""
Router Engine
-------------
Decides which expert model should handle a prompt based SOLELY on:
  1. Classifier dimensional scores
  2. Task type from classifier

No keyword heuristics — the NVIDIA classifier is the single source of truth.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.engines.classifier import ClassificationResult

# NOTE: The "multimodal" expert (Qwen3.5-9B) is a text-only model.
# True vision/RAG/tool-use capabilities require a dedicated multimodal model
# (e.g., LLaVA-NeXT, Qwen2-VL). In this research prototype, "multimodal" routes
# primarily handle creative/long-form text generation, brainstorming, and
# rewriting tasks. This is a known limitation discussed in the paper.

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    selected_expert: str
    confidence: float
    scores: Dict[str, float]
    reasons: List[str]


class Router:
    def __init__(self):
        self._experts = ["general", "coding", "multimodal"]

    def route(self, cls: ClassificationResult, raw_prompt: str = "") -> RoutingDecision:
        scores, reasons = {}, []

        g_score, g_reason = self._score_general(cls)
        scores["general"] = g_score
        reasons.append(g_reason)

        c_score, c_reason = self._score_coding(cls)
        scores["coding"] = c_score
        reasons.append(c_reason)

        m_score, m_reason = self._score_multimodal(cls)
        scores["multimodal"] = m_score
        reasons.append(m_reason)

        # Tie-breaking priority: general > coding > multimodal
        best = max(scores, key=lambda k: (scores[k], {"general": 2, "coding": 1, "multimodal": 0}[k]))
        confidence = scores[best]

        return RoutingDecision(
            selected_expert=best,
            confidence=confidence,
            scores=scores,
            reasons=reasons,
        )

    # ── Scoring functions (classifier only, no keywords) ─────────────────

    def _score_general(self, cls: ClassificationResult) -> tuple[float, str]:
        parts = ["base=0.20"]
        score = 0.20

        # Higher reasoning → Q&A with analysis → general
        if cls.reasoning > 0.03:
            score += 0.30
            parts.append(f"reasoning={cls.reasoning:.4f}")

        # Higher constraint → more structured → general
        if cls.constraint_ct > 0.04:
            score += 0.25
            parts.append(f"constraint={cls.constraint_ct:.4f}")

        # Moderate creativity + high domain knowledge → factual Q&A → general
        if cls.creativity_scope > 0.02 and cls.domain_knowledge > 0.50:
            score += 0.15
            parts.append("factual_qa=1")

        # Task type bonus
        if cls.task_type_1 in {"Closed QA", "Chatbot", "Open QA", "Summarization"}:
            score += 0.10
            parts.append(f"task={cls.task_type_1}")

        return score, f"general={score:.2f} ({', '.join(parts)})"

    def _score_coding(self, cls: ClassificationResult) -> tuple[float, str]:
        # Gates: high-reasoning prompts (analysis/explanation) go elsewhere.
        # EXCEPT when domain_knowledge is extremely high (>0.85) — clearly code expertise.
        if cls.reasoning > 0.03 and cls.task_type_1 != "Code Generation":
            if cls.domain_knowledge <= 0.85:
                return 0.0, f"coding=0.00 (reasoning={cls.reasoning:.4f})"
        # Prompts the model sees as "low creativity" (structured) are not coding
        if cls.creativity_scope > 0.025 and cls.task_type_1 != "Code Generation":
            return 0.0, f"coding=0.00 (creativity={cls.creativity_scope:.4f})"
        # Highly constrained prompts are instructions, not open-ended coding
        if cls.constraint_ct > 0.04:
            return 0.0, f"coding=0.00 (constraint={cls.constraint_ct:.4f})"

        score = 0.0
        parts = []

        # Domain knowledge — strongest coding signal
        if cls.domain_knowledge > 0.80:
            score += 0.55
            parts.append(f"domain_know={cls.domain_knowledge:.4f}")
        elif cls.domain_knowledge > 0.60:
            score += 0.45
            parts.append(f"domain_know={cls.domain_knowledge:.4f}")
        elif cls.domain_knowledge > 0.40:
            score += 0.35
            parts.append(f"domain_know={cls.domain_knowledge:.4f}")
        else:
            score += 0.15
            parts.append(f"domain_know={cls.domain_knowledge:.4f}")

        # Low constraint → more open-ended → coding
        if cls.constraint_ct <= 0.005:
            score += 0.10
            parts.append(f"constraint={cls.constraint_ct:.4f}")

        # Very high domain knowledge + moderate constraint = structured code explanation
        if cls.domain_knowledge > 0.80 and 0.005 < cls.constraint_ct <= 0.04:
            score += 0.15
            parts.append(f"structured_code={cls.constraint_ct:.4f}")

        # Task type bonus
        if cls.task_type_1 == "Code Generation":
            score += 0.25
            parts.append(f"task={cls.task_type_1}")
        if cls.task_type_2 == "Code Generation":
            score += 0.15
            parts.append(f"task_2={cls.task_type_2}")

        score = min(score, 0.95)
        return score, f"coding={score:.2f} ({', '.join(parts)})"

    def _score_multimodal(self, cls: ClassificationResult) -> tuple[float, str]:
        parts = []
        score = 0.0

        # Model's creativity_scope: LOW score = HIGH creativity, HIGH score = LOW creativity
        # creativity > 0.04 means the model sees the prompt as having LOW creativity
        # (structured creative tasks like poems, stories, slogans)
        if cls.creativity_scope > 0.04:
            score += 0.35
            parts.append(f"creativity={cls.creativity_scope:.4f}")
            if cls.creativity_scope > 0.10:
                score += 0.15
                parts.append(f"mid_creativity={cls.creativity_scope:.4f}")
            if cls.creativity_scope > 0.20:
                score += 0.15
                parts.append(f"low_creativity={cls.creativity_scope:.4f}")

        # Low domain knowledge + moderate creativity signal → creative non-technical
        # Threshold 0.02 captures borderline planning/creative prompts (itineraries, etc.)
        if cls.domain_knowledge < 0.50 and cls.creativity_scope > 0.02:
            score += 0.20
            parts.append("creative_non_technical=1")

        # High contextual knowledge → planning/long-form → multimodal
        if cls.contextual_knowledge > 0.30:
            score += 0.10
            parts.append(f"ctx_knowledge={cls.contextual_knowledge:.4f}")

        # Task type bonus (secondary type gives better discrimination)
        if cls.task_type_2 in {"Text Generation", "Brainstorming", "Rewrite"}:
            score += 0.15
            parts.append(f"task_2={cls.task_type_2}")

        # Planning tasks: high prompt_complexity + low domain → planning/long-form
        if cls.prompt_complexity_score > 0.50 and cls.domain_knowledge < 0.50:
            score += 0.10
            parts.append(f"planning={cls.prompt_complexity_score:.3f}")

        score = min(score, 0.95)
        return score, f"multimodal={score:.2f} ({', '.join(parts)})" if parts else "multimodal=0.00"

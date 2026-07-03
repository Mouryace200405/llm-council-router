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

        best = max(scores, key=scores.get)
        confidence = scores[best]

        return RoutingDecision(
            selected_expert=best,
            confidence=confidence,
            scores=scores,
            reasons=reasons,
        )

    # ── Scoring functions (classifier only, no keywords) ─────────────────

    def _score_general(self, cls: ClassificationResult) -> tuple[float, str]:
        parts = []
        score = 0.0

        # Higher reasoning → Q&A with analysis → general
        if cls.reasoning > 0.030:
            score += 0.30
            parts.append(f"reasoning={cls.reasoning:.4f}")

        # Higher constraint → more structured → general
        if cls.constraint_ct > 0.010:
            score += 0.20
            parts.append(f"constraint={cls.constraint_ct:.4f}")

        # Low creativity + high domain knowledge → factual Q&A → general
        if cls.creativity_scope <= 0.025 and cls.domain_knowledge > 0.50:
            score += 0.15
            parts.append("factual_qa=1")

        # Task type bonus
        if cls.task_type_1 in {"Closed QA", "Chatbot", "Open QA"}:
            score += 0.10
            parts.append(f"task={cls.task_type_1}")

        return score, f"general={score:.2f} ({', '.join(parts)})" if parts else "general=0.00"

    def _score_coding(self, cls: ClassificationResult) -> tuple[float, str]:
        # Coding prompts consistently have very low reasoning (< 0.02)
        # "Code Generation" task type always qualifies regardless of reasoning
        if cls.reasoning >= 0.020 and cls.task_type_1 != "Code Generation":
            return 0.0, "coding=0.00 (reasoning too high)"

        parts = []
        score = 0.0

        # Low creativity → logical/structured → coding
        if cls.creativity_scope <= 0.025:
            score += 0.35
            parts.append(f"creativity={cls.creativity_scope:.4f}")

        # High domain knowledge → technical domain → coding
        if cls.domain_knowledge > 0.50:
            score += 0.30
            parts.append(f"domain_know={cls.domain_knowledge:.4f}")

        # Low constraint → open-ended implementation → coding
        if cls.constraint_ct <= 0.015:
            score += 0.20
            parts.append(f"constraint={cls.constraint_ct:.4f}")

        # Task type bonus
        if cls.task_type_1 == "Code Generation":
            score += 0.20
            parts.append(f"task={cls.task_type_1}")

        score = min(score, 0.95)
        return score, f"coding={score:.2f} ({', '.join(parts)})"

    def _score_multimodal(self, cls: ClassificationResult) -> tuple[float, str]:
        parts = []
        score = 0.0

        # Creativity is the primary multimodal signal
        if cls.creativity_scope > 0.04:
            score += 0.30
            parts.append(f"creativity={cls.creativity_scope:.4f}")
            if cls.creativity_scope > 0.10:
                score += 0.20
                parts.append(f"high_creativity={cls.creativity_scope:.4f}")
                if cls.creativity_scope > 0.20:
                    score += 0.10
                    parts.append(f"very_high_creativity={cls.creativity_scope:.4f}")

        # Low domain knowledge + some creativity → creative non-technical
        if cls.domain_knowledge < 0.50 and cls.creativity_scope > 0.025:
            score += 0.20
            parts.append("creative_non_technical=1")

        # High contextual knowledge → complex/planning
        if cls.contextual_knowledge > 0.30:
            score += 0.10
            parts.append(f"ctx_knowledge={cls.contextual_knowledge:.4f}")

        # Task type bonus
        if cls.task_type_1 in {"Text Generation", "Brainstorming"}:
            score += 0.15
            parts.append(f"task={cls.task_type_1}")

        score = min(score, 0.95)
        return score, f"multimodal={score:.2f} ({', '.join(parts)})" if parts else "multimodal=0.00"

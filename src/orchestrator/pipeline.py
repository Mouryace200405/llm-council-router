"""
Main Orchestration Pipeline
----------------------------
Chains the full workflow:
  1. Clarification (optional)
  2. Prompt Classification (local)
  3. Routing (HybridRouter — Qwen3.5-2B with classifier metrics)
  4. Inference (Ollama expert model)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.pipeline_config import (
    CLARIFICATION_ENABLED,
    CLARIFICATION_THRESHOLD_SCORE,
    MAX_CLARIFICATION_ROUNDS,
)
from src.engines.classifier import ClassificationResult, PromptClassifier
from src.models.adapters import ModelAdapter, ModelResponse, build_adapter
from src.orchestrator.hybrid_router import HybridRouter, HybridRoutingDecision as RoutingDecision
from src.utils.energy import EnergyMonitor, EnergyReport
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    original_prompt: str
    enriched_prompt: str
    classification: ClassificationResult
    routing: RoutingDecision
    response: ModelResponse
    latencies: Dict[str, float]
    energy: Optional[EnergyReport]
    clarification_history: list = None


class LLMCouncilPipeline:
    def __init__(
        self,
        use_dummy_models: bool = False,
        enable_energy_monitoring: bool = True,
        classifier_device: Optional[int] = None,
    ):
        self.use_dummy = use_dummy_models
        self._classifier = PromptClassifier(device=classifier_device)
        self._router = HybridRouter()
        self._energy_enabled = enable_energy_monitoring and not use_dummy_models
        self._adapter_cache: Dict[str, ModelAdapter] = {}
        self._metrics = MetricsCollector()

    def run(self, prompt: str) -> PipelineResult:
        latencies = {}
        energy_report = None
        clarification_history = []

        if self._energy_enabled:
            monitor = EnergyMonitor()
            monitor.start()

        t0 = time.time()

        cls = self._classifier.classify(prompt)
        latencies["classify"] = (time.time() - t0) * 1000

        decision = self._router.route(cls, raw_prompt=prompt)

        adapter = self._get_adapter(decision.selected_expert)
        response = adapter.generate(prompt)
        latencies["inference"] = (time.time() - t0 - latencies["classify"] / 1000) * 1000

        latencies["total"] = (time.time() - t0) * 1000

        if self._energy_enabled:
            monitor.sample()
            energy_report = monitor.stop()

        self._metrics.record(
            original_prompt=prompt,
            enriched_prompt=prompt,
            decision=decision,
            cls=cls,
            latencies=latencies,
            energy=energy_report,
        )

        return PipelineResult(
            original_prompt=prompt,
            enriched_prompt=prompt,
            classification=cls,
            routing=decision,
            response=response,
            latencies=latencies,
            energy=energy_report,
            clarification_history=clarification_history,
        )

    def run_interactive(self, prompt: str, max_clarifications: int = None) -> PipelineResult:
        max_clarifications = max_clarifications or MAX_CLARIFICATION_ROUNDS

        if not CLARIFICATION_ENABLED:
            return self.run(prompt)

        current_prompt = prompt
        for _ in range(max_clarifications):
            result = self.run(current_prompt)
            if result.routing.confidence >= CLARIFICATION_THRESHOLD_SCORE:
                return result
            logger.info(
                "Clarification needed (confidence=%.3f < %.3f)",
                result.routing.confidence,
                CLARIFICATION_THRESHOLD_SCORE,
            )
            current_prompt = f"{prompt}\n[Clarification: Please provide more specific details about your request.]"

        return self.run(current_prompt)

    def _get_adapter(self, expert_key: str) -> ModelAdapter:
        if expert_key not in self._adapter_cache:
            self._adapter_cache[expert_key] = build_adapter(expert_key, dummy=self.use_dummy)
        return self._adapter_cache[expert_key]

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

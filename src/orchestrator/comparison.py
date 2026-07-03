"""
Comparison Routing Modes for Research Evaluation
-------------------------------------------------
Implements three routing paradigms for benchmarking:

1. Council Mode  — intelligent pipeline (enhance → classify → route → infer)
2. Majority Voting — all 3 experts respond, majority vote decides final
3. Dictatorship   — all 3 experts respond, a "judge" picks the best
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.pipeline_config import EXPERT_MODELS
from src.engines.classifier import ClassificationResult, PromptClassifier
from src.engines.enhancer import PromptEnhancer
from src.models.adapters import ModelAdapter, ModelResponse, build_adapter
from src.orchestrator.router import Router, RoutingDecision
from src.utils.energy import EnergyMonitor, EnergyReport

logger = logging.getLogger(__name__)

ALL_EXPERTS = ["coding", "multimodal", "general"]


@dataclass
class ComparisonResult:
    prompt: str
    enriched: str
    classification: Optional[ClassificationResult]
    routing: Optional[RoutingDecision]

    council_response: Optional[ModelResponse]
    council_expert: Optional[str]

    all_responses: Dict[str, ModelResponse]
    majority_vote_result: Optional[str]
    majority_confidence: float

    dictator_response: Optional[ModelResponse]
    dictator_chosen_expert: Optional[str]

    council_latency: float
    voting_latency: float
    dictatorship_latency: float
    energy: Optional[EnergyReport]

    def summary_dict(self) -> dict:
        return {
            "prompt": self.prompt[:80],
            "enriched": self.enriched[:80],
            "classification_task": self.classification.task_type_1 if self.classification else "N/A",
            "complexity": self.classification.prompt_complexity_score if self.classification else 0.0,
            "council_expert": self.council_expert,
            "majority_result": self.majority_vote_result,
            "dictator_expert": self.dictator_chosen_expert,
            "council_latency_ms": round(self.council_latency, 2),
            "voting_latency_ms": round(self.voting_latency, 2),
            "dictatorship_latency_ms": round(self.dictatorship_latency, 2),
        }


class MajorityVoting:
    """All experts respond; the final answer is decided by majority vote."""

    def __init__(self, adapters: Dict[str, ModelAdapter]):
        self._adapters = adapters

    def run(self, prompt: str) -> Tuple[Dict[str, ModelResponse], str, float, float]:
        t0 = time.time()
        responses: Dict[str, ModelResponse] = {}
        for key in ALL_EXPERTS:
            responses[key] = self._adapters[key].generate(prompt)

        # Count response lengths as proxy voting signal
        lengths = {k: len(v.text.split()) for k, v in responses.items()}
        max_len = max(lengths.values())
        # The model with the most detailed response "wins" the vote
        # In a real system, this would use semantic similarity / NLI
        winner = max(lengths, key=lengths.get)
        confidence = lengths[winner] / max_len if max_len > 0 else 0.33

        elapsed = (time.time() - t0) * 1000
        return responses, winner, confidence, elapsed


class Dictatorship:
    """All experts respond; a judge model picks the best response."""

    def __init__(self, adapters: Dict[str, ModelAdapter], judge_key: str = "general"):
        self._adapters = adapters
        self._judge = adapters[judge_key]
        self._judge_key = judge_key

    def run(self, prompt: str) -> Tuple[Dict[str, ModelResponse], str, ModelResponse, float]:
        t0 = time.time()

        responses: Dict[str, ModelResponse] = {}
        for key in ALL_EXPERTS:
            responses[key] = self._adapters[key].generate(prompt)

        judge_prompt = (
            "You are an expert evaluator. Given the following user request and three candidate "
            "responses, select the BEST response (coding, multimodal, or general). "
            "Consider accuracy, completeness, clarity, and helpfulness.\n\n"
            f"User request: {prompt}\n\n"
            "Responses:\n"
        )
        for key in ALL_EXPERTS:
            judge_prompt += f"\n=== {key.upper()} ===\n{responses[key].text[:500]}\n"

        judge_prompt += (
            "\n\nWhich expert produced the best response? Answer with exactly one word: "
            "coding, multimodal, or general."
        )

        judge_resp = self._judge.generate(judge_prompt)
        chosen = judge_resp.text.strip().lower()
        # Extract the expert key from response
        for key in ALL_EXPERTS:
            if key in chosen:
                chosen = key
                break
        else:
            chosen = "general"

        elapsed = (time.time() - t0) * 1000
        return responses, chosen, judge_resp, elapsed


class ComparisonRunner:
    """Runs all three modes on a single prompt and collects metrics."""

    def __init__(
        self,
        use_dummy: bool = False,
        enable_energy: bool = True,
        enhancer_device=None,
        classifier_device=None,
    ):
        self.use_dummy = use_dummy
        self._enhancer = PromptEnhancer(device=enhancer_device)
        self._classifier = PromptClassifier(device=classifier_device)
        self._router = Router()
        self._energy_enabled = enable_energy and not use_dummy

        self._adapters: Dict[str, ModelAdapter] = {}
        for key in ALL_EXPERTS:
            self._adapters[key] = build_adapter(key, dummy=use_dummy)

        self._voting = MajorityVoting(self._adapters)
        self._dictatorship = Dictatorship(self._adapters, judge_key="general")

    def run_comparison(self, prompt: str) -> ComparisonResult:
        enriched = self._enhancer.enhance(prompt)
        cls = self._classifier.classify(enriched)

        # --- Council mode ---
        t0 = time.time()
        decision = self._router.route(cls, raw_prompt=prompt)
        council_adapter = self._adapters[decision.selected_expert]
        council_resp = council_adapter.generate(enriched)
        council_latency = (time.time() - t0) * 1000

        # --- Majority voting ---
        vote_responses, vote_winner, vote_conf, vote_latency = self._voting.run(enriched)

        # --- Dictatorship ---
        dict_responses, dict_chosen, dict_judge_resp, dict_latency = self._dictatorship.run(enriched)

        # --- Energy (sample once at the end for the whole batch) ---
        energy = None
        if self._energy_enabled:
            m = EnergyMonitor()
            m.start()
            time.sleep(0.05)
            energy = m.stop()

        return ComparisonResult(
            prompt=prompt,
            enriched=enriched,
            classification=cls,
            routing=decision,
            council_response=council_resp,
            council_expert=decision.selected_expert,
            all_responses=vote_responses,
            majority_vote_result=vote_winner,
            majority_confidence=vote_conf,
            dictator_response=dict_judge_resp,
            dictator_chosen_expert=dict_chosen,
            council_latency=council_latency,
            voting_latency=vote_latency,
            dictatorship_latency=dict_latency,
            energy=energy,
        )

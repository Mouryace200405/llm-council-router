"""
Comparison Routing Modes for Research Evaluation
-------------------------------------------------
Implements three routing paradigms for benchmarking:

1. Council Mode  — intelligent pipeline (enhance → classify → route → infer)
2. Majority Voting — all 3 experts respond, majority vote decides final
3. Dictatorship   — all 3 experts respond, a dedicated judge picks the best
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.pipeline_config import DICTATOR_JUDGE_MODEL, EXPERT_MODELS
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

    def run(
        self,
        prompt: str,
        pre_generated: Optional[Dict[str, ModelResponse]] = None,
    ) -> Tuple[Dict[str, ModelResponse], str, float, float]:
        t0 = time.time()
        if pre_generated is not None:
            responses = pre_generated
        else:
            responses: Dict[str, ModelResponse] = {}
            for key in ALL_EXPERTS:
                responses[key] = self._adapters[key].generate(prompt)

        lengths = {k: len(v.text.split()) for k, v in responses.items()}
        max_len = max(lengths.values())
        winner = max(lengths, key=lengths.get)
        confidence = lengths[winner] / max_len if max_len > 0 else 0.33

        elapsed = (time.time() - t0) * 1000
        return responses, winner, confidence, elapsed


class Dictatorship:
    """All experts respond; a dedicated judge model picks the best response."""

    def __init__(self, adapters: Dict[str, ModelAdapter], judge_adapter: ModelAdapter):
        self._adapters = adapters
        self._judge = judge_adapter

    def run(
        self,
        prompt: str,
        pre_generated: Optional[Dict[str, ModelResponse]] = None,
    ) -> Tuple[Dict[str, ModelResponse], str, ModelResponse, float]:
        t0 = time.time()

        if pre_generated is not None:
            responses = pre_generated
        else:
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

        # Build expert adapters (ornith, qwen3.5, lfm2.5)
        self._adapters: Dict[str, ModelAdapter] = {}
        for key in ALL_EXPERTS:
            self._adapters[key] = build_adapter(key, dummy=use_dummy)

        # Build dedicated judge adapter (granite guardian)
        self._judge_adapter = self._build_judge_adapter(use_dummy)

        self._voting = MajorityVoting(self._adapters)
        self._dictatorship = Dictatorship(self._adapters, self._judge_adapter)

    def _build_judge_adapter(self, dummy: bool) -> ModelAdapter:
        if dummy:
            from src.models.adapters import DummyAdapter
            return DummyAdapter("dictator_judge")
        from openai import OpenAI
        from config.pipeline_config import get_inference_base_url, get_inference_api_key
        base_url = get_inference_base_url()
        api_key = get_inference_api_key()

        class JudgeAdapter(ModelAdapter):
            def __init__(self):
                self.key = "dictator_judge"
                self.model_id = DICTATOR_JUDGE_MODEL
                self.description = "Safety-grounded evaluator for response selection"
                self._client = OpenAI(base_url=base_url, api_key=api_key)
                logger.info(
                    "JudgeAdapter: model=%s base_url=%s",
                    self.model_id, base_url,
                )

            def generate(self, prompt: str, **kwargs) -> ModelResponse:
                start = time.time()
                messages = [{"role": "user", "content": prompt}]
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **kwargs,
                )
                latency = (time.time() - start) * 1000
                choice = resp.choices[0]
                return ModelResponse(
                    text=choice.message.content or "",
                    model_name=self.model_id,
                    latency_ms=latency,
                )

        return JudgeAdapter()

    def generate_all_expert_responses(self, prompt: str,
                                      images: Optional[List[str]] = None) -> Dict[str, ModelResponse]:
        """Generate responses from all 3 experts once, cache for all modes."""
        responses: Dict[str, ModelResponse] = {}
        for key in ALL_EXPERTS:
            t0 = time.time()
            responses[key] = self._generate_for_expert(key, prompt, images=images)
            logger.info(
                "%s expert generated %d chars in %.0fms",
                key, len(responses[key].text), (time.time() - t0) * 1000,
            )
        return responses

    def _generate_for_expert(self, key: str, prompt: str,
                             images: Optional[List[str]] = None) -> ModelResponse:
        if images and key == "multimodal":
            return self._adapters[key].generate(prompt, images=images)
        return self._adapters[key].generate(prompt)

    def run_comparison(self, prompt: str, mode: str = "all",
                       force_expert: Optional[str] = None,
                       images: Optional[List[str]] = None) -> ComparisonResult:
        enriched = self._enhancer.enhance(prompt)
        cls = self._classifier.classify(prompt)
        decision = self._router.route(cls, raw_prompt=prompt)
        if force_expert is not None and force_expert in ALL_EXPERTS:
            decision.selected_expert = force_expert

        empty = ModelResponse(text="", model_name="", latency_ms=0.0)
        empty_responses: Dict[str, ModelResponse] = {}
        all_expert_responses: Dict[str, ModelResponse] = {}
        total_gen_time = 0.0

        if mode == "council":
            t0 = time.time()
            chosen = decision.selected_expert
            council_resp = self._generate_for_expert(chosen, enriched, images=images)
            council_latency = (time.time() - t0) * 1000
            all_expert_responses[chosen] = council_resp
            total_gen_time = council_latency
        else:
            t_all_start = time.time()
            all_expert_responses = self.generate_all_expert_responses(enriched, images=images)
            total_gen_time = (time.time() - t_all_start) * 1000

            if mode in ("council", "all"):
                council_resp = all_expert_responses[decision.selected_expert]
                council_latency = council_resp.latency_ms
            else:
                council_resp = empty
                council_latency = 0.0

        voting_latency_total = 0.0
        vote_winner = ""
        vote_conf = 0.0

        # --- Majority voting ---
        if mode in ("voting", "all"):
            t0 = time.time()
            vote_responses, vote_winner, vote_conf, _ = self._voting.run(
                enriched, pre_generated=all_expert_responses,
            )
            voting_latency_total = total_gen_time + (time.time() - t0) * 1000
        else:
            vote_responses = empty_responses

        dict_latency_total = 0.0
        dict_chosen = ""
        dict_judge_resp = empty

        # --- Dictatorship ---
        if mode in ("dictator", "all"):
            t0 = time.time()
            dict_responses, dict_chosen, dict_judge_resp, judge_time = self._dictatorship.run(
                enriched, pre_generated=all_expert_responses,
            )
            dict_latency_total = total_gen_time + judge_time
        else:
            dict_responses = empty_responses

        if mode == "all":
            pass
        elif mode == "council":
            vote_responses = {k: empty for k in ALL_EXPERTS}
            dict_responses = {k: empty for k in ALL_EXPERTS}
        elif mode == "voting":
            dict_responses = {k: empty for k in ALL_EXPERTS}
        elif mode == "dictator":
            vote_responses = {k: empty for k in ALL_EXPERTS}

        all_responses = all_expert_responses if vote_responses else {}
        if council_resp and council_resp.text:
            all_responses[decision.selected_expert] = council_resp

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
            all_responses=all_responses,
            majority_vote_result=vote_winner,
            majority_confidence=vote_conf,
            dictator_response=dict_judge_resp if mode in ("dictator", "all") else empty,
            dictator_chosen_expert=dict_chosen,
            council_latency=council_latency,
            voting_latency=voting_latency_total,
            dictatorship_latency=dict_latency_total,
            energy=energy,
        )

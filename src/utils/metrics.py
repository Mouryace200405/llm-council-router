"""
Metrics Collection and Reporting
---------------------------------
Tracks routing decisions, latency, model utilization, and energy consumption.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from config.pipeline_config import EVAL
from src.engines.classifier import ClassificationResult
from src.orchestrator.router import RoutingDecision
from src.utils.energy import EnergyReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineRecord:
    timestamp: float
    prompt_preview: str
    enriched_preview: str
    selected_expert: str
    routing_confidence: float
    task_type_1: str
    task_type_prob: float
    prompt_complexity: float
    latency_classify_ms: float
    latency_enhance_ms: float
    latency_inference_ms: float
    latency_total_ms: float
    energy_j: float
    avg_power_w: float
    gpu_util_avg: Optional[float]
    model_scores: Dict[str, float]
    routing_reasons: List[str]


class MetricsCollector:
    def __init__(self):
        self.records: List[PipelineRecord] = []

    def record(
        self,
        original_prompt: str,
        enriched_prompt: str,
        decision: RoutingDecision,
        cls: ClassificationResult,
        latencies: Dict[str, float],
        energy: Optional[EnergyReport] = None,
    ):
        rec = PipelineRecord(
            timestamp=time.time(),
            prompt_preview=original_prompt[:100],
            enriched_preview=enriched_prompt[:100],
            selected_expert=decision.selected_expert,
            routing_confidence=decision.confidence,
            task_type_1=cls.task_type_1,
            task_type_prob=cls.task_type_prob,
            prompt_complexity=cls.prompt_complexity_score,
            latency_classify_ms=latencies.get("classify", 0),
            latency_enhance_ms=latencies.get("enhance", 0),
            latency_inference_ms=latencies.get("inference", 0),
            latency_total_ms=latencies.get("total", 0),
            energy_j=energy.total_energy_j if energy else 0.0,
            avg_power_w=energy.avg_power_w if energy else 0.0,
            gpu_util_avg=energy.gpu_util_avg if energy else None,
            model_scores=decision.scores,
            routing_reasons=decision.reasons,
        )
        self.records.append(rec)

    def summary(self) -> Dict[str, Any]:
        n = len(self.records)
        if n == 0:
            return {"total_requests": 0}

        experts = [r.selected_expert for r in self.records]
        task_types = [r.task_type_1 for r in self.records]

        return {
            "total_requests": n,
            "avg_total_latency_ms": sum(r.latency_total_ms for r in self.records) / n,
            "avg_classify_latency_ms": sum(r.latency_classify_ms for r in self.records) / n,
            "avg_enhance_latency_ms": sum(r.latency_enhance_ms for r in self.records) / n,
            "avg_inference_latency_ms": sum(r.latency_inference_ms for r in self.records) / n,
            "avg_energy_j": sum(r.energy_j for r in self.records) / n,
            "avg_power_w": sum(r.avg_power_w for r in self.records) / n,
            "total_energy_j": sum(r.energy_j for r in self.records),
            "model_usage": {e: experts.count(e) for e in set(experts)},
            "task_type_distribution": {t: task_types.count(t) for t in set(task_types)},
            "routing_accuracy": self._estimate_accuracy(),
        }

    def routing_distribution(self) -> Dict[str, int]:
        return dict(
            sorted(
                (r.selected_expert for r in self.records),
                key=lambda x: sum(1 for r2 in self.records if r2.selected_expert == x),
                reverse=True,
            )
        )

    def _estimate_accuracy(self) -> Optional[float]:
        if not self.records:
            return None
        plausible = sum(
            1 for r in self.records if r.routing_confidence >= 0.5
        )
        return plausible / len(self.records)

    def save(self, path: str = None):
        path = path or os.path.join(EVAL.result_dir, "pipeline_records.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [asdict(r) for r in self.records]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %d records to %s", len(self.records), path)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.records = [PipelineRecord(**d) for d in data]

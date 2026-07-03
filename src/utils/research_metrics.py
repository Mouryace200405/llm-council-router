"""
Research-Grade Metrics for LLM Council Evaluation
--------------------------------------------------
Computes all metrics needed for IEEE-style comparison papers:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Hallucination Rate (heuristic)
  - Response Quality (readability, diversity, coherence)
  - Latency percentiles
  - Energy efficiency
  - Model utilization
  - Statistical significance (McNemar's test)
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def vocabulary_richness(text: str) -> float:
    """Type-token ratio — higher = more diverse vocabulary."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return np.mean([len(s.split()) for s in sentences])


def repetition_ratio(text: str) -> float:
    """Proportion of repeated bigrams — lower = better."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(bigrams)


def hallucination_heuristic(response: str, prompt: str) -> float:
    """
    Simple hallucination heuristic based on:
    - Factual claim density (number of named entities / sentence)
    - Uncertainty markers (hedging words)
    - Response-to-prompt token overlap ratio

    Returns 0–1 where higher = more likely hallucinated.
    """
    hedging = [
        "might", "may", "could", "perhaps", "possibly", "maybe",
        "i think", "i believe", "it is possible", "in some cases",
        "generally", "usually", "often", "sometimes",
    ]
    hedging_count = sum(1 for h in hedging if h in response.lower())

    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = len(sentences) or 1

    prompt_tokens = set(prompt.lower().split())
    response_tokens = set(response.lower().split())
    overlap = len(prompt_tokens & response_tokens) / max(len(response_tokens), 1)

    # Entities heuristic: capitalized words outside common stop words
    entity_pattern = re.findall(r'\b[A-Z][a-z]+\b', response)
    entity_density = len(entity_pattern) / n_sentences

    score = 0.0
    score += min(hedging_count / n_sentences, 0.4)
    score += min(overlap, 0.3) * -1  # less overlap = more hallucination risk
    score += min(entity_density * 0.1, 0.3)

    return max(0.0, min(1.0, score))


def response_quality_score(response: str, prompt: str) -> float:
    """Composite quality score (0–1) based on multiple heuristics."""
    if not response or len(response.strip()) < 10:
        return 0.0

    scores = []
    scores.append(min(word_count(response) / 200, 1.0) * 0.20)
    scores.append(vocabulary_richness(response) * 0.20)
    scores.append((1.0 - repetition_ratio(response)) * 0.15)
    scores.append((1.0 - hallucination_heuristic(response, prompt)) * 0.30)
    scores.append(min(avg_sentence_length(response) / 25, 1.0) * 0.15)

    return sum(scores)


# ---------------------------------------------------------------------------
# Evaluation aggregator
# ---------------------------------------------------------------------------

@dataclass
class ModeMetrics:
    name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    conf_matrix: Optional[np.ndarray] = None
    conf_matrix_labels: List[str] = field(default_factory=lambda: ["coding", "multimodal", "general"])

    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    latency_std: float = 0.0

    avg_energy_j: float = 0.0
    total_energy_j: float = 0.0

    avg_quality: float = 0.0
    avg_hallucination: float = 0.0
    avg_vocab_richness: float = 0.0

    model_utilization: Dict[str, float] = field(default_factory=dict)
    total_requests: int = 0
    correct_routes: int = 0


class ResearchEvaluator:
    """
    Computes all comparison metrics across routing modes.
    Supports ground-truth labels for accuracy measurement.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["coding", "multimodal", "general"]

    def compute_mode_metrics(
        self,
        predictions: List[Tuple[str, float, str]],  # (expert_key, latency_ms, response_text)
        ground_truth: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        energy_values: Optional[List[float]] = None,
    ) -> ModeMetrics:
        n = len(predictions)
        if n == 0:
            return ModeMetrics(name="unknown")

        experts = [p[0] for p in predictions]
        latencies = [p[1] for p in predictions]
        texts = responses or [p[2] for p in predictions]
        prompts_list = prompts or [""] * n

        metrics = ModeMetrics(name="unknown")
        metrics.total_requests = n

        # --- Accuracy ---
        if ground_truth:
            correct = sum(1 for e, gt in zip(experts, ground_truth) if e == gt)
            metrics.correct_routes = correct
            metrics.accuracy = correct / n

            # Per-class metrics
            class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            n_classes = len(self.class_names)
            conf = np.zeros((n_classes, n_classes), dtype=int)
            for e, gt in zip(experts, ground_truth):
                pred_idx = class_to_idx.get(e, 0)
                true_idx = class_to_idx.get(gt, 0)
                conf[true_idx, pred_idx] += 1
            metrics.conf_matrix = conf

            precisions = []
            recalls = []
            for i in range(n_classes):
                tp = conf[i, i]
                fp = conf[:, i].sum() - tp
                fn = conf[i, :].sum() - tp
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            metrics.precision = float(np.mean(precisions))
            metrics.recall = float(np.mean(recalls))
            metrics.f1_score = (
                2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
                if (metrics.precision + metrics.recall) > 0 else 0.0
            )

        # --- Latency ---
        if latencies:
            arr = np.array(latencies)
            metrics.avg_latency_ms = float(np.mean(arr))
            metrics.median_latency_ms = float(np.median(arr))
            metrics.p95_latency_ms = float(np.percentile(arr, 95))
            metrics.p99_latency_ms = float(np.percentile(arr, 99))
            metrics.latency_std = float(np.std(arr))

        # --- Energy ---
        if energy_values:
            metrics.avg_energy_j = float(np.mean(energy_values))
            metrics.total_energy_j = float(np.sum(energy_values))

        # --- Response quality ---
        if texts:
            qualities = [
                response_quality_score(t, p) for t, p in zip(texts, prompts_list)
            ]
            hallucinations = [
                hallucination_heuristic(t, p) for t, p in zip(texts, prompts_list)
            ]
            vocab_rich = [vocabulary_richness(t) for t in texts]
            metrics.avg_quality = float(np.mean(qualities))
            metrics.avg_hallucination = float(np.mean(hallucinations))
            metrics.avg_vocab_richness = float(np.mean(vocab_rich))

        # --- Model utilization ---
        counts = Counter(experts)
        total = len(experts)
        metrics.model_utilization = {
            k: v / total for k, v in counts.items()
        }

        return metrics

    @staticmethod
    def format_confusion_matrix(conf: np.ndarray, labels: List[str]) -> str:
        rows = ["Confusion Matrix (true \\ predicted):"]
        header = " " * 15 + "".join(f"{l:>12}" for l in labels)
        rows.append(header)
        for i, label in enumerate(labels):
            row = f"{label:<15}" + "".join(f"{conf[i, j]:>12}" for j in range(len(labels)))
            rows.append(row)
        return "\n".join(rows)

    @staticmethod
    def format_latex_table(all_metrics: List[ModeMetrics]) -> str:
        """Generate LaTeX table for an IEEE paper."""
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Comparison of Routing Modes}",
            r"\label{tab:comparison}",
            r"\begin{tabular}{l" + "c" * len(all_metrics) + r"}",
            r"\toprule",
            "Metric & " + " & ".join(m.name for m in all_metrics) + r" \\",
            r"\midrule",
        ]
        rows = [
            ("Accuracy", lambda m: f"{m.accuracy:.3f}"),
            ("Precision", lambda m: f"{m.precision:.3f}"),
            ("Recall", lambda m: f"{m.recall:.3f}"),
            ("F1-Score", lambda m: f"{m.f1_score:.3f}"),
            ("Avg Latency (ms)", lambda m: f"{m.avg_latency_ms:.1f}"),
            ("P95 Latency (ms)", lambda m: f"{m.p95_latency_ms:.1f}"),
            ("Avg Energy (J)", lambda m: f"{m.avg_energy_j:.3f}"),
            ("Response Quality", lambda m: f"{m.avg_quality:.3f}"),
            ("Hallucination Rate", lambda m: f"{m.avg_hallucination:.3f}"),
            ("Vocab Richness", lambda m: f"{m.avg_vocab_richness:.3f}"),
        ]
        for name, fn in rows:
            line = f"{name} & " + " & ".join(fn(m) for m in all_metrics) + r" \\"
            lines.append(line)
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def mcnemar_test(
        predictions_a: List[str],
        predictions_b: List[str],
        ground_truth: List[str],
    ) -> float:
        """
        McNemar's test for comparing two classifiers.
        Returns p-value — lower = more significant difference.
        """
        b = c = 0
        for pa, pb, gt in zip(predictions_a, predictions_b, ground_truth):
            if pa == gt and pb != gt:
                c += 1
            elif pa != gt and pb == gt:
                b += 1
        denominator = b + c
        if denominator == 0:
            return 1.0
        chi2_val = (abs(b - c) - 1) ** 2 / denominator
        try:
            from scipy.stats import chi2 as chi2_dist
            return float(1.0 - chi2_dist.cdf(chi2_val, 1))
        except Exception:
            return 0.5

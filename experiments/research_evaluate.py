#!/usr/bin/env python3
"""
Research-Grade Evaluation for IEEE-style Comparison Paper
----------------------------------------------------------
Compares three routing modes:
  1. Council Mode (intelligent pipeline)
  2. Majority Voting (all models, vote)
  3. Dictatorship (all models, judge picks)

Metrics: accuracy, precision, recall, F1, latency, energy, quality, hallucination
Outputs: JSON results, LaTeX table, confusion matrices
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config.pipeline_config import EVAL
from src.models.adapters import ModelResponse
from src.orchestrator.comparison import ALL_EXPERTS, ComparisonRunner
from src.utils.research_metrics import ResearchEvaluator, ModeMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("research_eval")

# Ground truth labels for 20 research prompts
RESEARCH_PROMPTS: List[Tuple[str, str]] = [
    # (prompt, expected_expert)
    ("Write a Python function to implement binary search", "coding"),
    ("Debug this code: def foo(x): return x / 0", "coding"),
    ("Write a SQL query to join users and orders tables", "coding"),
    ("Create a REST API endpoint in FastAPI with authentication", "coding"),
    ("Implement a React component for a sortable data table", "coding"),
    ("Refactor this JavaScript callback into async/await", "coding"),
    ("Explain the steps to deploy a web application on AWS", "multimodal"),
    ("Plan a 3-day itinerary for Paris with budget estimates", "multimodal"),
    ("What are the pros and cons of microservices architecture", "multimodal"),
    ("Compare machine learning and deep learning approaches", "multimodal"),
    ("Analyze the environmental impact of electric vehicles", "multimodal"),
    ("Design a database schema for an e-commerce platform", "coding"),
    ("What is the capital of France?", "general"),
    ("Write a poem about the sea in Shakespearean style", "general"),
    ("Summarize the plot of Hamlet in three sentences", "general"),
    ("Tell me a programming joke", "general"),
    ("What is the boiling point of water in Fahrenheit?", "general"),
    ("Explain why the sky is blue to a 5-year-old", "general"),
    ("Write a short story about a robot learning to paint", "general"),
    ("What are the health benefits of drinking green tea?", "general"),
]

# Hallucination test prompts (known facts, easy to verify)
HALLUCINATION_PROMPTS: List[str] = [
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for water?",
    "How many continents are there?",
    "What color is the sky on a clear day?",
]


def run_trial(
    runner: ComparisonRunner,
    prompt: str,
    expected: str,
) -> dict:
    result = runner.run_comparison(prompt)

    selected = result.council_expert or "general"

    return {
        "prompt": prompt[:80],
        "expected": expected,
        "council_selected": selected,
        "majority_selected": result.majority_vote_result or "general",
        "dictator_selected": result.dictator_chosen_expert or "general",
        "council_latency_ms": result.council_latency,
        "voting_latency_ms": result.voting_latency,
        "dictatorship_latency_ms": result.dictatorship_latency,
        "task_type": result.classification.task_type_1 if result.classification else "N/A",
        "complexity": result.classification.prompt_complexity_score if result.classification else 0.0,
        "council_response": result.council_response.text if result.council_response else "",
        "majority_response": result.all_responses.get(result.majority_vote_result or "", ModelResponse("", "", 0)).text,
        "dictator_response": result.dictator_response.text if result.dictator_response else "",
        "all_responses": {
            k: v.text for k, v in result.all_responses.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Research-Grade Evaluation")
    parser.add_argument("--dummy", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of passes over the prompt set")
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    use_dummy = not args.live
    runner = ComparisonRunner(use_dummy=use_dummy)
    evaluator = ResearchEvaluator()

    # --- Collect predictions for each mode ---
    all_trials = []

    for trial in range(args.trials):
        logger.info("Trial %d/%d", trial + 1, args.trials)
        for prompt, expected in RESEARCH_PROMPTS:
            record = run_trial(runner, prompt, expected)
            all_trials.append(record)
        # Hallucination test
        for prompt in HALLUCINATION_PROMPTS:
            record = run_trial(runner, prompt, "general")
            all_trials.append(record)

    # --- Compute per-mode metrics ---
    council_preds = [(r["council_selected"], r["council_latency_ms"], r["council_response"])
                     for r in all_trials]
    majority_preds = [(r["majority_selected"], r["voting_latency_ms"], r["majority_response"])
                      for r in all_trials]
    dictator_preds = [(r["dictator_selected"], r["dictatorship_latency_ms"], r["dictator_response"])
                      for r in all_trials]
    ground_truth = [r["expected"] for r in all_trials]
    prompts_list = [r["prompt"] for r in all_trials]
    responses_list = [r["council_response"] for r in all_trials]

    council_metrics = evaluator.compute_mode_metrics(
        council_preds, ground_truth,
        responses=[r["council_response"] for r in all_trials],
        prompts=prompts_list,
    )
    council_metrics.name = "Council"

    majority_metrics = evaluator.compute_mode_metrics(
        majority_preds, ground_truth,
        responses=[r["majority_response"] for r in all_trials],
        prompts=prompts_list,
    )
    majority_metrics.name = "Majority Vote"

    dictator_metrics = evaluator.compute_mode_metrics(
        dictator_preds, ground_truth,
        responses=[r["dictator_response"] for r in all_trials],
        prompts=prompts_list,
    )
    dictator_metrics.name = "Dictatorship"

    all_metrics = [council_metrics, majority_metrics, dictator_metrics]

    # --- Print results ---
    print("\n" + "=" * 70)
    print("RESEARCH EVALUATION RESULTS")
    print("=" * 70)
    for m in all_metrics:
        print(f"\n--- {m.name} ---")
        print(f"  Accuracy:          {m.accuracy:.4f}")
        print(f"  Precision:         {m.precision:.4f}")
        print(f"  Recall:            {m.recall:.4f}")
        print(f"  F1-Score:          {m.f1_score:.4f}")
        print(f"  Avg Latency:       {m.avg_latency_ms:.1f} ms")
        print(f"  P95 Latency:       {m.p95_latency_ms:.1f} ms")
        print(f"  Avg Energy:        {m.avg_energy_j:.4f} J")
        print(f"  Response Quality:  {m.avg_quality:.4f}")
        print(f"  Hallucination:     {m.avg_hallucination:.4f}")
        print(f"  Vocab Richness:    {m.avg_vocab_richness:.4f}")
        print(f"  Model Utilization: {m.model_utilization}")
        if m.conf_matrix is not None:
            print(evaluator.format_confusion_matrix(m.conf_matrix, m.conf_matrix_labels))

    # --- McNemar test ---
    council_gt = [r["council_selected"] for r in all_trials]
    majority_gt = [r["majority_selected"] for r in all_trials]
    dictator_gt = [r["dictator_selected"] for r in all_trials]

    p_cv = evaluator.mcnemar_test(council_gt, majority_gt, ground_truth)
    p_cd = evaluator.mcnemar_test(council_gt, dictator_gt, ground_truth)
    p_vd = evaluator.mcnemar_test(majority_gt, dictator_gt, ground_truth)

    print(f"\n--- Statistical Significance (McNemar p-values) ---")
    print(f"  Council vs Majority:    {p_cv:.4f}")
    print(f"  Council vs Dictatorship: {p_cd:.4f}")
    print(f"  Majority vs Dictatorship: {p_vd:.4f}")

    # --- LaTeX table ---
    print("\n--- LaTeX Table for Paper ---")
    print(evaluator.format_latex_table(all_metrics))

    # --- Save ---
    if args.save:
        out_dir = os.path.join(EVAL.result_dir, "research")
        os.makedirs(out_dir, exist_ok=True)

        # Full results
        with open(os.path.join(out_dir, "all_trials.json"), "w") as f:
            json.dump(all_trials, f, indent=2, default=str)

        # Summary metrics
        summary = {
            m.name: {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "confusion_matrix": m.conf_matrix.tolist() if m.conf_matrix is not None else None,
                "avg_latency_ms": m.avg_latency_ms,
                "p95_latency_ms": m.p95_latency_ms,
                "avg_energy_j": m.avg_energy_j,
                "avg_quality": m.avg_quality,
                "avg_hallucination": m.avg_hallucination,
                "model_utilization": m.model_utilization,
            }
            for m in all_metrics
        }
        summary["mcnemar"] = {
            "council_vs_majority": p_cv,
            "council_vs_dictatorship": p_cd,
            "majority_vs_dictatorship": p_vd,
        }

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # LaTeX table
        with open(os.path.join(out_dir, "latex_table.tex"), "w") as f:
            f.write(evaluator.format_latex_table(all_metrics))

        logger.info("Results saved to %s", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

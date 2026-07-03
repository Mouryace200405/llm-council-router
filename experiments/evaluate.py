#!/usr/bin/env python3
"""
Evaluation Script
-----------------
Runs the routing pipeline against baseline single-model approaches and
reports on:
  - Routing accuracy
  - Energy and power efficiency
  - Resource utilization
  - Reduction in unnecessary execution of heavy models
  - Overall response quality

Usage:
    python experiments/evaluate.py --num-trials 30
    python experiments/evaluate.py --mode baseline-vs-pipeline
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config.pipeline_config import EVAL
from src.orchestrator.pipeline import LLMCouncilPipeline
from src.models.adapters import build_adapter

logging.basicConfig(
    level=getattr(logging, EVAL.log_level if hasattr(EVAL, "log_level") else "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("evaluate")


TEST_PROMPTS = [
    # Coding tasks
    ("Write a Python function to implement binary search", "coding"),
    ("Debug this: def foo(x): return x / 0", "coding"),
    ("Write a SQL query to join users and orders", "coding"),
    ("Implement a REST API endpoint in FastAPI", "coding"),
    ("Create a React component for a todo list", "coding"),
    # Reasoning / planning tasks
    ("Explain the steps to deploy a web application", "multimodal"),
    ("Plan a 3-day itinerary for Paris", "multimodal"),
    ("What are the pros and cons of microservices architecture?", "multimodal"),
    ("Compare machine learning and deep learning", "multimodal"),
    ("Analyze the environmental impact of electric vehicles", "multimodal"),
    # General / simple tasks
    ("What is the capital of France?", "general"),
    ("Write a poem about the sea", "general"),
    ("Summarize the plot of Hamlet", "general"),
    ("Tell me a joke", "general"),
    ("What is 2 + 2?", "general"),
]


def evaluate_pipeline(num_trials: int = 10, use_dummy: bool = True):
    pipeline = LLMCouncilPipeline(use_dummy_models=use_dummy)
    results = []

    for i in range(num_trials):
        for prompt, expected_expert in TEST_PROMPTS:
            t0 = time.time()
            result = pipeline.run(prompt)
            elapsed = (time.time() - t0) * 1000
            correct = result.routing.selected_expert == expected_expert
            results.append({
                "trial": i,
                "prompt": prompt[:60],
                "expected": expected_expert,
                "selected": result.routing.selected_expert,
                "confidence": result.routing.confidence,
                "task_type_1": result.classification.task_type_1,
                "complexity": result.classification.prompt_complexity_score,
                "latency_ms": elapsed,
                "energy_j": result.energy.total_energy_j if result.energy else 0.0,
                "correct": correct,
            })

    return results


def evaluate_baseline(num_trials: int = 5, use_dummy: bool = True):
    """Run the same prompts through each single model and collect energy data."""
    experts = ["coding", "multimodal", "general"]
    baseline_results = {}

    for expert in experts:
        adapter = build_adapter(expert, dummy=use_dummy)
        records = []
        for prompt, _ in TEST_PROMPTS:
            for _ in range(num_trials):
                t0 = time.time()
                resp = adapter.generate(prompt)
                elapsed = (time.time() - t0) * 1000
                records.append({
                    "expert": expert,
                    "prompt": prompt[:60],
                    "latency_ms": elapsed,
                })
        baseline_results[expert] = records

    return baseline_results


def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0

    latencies = [r["latency_ms"] for r in results]
    energies = [r["energy_j"] for r in results]

    return {
        "total_requests": total,
        "routing_accuracy": accuracy,
        "correct_routes": correct,
        "misroutes": total - correct,
        "avg_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "avg_energy_j": float(np.mean(energies)) if energies else 0.0,
        "total_energy_j": float(np.sum(energies)) if energies else 0.0,
        "routing_distribution": {
            "coding": sum(1 for r in results if r["selected"] == "coding"),
            "multimodal": sum(1 for r in results if r["selected"] == "multimodal"),
            "general": sum(1 for r in results if r["selected"] == "general"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Council Router")
    parser.add_argument("--num-trials", type=int, default=EVAL.num_trials,
                        help="Number of trials per prompt")
    parser.add_argument("--dummy", action="store_true", default=True,
                        help="Use dummy adapters")
    parser.add_argument("--live", action="store_true",
                        help="Use live HF models")
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["pipeline", "baseline", "baseline-vs-pipeline"],
                        help="Evaluation mode")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to disk")
    args = parser.parse_args()

    use_dummy = not args.live
    logger.info("Evaluation mode: %s | dummy=%s | trials=%d",
                args.mode, use_dummy, args.num_trials)

    if args.mode == "pipeline":
        logger.info("Running pipeline evaluation...")
        results = evaluate_pipeline(args.num_trials, use_dummy)
        metrics = compute_metrics(results)
        logger.info("Results:\n%s", json.dumps(metrics, indent=2))
        if args.save:
            path = os.path.join(EVAL.result_dir, "pipeline_eval.json")
            os.makedirs(EVAL.result_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"metrics": metrics, "results": results}, f, indent=2)
            logger.info("Saved to %s", path)

    elif args.mode == "baseline":
        logger.info("Running baseline evaluation...")
        baseline = evaluate_baseline(args.num_trials, use_dummy)
        for expert, records in baseline.items():
            latencies = [r["latency_ms"] for r in records]
            logger.info("%s: avg latency=%.1f ms", expert, np.mean(latencies))
        if args.save:
            path = os.path.join(EVAL.result_dir, "baseline_eval.json")
            os.makedirs(EVAL.result_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(baseline, f, indent=2)
            logger.info("Saved to %s", path)

    elif args.mode == "baseline-vs-pipeline":
        logger.info("Running baseline-vs-pipeline comparison...")
        pipeline_results = evaluate_pipeline(args.num_trials, use_dummy)
        baseline_results = evaluate_baseline(args.num_trials, use_dummy)
        pipeline_metrics = compute_metrics(pipeline_results)

        baseline_latencies = []
        for expert, records in baseline_results.items():
            baseline_latencies.extend(r["latency_ms"] for r in records)

        comparison = {
            "pipeline": pipeline_metrics,
            "baseline": {
                "avg_latency_ms": float(np.mean(baseline_latencies)),
                "total_requests": len(baseline_latencies),
            },
            "improvement": {
                "latency_reduction_pct": round(
                    (1 - pipeline_metrics["avg_latency_ms"] / np.mean(baseline_latencies)) * 100,
                    2,
                ) if baseline_latencies else 0,
            },
        }
        logger.info("Comparison:\n%s", json.dumps(comparison, indent=2))
        if args.save:
            path = os.path.join(EVAL.result_dir, "comparison.json")
            os.makedirs(EVAL.result_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(comparison, f, indent=2)
            logger.info("Saved to %s", path)


if __name__ == "__main__":
    main()

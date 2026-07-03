#!/usr/bin/env python3
"""
LLM Council Router — CLI entry point.

Usage:
    python src/main.py --prompt "Write a Python function"
    python src/main.py --interactive
    python src/main.py --dummy --interactive
    python src/main.py --benchmark

Evaluation:
    python src/main.py --evaluate --num-trials 20
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.pipeline_config import LOG_LEVEL, LOG_FORMAT, EVAL
from src.orchestrator.pipeline import LLMCouncilPipeline

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("llm-council")


def format_result(result):
    lines = []
    lines.append("=" * 70)
    lines.append("LLM COUNCIL ROUTER — RESULT")
    lines.append("=" * 70)
    lines.append(f"Original:     {result.original_prompt[:80]}...")
    lines.append(f"Enriched:     {result.enriched_prompt[:80]}...")
    lines.append("")
    lines.append("--- Classification ---")
    lines.append(f"  Task (Primary):    {result.classification.task_type_1}")
    lines.append(f"  Task (Secondary):  {result.classification.task_type_2}")
    lines.append(f"  Confidence:        {result.classification.task_type_prob:.3f}")
    lines.append(f"  Complexity:        {result.classification.prompt_complexity_score:.3f}")
    lines.append(f"  Reasoning:         {result.classification.reasoning:.3f}")
    lines.append(f"  Creativity:        {result.classification.creativity_scope:.3f}")
    lines.append(f"  Domain Knowledge:  {result.classification.domain_knowledge:.3f}")
    lines.append(f"  Constraints:       {result.classification.constraint_ct:.3f}")
    lines.append("")
    lines.append("--- Routing ---")
    lines.append(f"  Selected Expert:   {result.routing.selected_expert}")
    lines.append(f"  Confidence:        {result.routing.confidence:.3f}")
    lines.append(f"  Scores:            {result.routing.scores}")
    for reason in result.routing.reasons:
        lines.append(f"  Reason:            {reason}")
    lines.append("")
    lines.append("--- Latency ---")
    lines.append(f"  Enhance:   {result.latencies.get('enhance', 0):8.1f} ms")
    lines.append(f"  Classify:  {result.latencies.get('classify', 0):8.1f} ms")
    lines.append(f"  Inference: {result.latencies.get('inference', 0):8.1f} ms")
    lines.append(f"  Total:     {result.latencies.get('total', 0):8.1f} ms")
    if result.energy:
        lines.append("")
        lines.append("--- Energy ---")
        lines.append(f"  Total Energy:  {result.energy.total_energy_j:.3f} J")
        lines.append(f"  Avg Power:     {result.energy.avg_power_w:.3f} W")
        if result.energy.gpu_util_avg is not None:
            lines.append(f"  GPU Util:      {result.energy.gpu_util_avg:.1f} %")
    lines.append("")
    lines.append("--- Response ---")
    lines.append(f"  Model: {result.response.model_name}")
    lines.append(f"  {result.response.text[:500]}")
    lines.append("=" * 70)
    return "\n".join(lines)


def cmd_single(pipeline, args):
    result = pipeline.run(args.prompt)
    print(format_result(result))


def cmd_interactive(pipeline, args):
    print("\nLLM Council Router — Interactive Mode")
    print("Type your prompts.  'quit' to exit, 'summary' for stats.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "summary":
            s = pipeline.metrics.summary()
            import json
            print(json.dumps(s, indent=2))
            continue
        result = pipeline.run_interactive(prompt)
        print("\n" + format_result(result))


def cmd_benchmark(pipeline, args):
    test_prompts = [
        "Write a Python function to sort a list of integers.",
        "Explain the theory of relativity in simple terms.",
        "What is the capital of France?",
        "Write a poem about autumn.",
        "Debug this code: def foo(x): return x / 0",
        "Summarize the plot of Hamlet.",
        "Create a REST API endpoint for user authentication.",
        "What are the benefits of renewable energy?",
    ]
    print(f"\nBenchmarking {len(test_prompts)} prompts...")
    for i, p in enumerate(test_prompts):
        t0 = time.time()
        result = pipeline.run(p)
        elapsed = (time.time() - t0) * 1000
        print(f"  [{i+1}/{len(test_prompts)}] routed to '{result.routing.selected_expert}' "
              f"({result.routing.confidence:.2f}) — {elapsed:.0f} ms")
    summary = pipeline.metrics.summary()
    print(f"\nBenchmark complete.  {summary['total_requests']} requests.")
    print(f"  Avg latency:  {summary['avg_total_latency_ms']:.1f} ms")
    print(f"  Avg energy:   {summary['avg_energy_j']:.3f} J")
    print(f"  Model usage:  {summary['model_usage']}")
    pipeline.metrics.save()


def cmd_demo(pipeline, args):
    """Run a demonstration with detailed output."""
    prompts = [
        "Write a Python function for binary search",
        "Explain how neural networks work",
        "What is the weather in London?",
    ]
    for p in prompts:
        result = pipeline.run(p)
        print(format_result(result))
        print()


def main():
    parser = argparse.ArgumentParser(description="LLM Council Router")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--dummy", "-d", action="store_true",
                        help="Use dummy adapters (no network calls)")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample prompts")
    parser.add_argument("--enhancer-device", type=int, default=None,
                        help="Device for enhancer (0=GPU, -1=CPU)")
    parser.add_argument("--classifier-device", type=int, default=None,
                        help="Device for classifier (0=GPU, -1=CPU)")
    args = parser.parse_args()

    pipeline = LLMCouncilPipeline(
        use_dummy_models=args.dummy,
        enable_energy_monitoring=True,
        enhancer_device=args.enhancer_device,
        classifier_device=args.classifier_device,
    )

    if args.prompt:
        cmd_single(pipeline, args)
    elif args.interactive:
        cmd_interactive(pipeline, args)
    elif args.benchmark:
        cmd_benchmark(pipeline, args)
    elif args.demo:
        cmd_demo(pipeline, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

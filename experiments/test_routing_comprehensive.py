#!/usr/bin/env python3
"""
Comprehensive routing test across diverse prompts.
Saves results + classifier outputs to a file.
Displays expected vs actual routing decisions.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)

from src.engines.classifier import PromptClassifier
from src.orchestrator.router import Router

# ── Diverse test prompts ──────────────────────────────────────────────

PROMPTS = {
    # === CODING ===
    "binary_search": "Write a Python function that implements binary search on a sorted array",
    "sql_join": "Write a SQL query to join users, orders, and products tables to find the top 10 customers by revenue",
    "rest_api": "Create a FastAPI REST API endpoint with authentication, input validation, and error handling for a blog CRUD system",
    "debug_code": "Debug this code: def foo(lst): return sum(lst) / len(lst)  # crashes on empty list",
    "leetcode": "Given an array of integers, return indices of two numbers that add up to a target. Solve in O(n) time.",
    "react_component": "Write a React component that fetches data from an API, handles loading/error states, and renders a list",

    # === GENERAL (QA, summarization, explanation) ===
    "capital_france": "What is the capital of France?",
    "explain_microservices": "Explain microservices architecture: pros, cons, and when to use it",
    "ml_vs_dl": "Compare machine learning and deep learning — what are the key differences?",
    "why_sky_blue": "Explain why the sky is blue to a 5-year-old",
    "hamlet_summary": "Summarize the plot of Hamlet in 3 paragraphs",
    "climate_change": "What are the main causes and effects of climate change?",
    "programming_joke": "Tell me a programming joke",
    "python_vs_java": "Compare Python and Java for backend development",

    # === CREATIVE / MULTIMODAL ===
    "poem_sea": "Write a poem about the sea",
    "short_story": "Write a short story about a robot that learns to paint",
    "marketing_slogan": "Create a marketing slogan for a new eco-friendly water bottle brand",
    "song_lyrics": "Write song lyrics about overcoming challenges",
    "blog_intro": "Write an engaging introduction for a blog post about remote work productivity",
    "script_dialogue": "Write a short dialogue between two friends reuniting after 10 years",

    # === AMBIGUOUS / EDGE ===
    "travel_plan": "Plan a 3-day itinerary for Paris including museums, restaurants, and parks",
    "recipe": "Give me a recipe for chocolate chip cookies",
    "workout_plan": "Create a weekly workout plan for beginners",
    "interview_questions": "List 5 common JavaScript interview questions with answers",
    "data_analysis_prompt": "Analyze this CSV data and tell me the trends: sales, region, quarter",
}

# ── Expected routing (ground truth based on prompt content) ───────────

EXPECTED = {
    "binary_search": "coding",
    "sql_join": "coding",
    "rest_api": "coding",
    "debug_code": "coding",
    "leetcode": "coding",
    "react_component": "coding",
    "capital_france": "general",
    "explain_microservices": "general",
    "ml_vs_dl": "general",
    "why_sky_blue": "general",
    "hamlet_summary": "general",
    "climate_change": "general",
    "programming_joke": "general",
    "python_vs_java": "coding",  # borderline — could be coding or general
    "poem_sea": "multimodal",  # creative → multimodal
    "short_story": "multimodal",
    "marketing_slogan": "multimodal",
    "song_lyrics": "multimodal",
    "blog_intro": "multimodal",
    "script_dialogue": "multimodal",
    "travel_plan": "general",  # planning but not creative
    "recipe": "general",
    "workout_plan": "general",
    "interview_questions": "coding",  # JS questions → coding
    "data_analysis_prompt": "coding",  # analysis code → coding
}

# ── Analysis ──────────────────────────────────────────────────────────

ANALYSIS = {
    "binary_search": "Pure coding task",
    "sql_join": "SQL coding",
    "rest_api": "API coding",
    "debug_code": "Debugging = coding",
    "leetcode": "Algorithm = coding",
    "react_component": "Web dev = coding",
    "capital_france": "Factual Q&A = general",
    "explain_microservices": "Technical explanation = general",
    "ml_vs_dl": "Comparison = general",
    "why_sky_blue": "Simplified explanation = general",
    "hamlet_summary": "Summarization = general",
    "climate_change": "Factual explanation = general",
    "programming_joke": "Joke = general (casual)",
    "python_vs_java": "Technical comparison = coding or general",
    "poem_sea": "Creative writing = multimodal",
    "short_story": "Creative writing = multimodal",
    "marketing_slogan": "Creative marketing = multimodal",
    "song_lyrics": "Creative writing = multimodal",
    "blog_intro": "Creative writing = multimodal",
    "script_dialogue": "Creative dialogue = multimodal",
    "travel_plan": "Planning but factual = general",
    "recipe": "Instructional = general",
    "workout_plan": "Structured plan = general",
    "interview_questions": "JS questions = coding",
    "data_analysis_prompt": "Data analysis code = coding",
}


def main():
    print("=" * 80)
    print("COMPREHENSIVE ROUTING TEST")
    print("=" * 80)

    classifier = PromptClassifier()
    router = Router()

    results = []
    stats = {"correct": 0, "wrong": 0, "total": 0}
    expert_counts = {"coding": 0, "general": 0, "multimodal": 0}
    expected_counts = {"coding": 0, "general": 0, "multimodal": 0}
    confusion = {"coding": {"coding": 0, "general": 0, "multimodal": 0},
                  "general": {"coding": 0, "general": 0, "multimodal": 0},
                  "multimodal": {"coding": 0, "general": 0, "multimodal": 0}}

    for key, prompt in PROMPTS.items():
        print(f"\n{'─' * 60}")
        print(f"  [{key}] {prompt[:60]}...")
        print(f"{'─' * 60}")

        cls = classifier.classify(prompt)
        decision = router.route(cls, raw_prompt=prompt)

        expected = EXPECTED[key]
        expected_counts[expected] += 1
        expert_counts[decision.selected_expert] += 1
        stats["total"] += 1

        is_correct = decision.selected_expert == expected
        confusion[expected][decision.selected_expert] += 1

        if is_correct:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1

        print(f"  Expected:  {expected.upper():<12s} | "
              f"Got: {decision.selected_expert.upper():<12s} | "
              f"{'✓' if is_correct else '✗'}")
        print(f"  Confidence: {decision.confidence:.4f}")
        print(f"  Scores:    coding={decision.scores.get('coding',0):.4f}  "
              f"general={decision.scores.get('general',0):.4f}  "
              f"multimodal={decision.scores.get('multimodal',0):.4f}")
        print(f"  Classifier:")
        print(f"    Task Type:       {cls.task_type_1} / {cls.task_type_2}")
        print(f"    Complexity:      {cls.prompt_complexity_score:.4f}")
        print(f"    Creativity:      {cls.creativity_scope:.4f}")
        print(f"    Reasoning:       {cls.reasoning:.4f}")
        print(f"    Domain Know:     {cls.domain_knowledge:.4f}")
        print(f"    Context Know:    {cls.contextual_knowledge:.4f}")
        print(f"    Constraints:     {cls.constraint_ct:.4f}")
        print(f"    Few-Shots:       {cls.number_of_few_shots:.4f}")
        print(f"  Reasons:")
        for r in decision.reasons:
            print(f"    {r}")

        results.append({
            "key": key,
            "prompt": prompt,
            "expected": expected,
            "got": decision.selected_expert,
            "confidence": decision.confidence,
            "scores": decision.scores,
            "classification": {
                "task_type_1": cls.task_type_1,
                "task_type_2": cls.task_type_2,
                "creativity_scope": cls.creativity_scope,
                "reasoning": cls.reasoning,
                "domain_knowledge": cls.domain_knowledge,
                "contextual_knowledge": cls.contextual_knowledge,
                "constraint_ct": cls.constraint_ct,
                "prompt_complexity_score": cls.prompt_complexity_score,
                "number_of_few_shots": cls.number_of_few_shots,
            },
            "analysis": ANALYSIS[key],
        })

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    accuracy = stats["correct"] / stats["total"] * 100
    print(f"  Accuracy: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
    print()

    print("  Confusion Matrix (expected rows → got columns):")
    print(f"  {'':>14s} {'coding':>10s} {'general':>10s} {'multimodal':>10s}")
    for exp in ["coding", "general", "multimodal"]:
        row = confusion[exp]
        print(f"  {exp:>12s}  {row['coding']:>10d} {row['general']:>10d} {row['multimodal']:>10d}")

    print()
    print(f"  Expected distribution: {dict(expected_counts)}")
    print(f"  Actual distribution:   {dict(expert_counts)}")

    # Save detailed results
    out_path = Path("results/routing_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "accuracy": round(accuracy, 2),
        "correct": stats["correct"],
        "total": stats["total"],
        "confusion_matrix": confusion,
        "expected_distribution": expected_counts,
        "actual_distribution": expert_counts,
        "results": results,
    }, indent=2))
    print(f"\n  Detailed results saved to {out_path}")

    # Print wrong cases
    wrong = [r for r in results if r["expected"] != r["got"]]
    if wrong:
        print(f"\n  WRONG ROUTING ({len(wrong)} cases):")
        for w in wrong:
            print(f"    [{w['key']}] expected={w['expected']}, got={w['got']}")
            print(f"      Scores: {w['scores']}")
            print(f"      Analysis: {w['analysis']}")
            print(f"      Task: {w['classification']['task_type_1']} | "
                  f"Creativity={w['classification']['creativity_scope']:.4f} | "
                  f"Reasoning={w['classification']['reasoning']:.4f} | "
                  f"Domain={w['classification']['domain_knowledge']:.4f} | "
                  f"Constraint={w['classification']['constraint_ct']:.4f}")

    return accuracy


if __name__ == "__main__":
    main()

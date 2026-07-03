#!/usr/bin/env python3
"""
Simulate different router algorithms against cached classifier outputs.
No GPU needed — uses results from test_routing_comprehensive.json.
Allows rapid iteration on routing thresholds.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engines.classifier import ClassificationResult

# ── Load cached results ───────────────────────────────────────────────
cache_path = Path("results/routing_test_results.json")
if not cache_path.exists():
    print("ERROR: Run experiments/test_routing_comprehensive.py first")
    sys.exit(1)

data = json.loads(cache_path.read_text())

# ── Expected labels ───────────────────────────────────────────────────
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
    "python_vs_java": "coding",
    "poem_sea": "multimodal",
    "short_story": "multimodal",
    "marketing_slogan": "multimodal",
    "song_lyrics": "multimodal",
    "blog_intro": "multimodal",
    "script_dialogue": "multimodal",
    "travel_plan": "general",
    "recipe": "general",
    "workout_plan": "general",
    "interview_questions": "coding",
    "data_analysis_prompt": "coding",
}


# ── Router implementations ────────────────────────────────────────────

class OldRouter:
    """Original buggy router (baseline)."""

    def route(self, cls):
        scores = {}

        g = self._score_general(cls)
        scores["general"] = g[0]

        c = self._score_coding(cls)
        scores["coding"] = c[0]

        m = self._score_multimodal(cls)
        scores["multimodal"] = m[0]

        return max(scores, key=scores.get), scores

    def _score_general(self, cls):
        score = 0.0
        if cls.reasoning > 0.030:
            score += 0.30
        if cls.constraint_ct > 0.010:
            score += 0.20
        if cls.creativity_scope <= 0.025 and cls.domain_knowledge > 0.50:
            score += 0.15
        if cls.task_type_1 in {"Closed QA", "Chatbot", "Open QA"}:
            score += 0.10
        return score, ""

    def _score_coding(self, cls):
        if cls.reasoning >= 0.020 and cls.task_type_1 != "Code Generation":
            return 0.0, ""
        score = 0.0
        if cls.creativity_scope <= 0.025:
            score += 0.35
        if cls.domain_knowledge > 0.50:
            score += 0.30
        if cls.constraint_ct <= 0.015:
            score += 0.20
        if cls.task_type_1 == "Code Generation":
            score += 0.20
        score = min(score, 0.95)
        return score, ""

    def _score_multimodal(self, cls):
        score = 0.0
        if cls.creativity_scope > 0.04:
            score += 0.30
            if cls.creativity_scope > 0.10:
                score += 0.20
                if cls.creativity_scope > 0.20:
                    score += 0.10
        if cls.domain_knowledge < 0.50 and cls.creativity_scope > 0.025:
            score += 0.20
        if cls.contextual_knowledge > 0.30:
            score += 0.10
        if cls.task_type_1 in {"Text Generation", "Brainstorming"}:
            score += 0.15
        score = min(score, 0.95)
        return score, ""


class NewRouter:
    """Improved router — stricter gates, general base score, better multimodal detection."""

    def route(self, cls):
        scores = {}

        g = self._score_general(cls)
        scores["general"] = g[0]

        c = self._score_coding(cls)
        scores["coding"] = c[0]

        m = self._score_multimodal(cls)
        scores["multimodal"] = m[0]

        best = max(scores, key=scores.get)
        return best, scores

    def _score_general(self, cls):
        score = 0.20  # base
        if cls.reasoning > 0.025:
            score += 0.25
        if cls.constraint_ct > 0.04:
            score += 0.25
        if cls.domain_knowledge > 0.50 and cls.creativity_scope > 0.02:
            score += 0.15
        if cls.task_type_1 in {"Closed QA", "Open QA", "Chatbot", "Summarization"}:
            score += 0.10
        return score, ""

    def _score_coding(self, cls):
        if cls.reasoning > 0.03 and cls.task_type_1 != "Code Generation":
            return 0.0, ""
        if cls.creativity_scope > 0.025:
            return 0.0, ""
        if cls.constraint_ct > 0.04:
            return 0.0, ""

        score = 0.0
        if cls.domain_knowledge > 0.60:
            score += 0.50
        elif cls.domain_knowledge > 0.40:
            score += 0.40
        else:
            score += 0.30
        if cls.constraint_ct <= 0.005:
            score += 0.10
        if cls.task_type_1 == "Code Generation":
            score += 0.25
        score = min(score, 0.95)
        return score, ""

    def _score_multimodal(self, cls):
        score = 0.0
        if cls.creativity_scope > 0.04:
            score += 0.35
            if cls.creativity_scope > 0.10:
                score += 0.15
            if cls.creativity_scope > 0.20:
                score += 0.15
        if cls.domain_knowledge < 0.50 and cls.creativity_scope > 0.025:
            score += 0.20
        if cls.contextual_knowledge > 0.30:
            score += 0.10
        if cls.task_type_2 in {"Text Generation", "Brainstorming", "Rewrite"}:
            score += 0.15
        score = min(score, 0.95)
        return score, ""


class TentativeRouter:
    """
    More aggressive multimodal detection:
    - Lower creativity threshold for multimodal (0.025 instead of 0.04)
    - Higher general base for safety
    - Coding gated more carefully
    """

    def route(self, cls):
        scores = {}
        scores["general"] = self._score_general(cls)[0]
        scores["coding"] = self._score_coding(cls)[0]
        scores["multimodal"] = self._score_multimodal(cls)[0]
        best = max(scores, key=scores.get)
        return best, scores

    def _score_general(self, cls):
        score = 0.25
        if cls.reasoning > 0.02:
            score += 0.25
        if cls.constraint_ct > 0.03:
            score += 0.20
        if cls.task_type_1 in {"Closed QA", "Open QA", "Chatbot", "Summarization"}:
            score += 0.10
        return score, ""

    def _score_coding(self, cls):
        if cls.reasoning > 0.03 and cls.task_type_1 != "Code Generation":
            return 0.0, ""
        if cls.creativity_scope > 0.025 and cls.task_type_1 != "Code Generation":
            return 0.0, ""
        if cls.constraint_ct > 0.04:
            return 0.0, ""

        score = 0.0
        if cls.creativity_scope <= 0.025 and cls.domain_knowledge > 0.40:
            score += 0.45
        if cls.domain_knowledge > 0.60:
            score += 0.20
        elif cls.domain_knowledge > 0.40:
            score += 0.10
        if cls.task_type_1 == "Code Generation":
            score += 0.25
        score = min(score, 0.95)
        return score, ""

    def _score_multimodal(self, cls):
        score = 0.0
        if cls.creativity_scope > 0.03:
            score += 0.30
            if cls.creativity_scope > 0.10:
                score += 0.20
        if cls.domain_knowledge < 0.50 and cls.creativity_scope > 0.02:
            score += 0.20
        if cls.contextual_knowledge > 0.30:
            score += 0.10
        if cls.task_type_2 in {"Text Generation", "Brainstorming", "Rewrite"}:
            score += 0.15
        score = min(score, 0.95)
        return score, ""


def test_router(router_cls, name):
    results = data["results"]
    correct = 0
    total = len(results)
    confusion = {e: {e2: 0 for e2 in ("coding", "general", "multimodal")}
                 for e in ("coding", "general", "multimodal")}
    wrong = []

    for r in results:
        key = r["key"]
        cls = ClassificationResult(
            task_type_1=r["classification"]["task_type_1"],
            task_type_2=r["classification"]["task_type_2"],
            task_type_prob=0.0,
            creativity_scope=r["classification"]["creativity_scope"],
            reasoning=r["classification"]["reasoning"],
            domain_knowledge=r["classification"]["domain_knowledge"],
            contextual_knowledge=r["classification"]["contextual_knowledge"],
            constraint_ct=r["classification"]["constraint_ct"],
            number_of_few_shots=r["classification"]["number_of_few_shots"],
            prompt_complexity_score=r["classification"]["prompt_complexity_score"],
        )

        router = router_cls()
        got, scores = router.route(cls)
        exp = EXPECTED[key]

        confusion[exp][got] += 1
        if got == exp:
            correct += 1
        else:
            wrong.append((key, exp, got, scores))

    accuracy = correct / total * 100
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"\n  Confusion Matrix (expected rows → got cols):")
    print(f"  {'':>14s} {'coding':>10s} {'general':>10s} {'multimodal':>10s}")
    for exp in ("coding", "general", "multimodal"):
        row = confusion[exp]
        print(f"  {exp:>12s}  {row['coding']:>10d} {row['general']:>10d} {row['multimodal']:>10d}")

    if wrong:
        print(f"\n  Wrong ({len(wrong)}):")
        for key, exp, got, scores in wrong:
            sig = " <<<" if "general" in scores and scores["general"] >= scores.get("coding", 0) else ""
            print(f"    {key:<25s} exp={exp:<10s} got={got:<10s}  "
                  f"g={scores.get('general',0):.2f} c={scores.get('coding',0):.2f} "
                  f"m={scores.get('multimodal',0):.2f}{sig}")

    return accuracy


if __name__ == "__main__":
    test_router(OldRouter, "OLD ROUTER (current, buggy)")
    test_router(NewRouter, "NEW ROUTER (stricter gates, general base)")
    test_router(TentativeRouter, "TENTATIVE ROUTER (lower multimodal threshold)")

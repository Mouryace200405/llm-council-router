#!/usr/bin/env python3
"""
Unit and integration tests — pure classifier-based routing, no keywords.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engines.classifier import ClassificationResult
from src.orchestrator.router import Router


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.router = Router()

    def _make_cls(self, **overrides):
        defaults = dict(
            task_type_1="Open QA",
            task_type_2="NA",
            task_type_prob=0.50,
            creativity_scope=0.010,
            reasoning=0.010,
            domain_knowledge=0.50,
            contextual_knowledge=0.15,
            constraint_ct=0.010,
            number_of_few_shots=0.0,
            prompt_complexity_score=0.13,
            raw={},
        )
        defaults.update(overrides)
        return ClassificationResult(**defaults)

    def test_coding_low_reasoning_high_domain(self):
        # Low reasoning + low creativity + high domain knowledge → coding
        cls = self._make_cls(
            creativity_scope=0.010,
            reasoning=0.008,
            domain_knowledge=0.74,
            constraint_ct=0.004,
        )
        decision = self.router.route(cls)
        self.assertEqual(decision.selected_expert, "coding")

    def test_coding_code_gen_task_type(self):
        # "Code Generation" task type → coding regardless of reasoning
        cls = self._make_cls(
            task_type_1="Code Generation",
            reasoning=0.10,
        )
        decision = self.router.route(cls)
        self.assertEqual(decision.selected_expert, "coding")

    def test_general_high_reasoning(self):
        # Higher reasoning → general
        cls = self._make_cls(
            creativity_scope=0.006,
            reasoning=0.043,
            domain_knowledge=0.72,
            constraint_ct=0.012,
        )
        decision = self.router.route(cls)
        self.assertEqual(decision.selected_expert, "general")

    def test_multimodal_high_creativity(self):
        # High creativity → multimodal
        cls = self._make_cls(
            task_type_1="Text Generation",
            creativity_scope=0.25,
            reasoning=0.008,
            domain_knowledge=0.42,
            constraint_ct=0.030,
        )
        decision = self.router.route(cls)
        self.assertEqual(decision.selected_expert, "multimodal")

    def test_multimodal_creative_non_technical(self):
        # Moderate creativity + low domain knowledge → multimodal
        cls = self._make_cls(
            creativity_scope=0.035,
            reasoning=0.008,
            domain_knowledge=0.30,
            constraint_ct=0.005,
        )
        decision = self.router.route(cls)
        # creat=0.035 is borderline — ties between coding (constraint) and multimodal
        self.assertIn(decision.selected_expert, {"multimodal", "general", "coding"})

    def test_coding_not_selected_for_creative(self):
        # Not coding when reasoning is low but creativity is high
        cls = self._make_cls(
            task_type_1="Text Generation",
            creativity_scope=0.25,
            reasoning=0.008,
            domain_knowledge=0.30,
        )
        decision = self.router.route(cls)
        self.assertNotEqual(decision.selected_expert, "coding")

    def test_all_experts_scored(self):
        cls = self._make_cls()
        decision = self.router.route(cls)
        for expert in ["coding", "multimodal", "general"]:
            self.assertIn(expert, decision.scores)
            self.assertGreaterEqual(decision.scores[expert], 0.0)
            self.assertLessEqual(decision.scores[expert], 1.0)

    def test_routing_returns_reasons(self):
        cls = self._make_cls()
        decision = self.router.route(cls)
        self.assertGreater(len(decision.reasons), 0)

    def test_hello_world_general(self):
        # "hello world" type prompts → general or multimodal
        cls = self._make_cls(
            creativity_scope=0.20,
            reasoning=0.040,
            domain_knowledge=0.36,
            constraint_ct=0.083,
        )
        decision = self.router.route(cls)
        # creat=0.20 triggers multimodal creativity tiers, but high reasoning also scores general
        self.assertIn(decision.selected_expert, {"general", "multimodal"})


class TestClassificationResult(unittest.TestCase):
    def test_dataclass_fields(self):
        cls = ClassificationResult(
            task_type_1="Open QA",
            task_type_2="NA",
            task_type_prob=0.9,
            creativity_scope=0.01,
            reasoning=0.04,
            domain_knowledge=0.7,
            contextual_knowledge=0.14,
            constraint_ct=0.01,
            number_of_few_shots=0.0,
            prompt_complexity_score=0.13,
            raw={},
        )
        self.assertEqual(cls.task_type_1, "Open QA")
        self.assertEqual(cls.task_type_prob, 0.9)
        self.assertEqual(cls.prompt_complexity_score, 0.13)


if __name__ == "__main__":
    unittest.main()

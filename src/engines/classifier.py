"""
Prompt Classifier Engine
------------------------
NVIDIA prompt-task-and-complexity classifier that produces structured metadata:
  - task_category (primary + secondary)
  - reasoning, creativity, domain_knowledge, contextual_knowledge,
    constraints, few-shot complexity
  - overall prompt_complexity_score

This model does NOT route. It only provides structured understanding.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from config.pipeline_config import CLASSIFIER_PATH, CLASSIFIER_BACKBONE_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch components (mirrored exactly from the spec)
# ---------------------------------------------------------------------------
class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, 1)
        sum_mask = mask.sum(1).clamp(min=1e-9)
        return sum_emb / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module):
    """Local re-implementation — mirrors NVIDIA's exact `add_module` pattern."""

    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            CLASSIFIER_BACKBONE_PATH, weights_only=False
        )
        self.target_sizes = target_sizes
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in target_sizes.values()
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    # ------------------------------------------------------------------
    # Classification methods (identical logic to spec)
    # ------------------------------------------------------------------
    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(v, 3) for v in sublist] for sublist in top2_prob
            ]

            for i, sublist in enumerate(top2_prob_rounded):
                if sublist[1] < 0.1:
                    top2_strings[i][1] = "NA"

            return (
                [s[0] for s in top2_strings],
                [s[1] for s in top2_strings],
                [s[0] for s in top2_prob_rounded],
            )
        else:
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = (weighted_sum / self.divisor_map[target]).tolist()
            scores = [round(v, decimal) for v in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits):
        result = {}

        tt = self.compute_results(logits[0], target="task_type")
        result["task_type_1"] = tt[0]
        result["task_type_2"] = tt[1]
        result["task_type_prob"] = tt[2]

        result["creativity_scope"] = self.compute_results(logits[1], "creativity_scope")
        result["reasoning"] = self.compute_results(logits[2], "reasoning")
        result["contextual_knowledge"] = self.compute_results(logits[3], "contextual_knowledge")
        result["number_of_few_shots"] = self.compute_results(logits[4], "number_of_few_shots")
        result["domain_knowledge"] = self.compute_results(logits[5], "domain_knowledge")
        result["no_label_reason"] = self.compute_results(logits[6], "no_label_reason")
        result["constraint_ct"] = self.compute_results(logits[7], "constraint_ct")

        result["prompt_complexity_score"] = [
            round(
                0.35 * cr + 0.25 * re + 0.15 * co + 0.15 * dk + 0.05 * ck + 0.05 * fs,
                5,
            )
            for cr, re, co, dk, ck, fs in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]
        return result

    def forward(self, batch):
        outputs = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
        logits = [head(pooled) for head in self.heads]
        return self.process_logits(logits)


# ---------------------------------------------------------------------------
# Public classifier interface
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    task_type_1: str
    task_type_2: str
    task_type_prob: float
    creativity_scope: float
    reasoning: float
    domain_knowledge: float
    contextual_knowledge: float
    constraint_ct: float
    number_of_few_shots: float
    prompt_complexity_score: float
    raw: dict = field(default_factory=dict)


class PromptClassifier:
    """Wraps the NVIDIA classifier model for structured prompt understanding."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[int] = None):
        self.model_path = model_path or CLASSIFIER_PATH
        self._device = device if device is not None else (
            0 if torch.cuda.is_available() else -1
        )
        self._model: Optional[CustomModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, prompt: str) -> ClassificationResult:
        model, tokenizer = self._load()
        encoded = tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        if self._device >= 0:
            encoded = {k: v.to(f"cuda:{self._device}") for k, v in encoded.items()}

        with torch.no_grad():
            result = model(encoded)

        return self._to_dataclass(result)

    def classify_batch(self, prompts: list[str]) -> list[ClassificationResult]:
        return [self.classify(p) for p in prompts]

    def unload(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Classifier unloaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load_config(self):
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        return (
            cfg["target_sizes"],
            cfg["task_type_map"],
            cfg["weights_map"],
            cfg["divisor_map"],
        )

    def _load(self) -> Tuple[CustomModel, AutoTokenizer]:
        if self._loaded:
            return self._model, self._tokenizer  # type: ignore

        logger.info("Loading classifier from %s", self.model_path)

        target_sizes, task_type_map, weights_map, divisor_map = self._load_config()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, fix_mistral_regex=True
        )

        model = CustomModel(target_sizes, task_type_map, weights_map, divisor_map)
        safetensors_file = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_file):
            logger.info("Loading safetensors weights")
            state_dict = load_file(safetensors_file)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No safetensors found — using randomly initialized weights")

        model.eval()
        if self._device >= 0:
            model.to(f"cuda:{self._device}")

        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True
        return model, tokenizer

    @staticmethod
    def _to_dataclass(result: dict) -> ClassificationResult:
        return ClassificationResult(
            task_type_1=result["task_type_1"][0],
            task_type_2=result["task_type_2"][0],
            task_type_prob=result["task_type_prob"][0],
            creativity_scope=result["creativity_scope"][0],
            reasoning=result["reasoning"][0],
            domain_knowledge=result["domain_knowledge"][0],
            contextual_knowledge=result["contextual_knowledge"][0],
            constraint_ct=result["constraint_ct"][0],
            number_of_few_shots=result["number_of_few_shots"][0],
            prompt_complexity_score=result["prompt_complexity_score"][0],
            raw=result,
        )
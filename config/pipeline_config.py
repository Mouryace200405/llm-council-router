"""
LLM Council Router — Central Pipeline Configuration
All model paths, HF endpoints, routing thresholds, and evaluation settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Model paths (local)
# ---------------------------------------------------------------------------
PROMPT_ENHANCER_PATH = "/home/ivan/models/lfm2.5-prompt-enhancer"
CLASSIFIER_PATH = "/home/ivan/prompt-task-and-complexity-classifier"
CLASSIFIER_BACKBONE_PATH = "/home/ivan/deberta-v3-base"

# ---------------------------------------------------------------------------
# HF Inference endpoints (cloud)
# ---------------------------------------------------------------------------
HF_BASE_URL = "https://router.huggingface.co/v1"

# Mapping: internal name -> (HF model ID, provider suffix if any)
EXPERT_MODELS: Dict[str, Dict] = {
    "coding": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "provider": "nscale",
        "description": "Coding, programming, software engineering tasks",
        "method": "openai",
    },
    "multimodal": {
        "model_id": "Qwen/Qwen3.5-9B",
        "provider": "together",
        "description": "Creative/long-form text, brainstorming, rewriting (NOTE: text-only — no actual vision/RAG; uses Qwen3.5-9B for creative text tasks)",
        "method": "openai",
    },
    "general": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "provider": "nscale",
        "description": "General-purpose writing, QA, summarization, default fallback",
        "method": "openai",
    },
}

DEFAULT_EXPERT = "general"


# ---------------------------------------------------------------------------
# Routing thresholds  (derived from classifier output)
# ---------------------------------------------------------------------------
@dataclass
class RoutingThresholds:
    """Scores are 0-1. Controls routing decisions via weighted heuristics."""

    clarification_threshold: float = 0.30


THRESHOLDS = RoutingThresholds()

# ---------------------------------------------------------------------------
# Clarification settings
# ---------------------------------------------------------------------------
CLARIFICATION_ENABLED = True
CLARIFICATION_THRESHOLD_SCORE = 0.3  # below this → ask user
MAX_CLARIFICATION_ROUNDS = 2

# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    num_trials: int = 5
    warmup_trials: int = 2
    energy_sample_rate_hz: int = 10  # sampling frequency for power measurements
    result_dir: str = "results"
    log_dir: str = "results/logs"
    save_predictions: bool = True


EVAL = EvalConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
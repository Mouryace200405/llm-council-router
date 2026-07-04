"""
LLM Council Router — Central Pipeline Configuration
All model paths, HF endpoints, routing thresholds, and evaluation settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Inference backend
# ---------------------------------------------------------------------------
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "ollama")

# ---------------------------------------------------------------------------
# Model paths (local)
# ---------------------------------------------------------------------------
PROMPT_ENHANCER_PATH = "/home/ivan/models/lfm2.5-prompt-enhancer"
CLASSIFIER_PATH = "/home/ivan/prompt-task-and-complexity-classifier"
CLASSIFIER_BACKBONE_PATH = "/home/ivan/deberta-v3-base"

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
HF_BASE_URL = "https://router.huggingface.co/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def get_inference_base_url() -> str:
    backend = INFERENCE_BACKEND.lower()
    if backend == "ollama":
        return OLLAMA_BASE_URL
    elif backend == "together":
        return TOGETHER_BASE_URL
    else:
        return HF_BASE_URL


def get_inference_api_key() -> str:
    backend = INFERENCE_BACKEND.lower()
    if backend == "ollama":
        return "ollama"  # placeholder — Ollama ignores the key
    elif backend == "together":
        return os.environ.get("TOGETHER_API_KEY", "")
    else:
        return os.environ.get("HF_TOKEN", "")


# Mapping: internal name -> model config
EXPERT_MODELS: Dict[str, Dict] = {
    "coding": {
        "model_id": "ornith:9b",
        "description": "Agentic coding, programming, software engineering (Ornith 9B)",
    },
    "multimodal": {
        "model_id": "qwen3.5:9b",
        "description": "Creative/long-form text, vision, RAG, brainstorming (Qwen3.5-9B)",
    },
    "general": {
        "model_id": "lfm2.5:latest",
        "description": "General-purpose writing, QA, summarization, default fallback",
    },
}

DEFAULT_EXPERT = "general"

# ---------------------------------------------------------------------------
# Dictatorship judge model
# ---------------------------------------------------------------------------
DICTATOR_JUDGE_MODEL = "granite4.1-guardian:8b-q4_K_S"
DICTATOR_JUDGE_DESC = "Safety-grounded evaluator for response selection"

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

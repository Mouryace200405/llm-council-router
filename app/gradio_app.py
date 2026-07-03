#!/usr/bin/env python3
"""
LLM Council Router — Gradio Research Dashboard
-----------------------------------------------
Provides full pipeline visibility:
  - Stage-by-stage view (raw prompt → enhanced → classified → routed → inferred)
  - Side-by-side comparison of Council, Majority Vote, and Dictatorship modes
  - Real-time metrics: latency, energy, quality scores, hallucination estimates
  - Confusion matrix and statistical comparison tables

Usage:
    python app/gradio_app.py
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None

from config.pipeline_config import EXPERT_MODELS
from src.orchestrator.comparison import ALL_EXPERTS, ComparisonRunner
from src.utils.research_metrics import (
    ResearchEvaluator,
    hallucination_heuristic,
    response_quality_score,
    vocabulary_richness,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("gradio_app")

RESEARCH_PROMPTS = [
    "Write a Python function to implement binary search",
    "Debug this code: def foo(x): return x / 0",
    "Write a SQL query to join users and orders tables",
    "Create a REST API endpoint in FastAPI with authentication",
    "Explain microservices architecture pros and cons",
    "Plan a 3-day itinerary for Paris",
    "Compare machine learning and deep learning",
    "What is the capital of France?",
    "Write a poem about the sea",
    "Summarize the plot of Hamlet",
    "Tell me a programming joke",
    "Explain why the sky is blue to a 5-year-old",
]


def build_pipeline_visual(
    prompt: str,
    enriched: str,
    cls,
    decision,
    response_text: str,
    expert_key: str,
) -> str:
    """Build a visual trace of the full pipeline."""
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE STAGE-BY-STAGE TRACE")
    lines.append("=" * 60)

    lines.append("\n[1] RAW PROMPT")
    lines.append(f"  {prompt}")

    lines.append(f"\n[2] ENHANCED PROMPT (via LFM2.5-Prompt-Enhancer)")
    lines.append(f"  {enriched}")

    lines.append(f"\n[3] CLASSIFICATION (via NVIDIA Prompt Classifier)")
    lines.append(f"  Task Type (Primary):    {cls.task_type_1}")
    lines.append(f"  Task Type (Secondary):  {cls.task_type_2}")
    lines.append(f"  Confidence:             {cls.task_type_prob:.3f}")
    lines.append(f"  Complexity Score:       {cls.prompt_complexity_score:.3f}")
    lines.append(f"  Reasoning:              {cls.reasoning:.3f}")
    lines.append(f"  Creativity:             {cls.creativity_scope:.3f}")
    lines.append(f"  Domain Knowledge:       {cls.domain_knowledge:.3f}")
    lines.append(f"  Contextual Knowledge:   {cls.contextual_knowledge:.3f}")
    lines.append(f"  Constraints:            {cls.constraint_ct:.3f}")
    lines.append(f"  Few-Shot Count:         {cls.number_of_few_shots:.3f}")

    lines.append(f"\n[4] ROUTING (via Lightweight Router)")
    for expert in ["coding", "multimodal", "general"]:
        score = decision.scores.get(expert, 0)
        label = f"  {expert:<12s} score={score:.3f}"
        if expert == decision.selected_expert:
            label += "  <-- SELECTED"
        lines.append(label)
    for reason in decision.reasons:
        lines.append(f"  Reason: {reason}")

    lines.append(f"\n[5] INFERENCE (via {expert_key} → {EXPERT_MODELS[expert_key]['model_id']})")
    lines.append(f"  {response_text[:600]}")

    return "\n".join(lines)


def build_comparison_table(cr) -> str:
    """Build side-by-side comparison of the three modes."""
    rows = []
    rows.append(f"{'Metric':<30} {'Council':<25} {'Majority Vote':<25} {'Dictatorship':<25}")
    rows.append("-" * 105)
    rows.append(f"{'Selected Expert':<30} {cr.council_expert or 'N/A':<25} "
                f"{cr.majority_vote_result or 'N/A':<25} {cr.dictator_chosen_expert or 'N/A':<25}")
    rows.append(f"{'Latency (ms)':<30} {cr.council_latency:<25.1f} "
                f"{cr.voting_latency:<25.1f} {cr.dictatorship_latency:<25.1f}")
    if cr.classification:
        rows.append(f"{'Task Type':<30} {cr.classification.task_type_1:<25} - -")
        rows.append(f"{'Complexity':<30} {cr.classification.prompt_complexity_score:<25.3f} - -")

    rows.append(f"\n{'Response Quality':<30} {'Council':<25} {'Majority':<25} {'Dictator':<25}")
    rows.append("-" * 105)

    def _resp_text(obj):
        if hasattr(obj, "text"):
            return obj.text
        return str(obj) if obj else ""

    texts = {"Council": _resp_text(cr.council_response),
             "Majority": _resp_text(cr.all_responses.get(cr.majority_vote_result or "")),
             "Dictator": _resp_text(cr.dictator_response)}

    for name in ["Council", "Majority", "Dictator"]:
        t = texts.get(name, "")
        qual = response_quality_score(t, cr.prompt)
        hall = hallucination_heuristic(t, cr.prompt)
        vocab = vocabulary_richness(t)
    rows.append(f"{'Quality Score':<30} {response_quality_score(texts['Council'], cr.prompt):<25.3f} "
                f"{response_quality_score(texts['Majority'], cr.prompt):<25.3f} "
                f"{response_quality_score(texts['Dictator'], cr.prompt):<25.3f}")
    rows.append(f"{'Hallucination Risk':<30} {hallucination_heuristic(texts['Council'], cr.prompt):<25.3f} "
                f"{hallucination_heuristic(texts['Majority'], cr.prompt):<25.3f} "
                f"{hallucination_heuristic(texts['Dictator'], cr.prompt):<25.3f}")
    rows.append(f"{'Vocab Richness':<30} {vocabulary_richness(texts['Council']):<25.3f} "
                f"{vocabulary_richness(texts['Majority']):<25.3f} "
                f"{vocabulary_richness(texts['Dictator']):<25.3f}")

    return "\n".join(rows)


def format_all_responses(cr) -> str:
    """Show what each model said."""
    lines = []
    for expert in ALL_EXPERTS:
        resp = cr.all_responses.get(expert)
        if resp:
            lines.append(f"=== {expert.upper()} ({resp.model_name}) ===")
            lines.append(resp.text[:400])
            lines.append("")
    if cr.council_response:
        lines.append(f"=== COUNCIL SELECTED ({cr.council_expert}) ===")
        lines.append(cr.council_response.text[:400])
        lines.append("")
    return "\n".join(lines)


# ===================================================================
# Global state
# ===================================================================
runner = None
history_metrics = {"council": [], "majority": [], "dictator": []}


def ensure_runner():
    global runner
    if runner is None:
        runner = ComparisonRunner(use_dummy=True)
    return runner


# ===================================================================
# Gradio handlers
# ===================================================================

def process_prompt(prompt: str) -> Tuple[str, str, str, str, str]:
    """Handle single-prompt processing."""
    r = ensure_runner()
    cr = r.run_comparison(prompt)

    trace = build_pipeline_visual(
        cr.prompt, cr.enriched,
        cr.classification, cr.routing,
        cr.council_response.text if cr.council_response else "",
        cr.council_expert or "N/A",
    )
    comparison = build_comparison_table(cr)
    responses = format_all_responses(cr)

    # Metrics summary
    metrics_lines = []
    metrics_lines.append(f"Total Energy: {cr.energy.total_energy_j:.4f} J" if cr.energy else "Energy: N/A (dummy mode)")
    metrics_lines.append(f"Council Latency: {cr.council_latency:.1f} ms")
    metrics_lines.append(f"Voting Latency:  {cr.voting_latency:.1f} ms")
    metrics_lines.append(f"Dictator Latency: {cr.dictatorship_latency:.1f} ms")
    metrics_str = "\n".join(metrics_lines)

    return trace, comparison, responses, metrics_str, cr.enriched


def run_benchmark() -> str:
    """Run benchmark over all research prompts and compute aggregate metrics."""
    from src.utils.research_metrics import ResearchEvaluator, ModeMetrics

    r = ensure_runner()
    evaluator = ResearchEvaluator()
    all_records = []

    for prompt in RESEARCH_PROMPTS:
        cr = r.run_comparison(prompt)
        all_records.append(cr)

    # Build mode-specific prediction lists
    council = [(cr.council_expert or "general", cr.council_latency, cr.council_response.text or "")
               for cr in all_records]
    majority = [(cr.majority_vote_result or "general", cr.voting_latency,
                 cr.all_responses.get(cr.majority_vote_result or "", None))
                for cr in all_records]
    majority = [(e, l, (t.text if hasattr(t, 'text') else str(t) if t else "")) for e, l, t in majority]
    dictator = [(cr.dictator_chosen_expert or "general", cr.dictatorship_latency, cr.dictator_response.text or "")
                for cr in all_records]

    cm = evaluator.compute_mode_metrics(council)
    cm.name = "Council"
    mm = evaluator.compute_mode_metrics(majority)
    mm.name = "Majority Vote"
    dm = evaluator.compute_mode_metrics(dictator)
    dm.name = "Dictatorship"

    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK RESULTS (over {} prompts)".format(len(RESEARCH_PROMPTS)))
    lines.append("=" * 70)
    for m in [cm, mm, dm]:
        lines.append(f"\n--- {m.name} ---")
        lines.append(f"  Avg Latency:    {m.avg_latency_ms:.1f} ms")
        lines.append(f"  P95 Latency:    {m.p95_latency_ms:.1f} ms")
        lines.append(f"  Response Quality: {m.avg_quality:.3f}")
        lines.append(f"  Hallucination:  {m.avg_hallucination:.3f}")
        lines.append(f"  Vocab Richness: {m.avg_vocab_richness:.3f}")
        lines.append(f"  Model Usage:    {m.model_utilization}")

    lines.append(f"\n--- LaTeX Table ---")
    lines.append(evaluator.format_latex_table([cm, mm, dm]))

    return "\n".join(lines)


def live_update(prompt: str) -> Dict:
    """Return metrics as JSON for the live dashboard."""
    r = ensure_runner()
    cr = r.run_comparison(prompt)

    return {
        "council_expert": cr.council_expert,
        "majority_expert": cr.majority_vote_result,
        "dictator_expert": cr.dictator_chosen_expert,
        "council_latency": round(cr.council_latency, 1),
        "voting_latency": round(cr.voting_latency, 1),
        "dictator_latency": round(cr.dictatorship_latency, 1),
        "complexity": round(cr.classification.prompt_complexity_score, 3) if cr.classification else 0,
        "task_type": cr.classification.task_type_1 if cr.classification else "N/A",
    }


# ===================================================================
# Build Gradio UI
# ===================================================================


CSS = """
.pipeline-trace { font-family: 'Courier New', monospace; font-size: 13px; }
.metric-card { background: #f0f4ff; border-radius: 8px; padding: 12px; }
"""

def create_ui():
    with gr.Blocks(
        title="LLM Council Router — Research Dashboard",
    ) as demo:
        gr.Markdown(
            """
            # 🤖 LLM Council Router — Research Dashboard
            ### Intelligent Prompt Routing with 3-Expert LLM Council

            This dashboard shows **every stage** of the routing pipeline and compares
            **three routing paradigms** for your research paper:
            - **Council Mode** (intelligent enhance → classify → route → infer)
            - **Majority Voting** (all 3 models respond, majority decides)
            - **Dictatorship** (all 3 respond, judge picks best)

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="e.g. Write a Python function for binary search...",
                    lines=3,
                )
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["Council", "Majority Voting", "Dictatorship", "All Three"],
                    label="Routing Mode",
                    value="All Three",
                )

        with gr.Row():
            submit_btn = gr.Button("🚀 Run Pipeline", variant="primary", scale=2)
            benchmark_btn = gr.Button("📊 Run Benchmark (12 prompts)", scale=2)

        # --- Stage-by-stage pipeline trace ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔍 Pipeline Stage-by-Stage Trace")
                pipeline_trace = gr.Textbox(
                    label="",
                    lines=30,
                    max_lines=40,
                    elem_classes=["pipeline-trace"],
                )

        # --- Live metrics cards ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("## ⚡ Real-Time Metrics")
            with gr.Column():
                enhanced_output = gr.Textbox(label="Enhanced Prompt", lines=1)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["metric-card"]):
                task_type_display = gr.Textbox(label="Task Type", value="—")
                complexity_display = gr.Textbox(label="Complexity", value="—")
            with gr.Column(scale=1, elem_classes=["metric-card"]):
                council_expert_display = gr.Textbox(label="Council → Expert", value="—")
                council_latency_display = gr.Textbox(label="Council Latency", value="— ms")
            with gr.Column(scale=1, elem_classes=["metric-card"]):
                majority_expert_display = gr.Textbox(label="Majority → Winner", value="—")
                voting_latency_display = gr.Textbox(label="Voting Latency", value="— ms")
            with gr.Column(scale=1, elem_classes=["metric-card"]):
                dictator_expert_display = gr.Textbox(label="Dictator → Chosen", value="—")
                dictator_latency_display = gr.Textbox(label="Dictator Latency", value="— ms")

        # --- Side-by-side comparison ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Mode Comparison")
                comparison_output = gr.Textbox(
                    label="Council vs Majority vs Dictatorship",
                    lines=18,
                    max_lines=25,
                )

        # --- All model responses ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💬 All Expert Responses")
                responses_output = gr.Textbox(
                    label="What each model said",
                    lines=20,
                    max_lines=30,
                )

        # --- Benchmark output ---
        with gr.Row():
            benchmark_output = gr.Textbox(
                label="Benchmark Results",
                lines=20,
                max_lines=30,
                visible=True,
            )

        # ================================================================
        # Event handlers
        # ================================================================

        def on_submit(prompt_text, mode_choice):
            if not prompt_text.strip():
                return (
                    "Please enter a prompt.", "", "", "", "—", "—",
                    "—", "— ms", "—", "— ms", "—", "— ms",
                )

            r = ensure_runner()
            cr = r.run_comparison(prompt_text)

            trace = build_pipeline_visual(
                cr.prompt, cr.enriched,
                cr.classification, cr.routing,
                cr.council_response.text if cr.council_response else "",
                cr.council_expert or "—",
            )

            comparison = build_comparison_table(cr) if mode_choice == "All Three" else "Select 'All Three' to see comparison"
            responses = format_all_responses(cr) if mode_choice == "All Three" else "Select 'All Three' to see all responses"

            task_type = cr.classification.task_type_1 if cr.classification else "—"
            complexity = f"{cr.classification.prompt_complexity_score:.3f}" if cr.classification else "—"

            return (
                trace, comparison, responses, cr.enriched,
                task_type, complexity,
                cr.council_expert or "—", f"{cr.council_latency:.1f} ms",
                cr.majority_vote_result or "—", f"{cr.voting_latency:.1f} ms",
                cr.dictator_chosen_expert or "—", f"{cr.dictatorship_latency:.1f} ms",
            )

        def on_benchmark():
            result = run_benchmark()
            return result

        submit_event = submit_btn.click(
            fn=on_submit,
            inputs=[prompt_input, mode],
            outputs=[
                pipeline_trace,
                comparison_output,
                responses_output,
                enhanced_output,
                task_type_display,
                complexity_display,
                council_expert_display,
                council_latency_display,
                majority_expert_display,
                voting_latency_display,
                dictator_expert_display,
                dictator_latency_display,
            ],
        )

        benchmark_btn.click(
            fn=on_benchmark,
            inputs=[],
            outputs=[benchmark_output],
        )

        gr.Markdown(
            """
            ---
            **Architecture:**
            - 🧠 *Prompt Enhancer*: LFM2.5 (local) rewrites raw prompts
            - 📐 *Classifier*: NVIDIA DeBERTa-v3 (local) scores 10 dimensions
            - 🔀 *Router*: Lightweight heuristic scorer (no model loaded)
            - ⚡ *Experts*: Qwen2.5-Coder-7B | Qwen3.5-9B | Llama-3.1-8B (HF cloud)

            **Built for IEEE-style research evaluation.**
            """
        )

    return demo


# ===================================================================
# Main
# ===================================================================

def main():
    if gr is None:
        print("ERROR: gradio not installed. Run: pip install gradio")
        sys.exit(1)

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()

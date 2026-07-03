#!/usr/bin/env python3
"""
LLM Council Router — Gradio Research Dashboard
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


def build_pipeline_visual(prompt, enriched, cls, decision, response_text, expert_key):
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE STAGE-BY-STAGE TRACE")
    lines.append("=" * 60)
    lines.append(f"\n[1] RAW PROMPT\n  {prompt}")
    lines.append(f"\n[2] ENHANCED PROMPT\n  {enriched}")
    lines.append(f"\n[3] CLASSIFICATION (NVIDIA Prompt Classifier)")
    if cls:
        lines.append(f"  Task Type:         {cls.task_type_1} / {cls.task_type_2}")
        lines.append(f"  Confidence:        {cls.task_type_prob:.3f}")
        lines.append(f"  Complexity:        {cls.prompt_complexity_score:.3f}")
        lines.append(f"  Reasoning:         {cls.reasoning:.3f}")
        lines.append(f"  Creativity:        {cls.creativity_scope:.3f}")
        lines.append(f"  Domain Knowledge:  {cls.domain_knowledge:.3f}")
        lines.append(f"  Contextual Know:   {cls.contextual_knowledge:.3f}")
        lines.append(f"  Constraints:       {cls.constraint_ct:.3f}")
        lines.append(f"  Few-Shot Count:    {cls.number_of_few_shots:.3f}")
    lines.append(f"\n[4] ROUTING (Lightweight Router)")
    if decision:
        for expert in ["coding", "multimodal", "general"]:
            score = decision.scores.get(expert, 0)
            label = f"  {expert:<12s} score={score:.3f}"
            if expert == decision.selected_expert:
                label += "  <-- SELECTED"
            lines.append(label)
        for reason in decision.reasons:
            lines.append(f"  Reason: {reason}")
    lines.append(f"\n[5] INFERENCE")
    lines.append(f"  Model: {expert_key} -> {EXPERT_MODELS.get(expert_key, {}).get('model_id', 'N/A')}")
    lines.append(f"  {response_text[:600]}")
    return "\n".join(lines)


def build_comparison_table(cr):
    if not cr:
        return "No comparison data available."

    rows = []
    rows.append(f"{'Metric':<30} {'Council':<30} {'Majority Vote':<30} {'Dictatorship':<30}")
    rows.append("-" * 120)
    rows.append(f"{'Selected Expert':<30} {cr.council_expert or 'N/A':<30} "
                f"{cr.majority_vote_result or 'N/A':<30} {cr.dictator_chosen_expert or 'N/A':<30}")
    rows.append(f"{'Latency (ms)':<30} {cr.council_latency:<30.1f} "
                f"{cr.voting_latency:<30.1f} {cr.dictatorship_latency:<30.1f}")
    if cr.classification:
        rows.append(f"{'Task Type':<30} {cr.classification.task_type_1:<30} - -")
        rows.append(f"{'Complexity':<30} {cr.classification.prompt_complexity_score:<30.3f} - -")

    def _resp_text(obj):
        if hasattr(obj, "text"):
            return obj.text
        return str(obj) if obj else ""

    council_t = _resp_text(cr.council_response)
    majority_key = cr.majority_vote_result or ""
    majority_t = _resp_text(cr.all_responses.get(majority_key, ""))
    dictator_key = cr.dictator_chosen_expert or ""
    dictator_t = _resp_text(cr.all_responses.get(dictator_key, ""))

    rows.append(f"\n{'Response Quality':<30} {'Council':<30} {'Majority':<30} {'Dictator':<30}")
    rows.append("-" * 120)
    rows.append(f"{'Quality Score':<30} {response_quality_score(council_t, cr.prompt):<30.3f} "
                f"{response_quality_score(majority_t, cr.prompt):<30.3f} "
                f"{response_quality_score(dictator_t, cr.prompt):<30.3f}")
    rows.append(f"{'Hallucination Risk':<30} {hallucination_heuristic(council_t, cr.prompt):<30.3f} "
                f"{hallucination_heuristic(majority_t, cr.prompt):<30.3f} "
                f"{hallucination_heuristic(dictator_t, cr.prompt):<30.3f}")
    rows.append(f"{'Vocab Richness':<30} {vocabulary_richness(council_t):<30.3f} "
                f"{vocabulary_richness(majority_t):<30.3f} "
                f"{vocabulary_richness(dictator_t):<30.3f}")

    return "\n".join(rows)


def format_all_responses(cr):
    if not cr:
        return "No response data available."

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
    if cr.dictator_chosen_expert:
        chosen_resp = cr.all_responses.get(cr.dictator_chosen_expert, None)
        if chosen_resp:
            lines.append(f"=== DICTATOR CHOSEN ({cr.dictator_chosen_expert}) ===")
            lines.append(chosen_resp.text[:400])
            lines.append("")
        else:
            lines.append(f"=== DICTATOR CHOSEN ({cr.dictator_chosen_expert}) ===")
            lines.append("(response not available)")
            lines.append("")
    return "\n".join(lines)


# ===================================================================
# Global state
# ===================================================================
runner = None


def ensure_runner():
    global runner
    if runner is None:
        use_live = "--live" in sys.argv
        mode = "LIVE" if use_live else "dummy"
        logger.info("Initializing ComparisonRunner (%s mode)...", mode)
        runner = ComparisonRunner(use_dummy=not use_live)
        logger.info("ComparisonRunner ready.")
    return runner


# ===================================================================
# Gradio handlers
# ===================================================================

def on_submit(prompt_text, mode_choice):
    if not prompt_text.strip():
        return ("Please enter a prompt.", "", "", "", "—", "—",
                "—", "— ms", "—", "— ms", "—", "— ms")

    try:
        mode_map = {
            "Council": "council",
            "Majority Voting": "voting",
            "Dictatorship": "dictator",
            "All Three": "all",
        }
        r = ensure_runner()
        cr = r.run_comparison(prompt_text, mode=mode_map.get(mode_choice, "all"))

        trace = build_pipeline_visual(
            cr.prompt, cr.enriched,
            cr.classification, cr.routing,
            cr.council_response.text if cr.council_response else "",
            cr.council_expert or "—",
        )

        show_full = mode_choice == "All Three"
        comparison = build_comparison_table(cr) if show_full else "Select 'All Three' to see comparison"
        responses = format_all_responses(cr) if show_full else "Select 'All Three' to see all responses"

        task_type = cr.classification.task_type_1 if cr.classification else "—"
        complexity = f"{cr.classification.prompt_complexity_score:.3f}" if cr.classification else "—"

        council_exp = cr.council_expert or "—"
        council_lat = f"{cr.council_latency:.1f} ms" if cr.council_latency > 0 else "—"
        maj_exp = cr.majority_vote_result or "—"
        maj_lat = f"{cr.voting_latency:.1f} ms" if cr.voting_latency > 0 else "—"
        dict_exp = cr.dictator_chosen_expert or "—"
        dict_lat = f"{cr.dictatorship_latency:.1f} ms" if cr.dictatorship_latency > 0 else "—"

        return (
            trace, comparison, responses, cr.enriched,
            task_type, complexity,
            council_exp, council_lat,
            maj_exp, maj_lat,
            dict_exp, dict_lat,
        )
    except Exception as e:
        logger.error("Error processing prompt: %s", e)
        return (f"Error: {e}", "", "", "", "—", "—",
                "—", "— ms", "—", "— ms", "—", "— ms")


def run_benchmark():
    from src.utils.research_metrics import ResearchEvaluator

    r = ensure_runner()
    evaluator = ResearchEvaluator()
    all_records = []

    for i, prompt in enumerate(RESEARCH_PROMPTS):
        try:
            cr = r.run_comparison(prompt, mode="all")
            all_records.append(cr)
        except Exception as e:
            logger.error("Benchmark failed at prompt %d: %s", i, e)

    if not all_records:
        return "Benchmark failed — no records collected."

    council = [(cr.council_expert or "general", cr.council_latency,
                cr.council_response.text or "") for cr in all_records]
    majority = [(cr.majority_vote_result or "general", cr.voting_latency,
                 cr.all_responses.get(cr.majority_vote_result or "", ""))
                for cr in all_records]
    majority = [(e, l, (t.text if hasattr(t, 'text') else str(t) if t else "")) for e, l, t in majority]
    dictator = [(cr.dictator_chosen_expert or "general", cr.dictatorship_latency,
                 cr.all_responses.get(cr.dictator_chosen_expert or "", "").text
                 if isinstance(cr.all_responses.get(cr.dictator_chosen_expert or ""), type(cr.council_response))
                 else str(cr.all_responses.get(cr.dictator_chosen_expert or "", "")))
                for cr in all_records]

    cm = evaluator.compute_mode_metrics(council)
    cm.name = "Council"
    mm = evaluator.compute_mode_metrics(majority)
    mm.name = "Majority Vote"
    dm = evaluator.compute_mode_metrics(dictator)
    dm.name = "Dictatorship"

    lines = []
    lines.append("=" * 70)
    lines.append(f"BENCHMARK RESULTS ({len(all_records)} prompts)")
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


# ===================================================================
# Build Gradio UI
# ===================================================================

def create_ui():
    with gr.Blocks(title="LLM Council Router — Research Dashboard") as demo:
        gr.Markdown(
            """
            # LLM Council Router — Research Dashboard
            ### Intelligent Prompt Routing with 3-Expert LLM Council

            This dashboard shows **every stage** of the routing pipeline and compares
            **three routing paradigms**:
            - **Council Mode** (intelligent enhance → classify → route → infer)
            - **Majority Voting** (all 3 models respond, majority decides)
            - **Dictatorship** (all 3 respond, judge picks best)
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
            submit_btn = gr.Button("Run Pipeline", variant="primary", scale=2)
            benchmark_btn = gr.Button("Run Benchmark (12 prompts)", scale=2)

        # Pipeline trace
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Pipeline Stage-by-Stage Trace")
                pipeline_trace = gr.Textbox(
                    label="",
                    lines=25,
                    max_lines=40,
                )

        # Live metrics
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Real-Time Metrics")
            with gr.Column():
                enhanced_output = gr.Textbox(label="Enhanced Prompt", lines=1)

        with gr.Row():
            with gr.Column(scale=1):
                task_type_display = gr.Textbox(label="Task Type", value="—")
                complexity_display = gr.Textbox(label="Complexity", value="—")
            with gr.Column(scale=1):
                council_expert_display = gr.Textbox(label="Council Expert", value="—")
                council_latency_display = gr.Textbox(label="Council Latency", value="— ms")
            with gr.Column(scale=1):
                majority_expert_display = gr.Textbox(label="Majority Winner", value="—")
                voting_latency_display = gr.Textbox(label="Voting Latency", value="— ms")
            with gr.Column(scale=1):
                dictator_expert_display = gr.Textbox(label="Dictator Chosen", value="—")
                dictator_latency_display = gr.Textbox(label="Dictator Latency", value="— ms")

        # Side-by-side comparison
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Mode Comparison")
                comparison_output = gr.Textbox(
                    label="Council vs Majority vs Dictatorship",
                    lines=15,
                    max_lines=25,
                )

        # All model responses
        with gr.Row():
            with gr.Column():
                gr.Markdown("## All Expert Responses")
                responses_output = gr.Textbox(
                    label="What each model said",
                    lines=18,
                    max_lines=30,
                )

        # Benchmark output
        with gr.Row():
            benchmark_output = gr.Textbox(
                label="Benchmark Results",
                lines=18,
                max_lines=30,
            )

        # ================================================================
        # Event handlers
        # ================================================================

        submit_btn.click(
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
            fn=run_benchmark,
            inputs=[],
            outputs=[benchmark_output],
        )

        mode_status = "LIVE (HF Inference API)" if "--live" in sys.argv else "DUMMY (simulated)"
        gr.Markdown(
            f"""
            ---
            **Architecture:**
            - Prompt Enhancer: LFM2.5 (local) rewrites raw prompts
            - Classifier: NVIDIA DeBERTa-v3 (local) scores 10 dimensions
            - Router: Pure classifier-based scorer (no model loaded)
            - Experts: Qwen2.5-Coder-7B | Qwen3.5-9B | Llama-3.1-8B

            **Status:** {mode_status}
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

    global runner
    use_live = "--live" in sys.argv

    if use_live:
        print("LIVE mode — using HF Inference API for model responses")
        runner = ComparisonRunner(use_dummy=False)
    else:
        print("Dummy mode — simulated responses (use --live for real models)")
        runner = ComparisonRunner(use_dummy=True)

    print("Models ready. Launching dashboard...")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

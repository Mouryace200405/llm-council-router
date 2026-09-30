#!/usr/bin/env python3
"""
LLM Council Router — Clean Gradio Interface
Response-first design: see the model answer immediately.
"""

import base64
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gradio as gr
except ImportError:
    gr = None

from config.pipeline_config import EXPERT_MODELS, ALL_ROUTES
from src.orchestrator.comparison import ALL_EXPERTS, ComparisonRunner

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger("gradio_app")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".json", ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".html",
                   ".md", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".rs", ".go",
                   ".java", ".c", ".cpp", ".h", ".hpp", ".csv", ".xml", ".sql", ".rb",
                   ".php", ".r", ".swift", ".kt"}


def is_image_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSIONS


def read_image_b64(file_path: str, max_dim: int = 1024) -> Optional[str]:
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(file_path)
        w, h = img.size
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error("Failed to read image %s: %s", file_path, e)
        return None


def get_file_label(file_path: str) -> str:
    name = os.path.basename(file_path)
    if is_image_file(file_path):
        return f"[Image: {name}]"
    ext = os.path.splitext(file_path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return f"[Code/Text: {name}]"
    return f"[File: {name}]"


def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[Error: {e}]"


runner = None


def ensure_runner():
    global runner
    if runner is None:
        logger.info("Initializing ComparisonRunner (Ollama)...")
        runner = ComparisonRunner(use_dummy=False)
    return runner


def on_submit(prompt_text, mode_choice, uploaded_files):
    if not prompt_text.strip():
        return "Please enter a prompt.", "", "", "", "", "—", "—", "", ""

    try:
        image_b64_list: List[str] = []
        filenames: List[str] = []
        file_context = ""
        if uploaded_files is not None and len(uploaded_files) > 0:
            file_paths = [f.name if hasattr(f, 'name') else str(f) for f in uploaded_files]
            labels = []
            for fp in file_paths:
                filenames.append(os.path.basename(fp))
                if is_image_file(fp):
                    b64 = read_image_b64(fp)
                    if b64:
                        image_b64_list.append(b64)
                    labels.append(get_file_label(fp))
                else:
                    content = read_text_file(fp)
                    labels.append(f"=== {os.path.basename(fp)} ===\n{content}")
            file_context = "\n\n".join(labels)

        mode_map = {
            "Council": "council",
            "Majority Voting": "voting",
            "Dictatorship": "dictator",
            "All Three": "all",
        }
        r = ensure_runner()
        full_prompt = prompt_text + ("\n\n[Attached files]\n" + file_context if file_context else "")
        image_kw = {"images": image_b64_list} if image_b64_list else {}
        # When actual image data is present, force route to multimodal
        # (only the multimodal expert qwen3.5:9b supports vision)
        force_route = "multimodal" if image_b64_list else None
        cr = r.run_comparison(full_prompt, mode=mode_map.get(mode_choice, "all"),
                              filenames=filenames if filenames else None,
                              force_route=force_route,
                              **image_kw)

        response_text = cr.council_response.text if cr.council_response else ""
        route = cr.council_route or "—"
        expert = cr.council_expert or "—"
        model_id = EXPERT_MODELS.get(cr.council_expert, {}).get("model_id", "—") if cr.council_expert else "—"
        latency = f"{cr.council_latency:.1f} ms" if cr.council_latency > 0 else "—"
        task_type = cr.classification.task_type_1 if cr.classification else "—"
        complexity = f"{cr.classification.prompt_complexity_score:.3f}" if cr.classification else "—"

        info_lines = [
            f"Routed to: {route} → {expert} ({model_id})",
            f"Latency: {latency}",
            f"Task: {task_type}  |  Complexity: {complexity}",
        ]
        info_text = "  |  ".join(info_lines)

        # Full classifier metrics
        metrics_str = ""
        if cr.classification:
            metrics_str = (
                f"task_type_1: {cr.classification.task_type_1}\n"
                f"task_type_2: {cr.classification.task_type_2}\n"
                f"task_type_conf: {cr.classification.task_type_prob:.3f}\n"
                f"creativity: {cr.classification.creativity_scope:.3f}\n"
                f"reasoning: {cr.classification.reasoning:.3f}\n"
                f"domain_knowledge: {cr.classification.domain_knowledge:.3f}\n"
                f"contextual_knowledge: {cr.classification.contextual_knowledge:.3f}\n"
                f"constraints: {cr.classification.constraint_ct:.3f}\n"
                f"few_shots: {cr.classification.number_of_few_shots:.3f}\n"
                f"overall_complexity: {cr.classification.prompt_complexity_score:.3f}"
            )

        # Full router raw output
        router_str = ""
        if cr.routing:
            router_str = (
                f"model: {cr.routing.router_model}\n"
                f"route: {cr.routing.selected_route}\n"
                f"raw_output: {cr.routing.raw_router_output}"
            )

        return response_text, info_text, route, expert, latency, task_type, complexity, metrics_str, router_str

    except Exception as e:
        logger.error("Error: %s", e)
        return f"Error: {e}", "", "", "", "", "—", "—", "", ""


def create_ui():
    with gr.Blocks(title="LLM Council Router", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # LLM Council Router v2
            ### Qwen3.5-2B hybrid router + 5-category dispatch

            Type any prompt, optionally attach files, and the router picks the right expert.
            """
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Your prompt",
                placeholder="e.g. Write a Python function for binary search...",
                lines=4,
                scale=4,
            )
            mode = gr.Radio(
                ["Council", "Majority Voting", "Dictatorship", "All Three"],
                label="Mode",
                value="Council",
                scale=1,
            )

        with gr.Row():
            file_input = gr.File(
                label="Attach files (images → multimodal, code/docs → content extraction)",
                file_count="multiple",
                type="filepath",
                scale=4,
            )
            submit_btn = gr.Button("Go", variant="primary", size="lg", scale=1, min_width=100)

        gr.Markdown("---")

        info_text = gr.Textbox(label="Routing Info", lines=1, max_lines=1)

        response_output = gr.Textbox(
            label="Response",
            lines=15,
            max_lines=40,
        )

        with gr.Accordion("Pipeline Details", open=False):
            with gr.Tab("Routing"):
                with gr.Row():
                    route_display = gr.Textbox(label="Route")
                    expert_display = gr.Textbox(label="Expert")
                    latency_display = gr.Textbox(label="Latency")
                with gr.Row():
                    task_type_display = gr.Textbox(label="Task Type")
                    complexity_display = gr.Textbox(label="Complexity")
            with gr.Tab("Classifier Metrics"):
                metrics_display = gr.Textbox(label="NVIDIA DeBERTa-v3 (all 8 dimensions)", lines=12)
            with gr.Tab("Router Output"):
                router_display = gr.Textbox(label="Qwen3.5-2B raw output", lines=8)

        submit_btn.click(
            fn=on_submit,
            inputs=[prompt_input, mode, file_input],
            outputs=[
                response_output,
                info_text,
                route_display,
                expert_display,
                latency_display,
                task_type_display,
                complexity_display,
                metrics_display,
                router_display,
            ],
        )

        gr.Markdown(
            """
            ---
            **Router:** Qwen3.5-2B (hybrid) → **coder** (Ornith 9B) | **rag** / **multimodal** / **reasoning** (Qwen3.5-9B) | **general** (LFM2.5)
            **Backend:** Ollama (local)
            """
        )

    return demo


def main():
    if gr is None:
        print("ERROR: gradio not installed. Run: pip install gradio")
        sys.exit(1)

    print("LLM Council Router — Ollama backend")
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

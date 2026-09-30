import gradio as gr
import ollama
import torch
import torch.nn as nn
import json
import os
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

# ── Config ───────────────────────────────────────────────────
CLASSIFIER_PATH = "/home/ivan/prompt-task-and-complexity-classifier"
BACKBONE_PATH = "/home/ivan/deberta-v3-base"
ROUTER_MODEL = "qwen3.5:4b"
VALID_ROUTES = ["coder", "multimodal", "reasoning", "rag", "general"]


# ── NVIDIA Classifier ─────────────────────────────────────────


class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
        )


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class NvidiaClassifierModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, target_sizes, task_type_map, weights_map, divisor_map, backbone_path
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_path)
        self.target_sizes = target_sizes
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes.values()
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2_strings = [
                [self.task_type_map[str(i)] for i in s]
                for s in top2_indices.detach().cpu().tolist()
            ]
            top2_prob_rounded = [
                [round(v, 3) for v in s] for s in top2_probs.detach().cpu().tolist()
            ]
            return (
                [s[0] for s in top2_strings],
                [s[1] for s in top2_strings],
                [s[0] for s in top2_prob_rounded],
            )
        else:
            import numpy as np

            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(
                np.array(preds.detach().cpu(), dtype=np.float32) * weights, axis=1
            )
            scores = [
                round(v, decimal) for v in weighted_sum / self.divisor_map[target]
            ]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits):
        result = {}
        t1, t2, tp = self.compute_results(logits[0], target="task_type")
        result["task_type_1"] = t1
        result["task_type_2"] = t2
        result["task_type_prob"] = tp
        for idx, target in enumerate(
            [
                "creativity_scope",
                "reasoning",
                "contextual_knowledge",
                "number_of_few_shots",
                "domain_knowledge",
                "no_label_reason",
                "constraint_ct",
            ],
            start=1,
        ):
            result[target] = self.compute_results(logits[idx], target=target)
        result["prompt_complexity_score"] = [
            round(
                0.35 * c + 0.25 * r + 0.15 * con + 0.15 * d + 0.05 * ctx + 0.05 * fs, 5
            )
            for c, r, con, d, ctx, fs in zip(
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
        out = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pooled = self.pool(out.last_hidden_state, batch["attention_mask"])
        logits = [self.heads[k](pooled) for k in range(len(self.target_sizes))]
        return self.process_logits(logits)


class LocalClassifier:
    def __init__(self):
        print("[classifier] loading ...")
        with open(os.path.join(CLASSIFIER_PATH, "config.json")) as f:
            cfg = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            CLASSIFIER_PATH, fix_mistral_regex=True
        )
        self.model = NvidiaClassifierModel(
            target_sizes=cfg["target_sizes"],
            task_type_map=cfg["task_type_map"],
            weights_map=cfg["weights_map"],
            divisor_map=cfg["divisor_map"],
            backbone_path=BACKBONE_PATH,
        )
        state_dict = load_file(os.path.join(CLASSIFIER_PATH, "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("[classifier] ready.\n")

    @torch.no_grad()
    def classify(self, prompt: str) -> dict:
        encoded = self.tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        r = self.model(encoded)
        return {
            "task_type_1": r["task_type_1"][0],
            "task_type_2": r["task_type_2"][0],
            "task_type_conf": r["task_type_prob"][0],
            "creativity": r["creativity_scope"][0],
            "reasoning": r["reasoning"][0],
            "contextual_knowledge": r["contextual_knowledge"][0],
            "domain_knowledge": r["domain_knowledge"][0],
            "constraints": r["constraint_ct"][0],
            "few_shots": r["number_of_few_shots"][0],
            "overall_complexity": r["prompt_complexity_score"][0],
        }


# ── Router (SIMPLIFIED - No Keywords, Just Understanding) ────

ROUTER_SYSTEM_PROMPT = """You are an intelligent routing engine that classifies user requests into one of five lanes.

Your job is to understand the INTENT of the request and route it appropriately.

═══════════════════════════════════════════════════════════════════
THE FIVE LANES:
═══════════════════════════════════════════════════════════════════

1. MULTIMODAL - For requests that involve VISUAL content:
   • Asking about images, photos, screenshots, diagrams, charts, graphs
   • Requests to describe, analyze, or extract information from visual media
   • ANY request with a visual file attached (.png, .jpg, .jpeg, .svg, .webp, .gif, .bmp)

2. REASONING - For requests that require MATHEMATICAL or LOGICAL PROOF:
   • Proving theorems, deriving equations, complexity analysis
   • Formal verification, mathematical derivations, algorithm analysis
   • Requests that ask "prove", "derive", "complexity", or involve formal math

3. CODER - For requests about WRITING or FIXING CODE:
   • Writing functions, implementing algorithms, debugging code
   • Refactoring, optimizing code, writing tests
   • Building APIs, creating scripts, database queries
   • ONLY when the user explicitly wants CODE to be written or modified

4. RAG - For requests that need to READ CONTENT from ATTACHED DOCUMENTS:
   • Summarizing documents, extracting information from files
   • Finding specific data inside text files, PDFs, spreadsheets
   • MUST have a text file attached (.pdf, .txt, .docx, .json, .csv, .xlsx, .md)
   • EXCLUDES metadata questions (title, page count, file size) - these go to GENERAL

5. GENERAL - The default lane for everything else:
   • Conceptual explanations ("what is X", "how does Y work")
   • Comparisons between technologies ("React vs Vue", "SQL vs NoSQL")
   • Creative writing (poems, stories, naming, brainstorming)
   • Advice, planning, translations, historical summaries
   • File metadata (title, page count, author, file size)
   • Questions without attached files that ask about general knowledge

═══════════════════════════════════════════════════════════════════
DECISION RULES:
═══════════════════════════════════════════════════════════════════

• MULTIMODAL trumps everything else - if there's a visual file or clear visual reference → MULTIMODAL
• REASONING trumps CODER - if asking to prove/derive AND write code → REASONING
• RAG requires a text file AND content extraction - metadata alone → GENERAL
• "Summarize" without a file → GENERAL (external knowledge)
• "Write code" + creative task (naming/brainstorming) → GENERAL
• When in doubt → GENERAL (it's the safe default)

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════

Output ONLY the category name in lowercase:
coder, multimodal, reasoning, rag, general

No explanations. No extra text. Just the category.
"""


def route(prompt: str, clf: dict, filename: str | None, max_retries: int = 3) -> str:
    """Route the prompt using LLM understanding with NVIDIA features as reference."""

    # Build the user message with context
    user_message = (
        "Here is the user request. Please route it to the appropriate lane.\n\n"
        f"User request: {prompt}\n"
    )

    if filename:
        user_message += f"Attached file: {filename}\n"

    user_message += (
        "\n"
        "For reference only (do not let these override your understanding):\n"
        f"  - A classifier estimated task type: {clf['task_type_1']} / {clf['task_type_2']}\n"
        f"  - Estimated reasoning level: {clf['reasoning']:.2f}\n"
        f"  - Estimated domain knowledge: {clf['domain_knowledge']:.2f}\n"
        f"  - Estimated creativity: {clf['creativity']:.2f}\n"
        f"  - Estimated complexity: {clf['overall_complexity']:.2f}\n"
        "\n"
        "Important: Use these numbers as a hint only. Your understanding of the "
        "user's intent is what matters most. If the numbers conflict with what "
        "the user is actually asking, trust the user's request.\n\n"
        "What is the correct routing for this request?"
    )

    for attempt in range(max_retries):
        response = ollama.chat(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0.1},  # Slight temperature for robustness
            think=False,
        )
        raw = response["message"]["content"].strip()
        if not raw:
            print(f"  [router] empty response attempt {attempt + 1}, retrying ...")
            continue

        # Extract just the category
        for token in raw.lower().split():
            word = token.strip(".,!?;:()'\"")
            if word in VALID_ROUTES:
                return word

        print(f"  [router] invalid '{raw[:50]}' attempt {attempt + 1}, retrying ...")

    return "general"


# ── Gradio app ───────────────────────────────────────────────

print("[app] loading classifier (this happens once at startup) ...")
clf_model = LocalClassifier()


def run_single_prompt(prompt: str, filename: str):
    if not prompt or not prompt.strip():
        return "Please enter a prompt.", ""

    filename = filename.strip() if filename and filename.strip() else None

    clf = clf_model.classify(prompt)
    pred = route(prompt, clf, filename)

    details = (
        f"task_type       : {clf['task_type_1']} / {clf['task_type_2']} "
        f"(conf: {clf['task_type_conf']:.3f})\n"
        f"reasoning       : {clf['reasoning']:.3f}\n"
        f"domain_knowledge: {clf['domain_knowledge']:.3f}\n"
        f"creativity      : {clf['creativity']:.3f}\n"
        f"contextual_knw  : {clf['contextual_knowledge']:.3f}\n"
        f"constraints     : {clf['constraints']:.3f}\n"
        f"few_shots       : {clf['few_shots']:.3f}\n"
        f"overall_complex : {clf['overall_complexity']:.3f}"
    )

    return pred, details


with gr.Blocks(title="LLM Council Router") as demo:
    gr.Markdown("## LLM Council — Prompt Router")
    gr.Markdown(
        "Type a prompt, optionally name an attached file (just the filename, "
        "used by the router as context), and see which lane it's routed to."
    )

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Prompt", placeholder="Type your prompt here...", lines=4
        )
    with gr.Row():
        filename_box = gr.Textbox(
            label="Attached filename (optional)",
            placeholder="e.g. screenshot.png, notes.pdf",
        )

    submit_btn = gr.Button("Route", variant="primary")

    with gr.Row():
        route_output = gr.Textbox(label="Predicted Route")
    with gr.Row():
        details_output = gr.Textbox(label="Classifier Details", lines=8)

    submit_btn.click(
        fn=run_single_prompt,
        inputs=[prompt_box, filename_box],
        outputs=[route_output, details_output],
    )
    prompt_box.submit(
        fn=run_single_prompt,
        inputs=[prompt_box, filename_box],
        outputs=[route_output, details_output],
    )

if __name__ == "__main__":
    demo.launch()

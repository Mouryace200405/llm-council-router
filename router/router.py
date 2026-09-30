# ============================================================
# LLM COUNCIL — Routing Benchmark Runner (FIXED)
# ============================================================
# Uses NVIDIA classifier features as guidance ONLY
# Router LLM makes final decision with better system prompt
# ============================================================

import json
import os
import time

import numpy as np
import ollama
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import AutoModel, AutoTokenizer

# ── Config ───────────────────────────────────────────────────
CLASSIFIER_PATH = "/home/ivan/prompt-task-and-complexity-classifier"
BACKBONE_PATH = "/home/ivan/deberta-v3-base"
ROUTER_MODEL = "qwen3.5:4b"
PROMPTS_FILE = "benchmark_prompts.json"
RESULTS_FILE = "benchmark_results.json"
REPORT_FILE = "benchmark_report.txt"
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


# ── Confusion matrix formatter ────────────────────────────────


def format_confusion_matrix(cm, labels):
    col_w = 13
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    lines = ["Predicted →", header, "-" * (col_w * (len(labels) + 1))]
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_w}}" + "".join(
            f"{cm[i][j]:>{col_w}}" for j in range(len(labels))
        )
        lines.append(row)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────


def run_benchmark():
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    clf_model = LocalClassifier()
    y_true, y_pred = [], []
    results, fails = [], []

    print(f"[benchmark] running {len(prompts)} prompts ...\n")
    start_total = time.time()

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        filename = item.get("filename")
        expected = item["expected_route"]

        t0 = time.time()
        clf = clf_model.classify(prompt)
        pred = route(prompt, clf, filename)
        elapsed = round(time.time() - t0, 2)

        correct = pred == expected
        y_true.append(expected)
        y_pred.append(pred)

        status = "✅" if correct else "❌"
        print(
            f"[{i + 1:03}/{len(prompts)}] {status}  "
            f"expected={expected:12}  got={pred:12}  "
            f"({elapsed}s)  {prompt[:50]}"
        )

        entry = {
            "id": i + 1,
            "prompt": prompt,
            "filename": filename,
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "latency_s": elapsed,
            "classifier": clf,
        }
        results.append(entry)
        if not correct:
            fails.append(entry)

    total_time = round(time.time() - start_total, 2)

    # ── Compute metrics ───────────────────────────────────────
    labels = VALID_ROUTES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_mic = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    f1_wtd = f1_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    prec_mac = precision_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    rec_mac = recall_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    per_cls = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    per_class_acc = {}
    for label in labels:
        idx = [j for j, t in enumerate(y_true) if t == label]
        if idx:
            per_class_acc[label] = round(
                sum(y_pred[j] == y_true[j] for j in idx) / len(idx), 4
            )

    # ── Build report ──────────────────────────────────────────
    sep = "─" * 62
    report = (
        f"""
╔══════════════════════════════════════════════════════════════╗
║         LLM COUNCIL — ROUTING BENCHMARK REPORT              ║
╚══════════════════════════════════════════════════════════════╝

Total prompts      : {len(prompts)}
Correct            : {int(accuracy * len(prompts))} / {len(prompts)}
Total runtime      : {total_time}s
Avg latency/prompt : {round(total_time / len(prompts), 2)}s

{sep}
OVERALL METRICS
{sep}
Accuracy               : {accuracy:.4f}
F1 Score (macro)       : {f1_mac:.4f}
F1 Score (micro)       : {f1_mic:.4f}
F1 Score (weighted)    : {f1_wtd:.4f}
Precision (macro)      : {prec_mac:.4f}
Recall    (macro)      : {rec_mac:.4f}
Cohen Kappa            : {kappa:.4f}
Matthews Corr Coef     : {mcc:.4f}

{sep}
PER-CLASS ACCURACY
{sep}
"""
        + "\n".join(f"  {k:14}: {v:.4f}" for k, v in per_class_acc.items())
        + f"""

{sep}
PER-CLASS PRECISION / RECALL / F1
{sep}
{per_cls}
{sep}
CONFUSION MATRIX  (Rows = Actual, Columns = Predicted)
{sep}
{format_confusion_matrix(cm, labels)}

{sep}
FAILED CASES  ({len(fails)} / {len(prompts)})
{sep}
"""
    )
    for fl in fails:
        c = fl["classifier"]
        report += (
            f"\n  [{fl['id']:03}]  expected={fl['expected']:12}  got={fl['predicted']}\n"
            f"         prompt    : {fl['prompt']}\n"
            f"         file      : {fl['filename']}\n"
            f"         task_type : {c['task_type_1']} (conf: {c['task_type_conf']:.3f})\n"
            f"         reasoning : {c['reasoning']:.3f}   "
            f"domain: {c['domain_knowledge']:.3f}   "
            f"overall: {c['overall_complexity']:.3f}\n"
        )

    print(report)

    # ── Save JSON ─────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": len(prompts),
                    "correct": int(accuracy * len(prompts)),
                    "accuracy": round(accuracy, 4),
                    "f1_macro": round(f1_mac, 4),
                    "f1_micro": round(f1_mic, 4),
                    "f1_weighted": round(f1_wtd, 4),
                    "precision_macro": round(prec_mac, 4),
                    "recall_macro": round(rec_mac, 4),
                    "cohen_kappa": round(kappa, 4),
                    "matthews_mcc": round(mcc, 4),
                    "per_class_accuracy": per_class_acc,
                    "total_runtime_s": total_time,
                    "avg_latency_s": round(total_time / len(prompts), 2),
                },
                "confusion_matrix": {
                    "labels": labels,
                    "matrix": cm.tolist(),
                },
                "results": results,
                "failures": fails,
            },
            f,
            indent=2,
        )

    # ── Save text report ──────────────────────────────────────
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(f"\n[done] results -> {RESULTS_FILE}")
    print(f"[done] report  -> {REPORT_FILE}")


if __name__ == "__main__":
    run_benchmark()

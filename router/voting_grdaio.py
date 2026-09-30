"""
Local Multi-Model Voting Benchmark — Gradio UI
================================================
Sequential-load version of the Kaggle multi-GPU voting harness, adapted for a
single laptop GPU (or CPU). Only ONE model is ever resident in memory at a
time: load -> generate -> confirm-unload -> load next.

Run:
    pip install gradio requests
    ollama serve                     # in another terminal
    python app.py

Then open the printed local URL.
"""

import json
import re
import time
import random
import subprocess
import requests
import gradio as gr

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "10s"      # how long Ollama keeps the model warm after a call
UNLOAD_POLL_TIMEOUT = 20        # seconds to wait for /api/ps to confirm eviction
RESULTS_PATH = "/tmp/voting_results.json"

JSON_VOTE_SCHEMA = {
    "type": "object",
    "properties": {"vote": {"type": "string", "enum": ["A", "B"]}},
    "required": ["vote"],
}

JUDGE_SYSTEM_MSG = (
    "You are an impartial judge. Your job is to pick the better of two AI responses. "
    "Do not explain your reasoning. Do not use markdown. "
    "Output ONLY a valid JSON object with a single key 'vote' whose value is either 'A' or 'B'."
)


# ── Ollama helpers ────────────────────────────────────────────────────────

def strip_thinking_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def list_local_models(host):
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def currently_loaded(host):
    """Returns the set of model names Ollama currently holds in memory (/api/ps)."""
    try:
        r = requests.get(f"{host}/api/ps", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


def wait_until_unloaded(host, model, log):
    """Polls /api/ps until `model` is no longer resident, so the next model gets
    the full VRAM budget instead of sharing it."""
    t0 = time.time()
    while time.time() - t0 < UNLOAD_POLL_TIMEOUT:
        loaded = currently_loaded(host)
        if model not in loaded:
            return True
        time.sleep(0.5)
    log(f"  ⚠️ {model} still reported loaded after {UNLOAD_POLL_TIMEOUT}s — continuing anyway")
    return False


def unload_model(host, model, log):
    """Explicitly evicts a model from memory right now (keep_alive=0)."""
    try:
        requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
    except Exception as e:
        log(f"  ⚠️ unload request for {model} failed: {e}")
        return
    wait_until_unloaded(host, model, log)


def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        out = {}
        for line in r.stdout.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                out[f"gpu{p[0]}"] = {
                    "memory_used_mb": int(p[1]),
                    "memory_total_mb": int(p[2]),
                    "utilization_pct": int(p[3]),
                }
        return out
    except Exception:
        return {}


def _error_result(msg):
    return {
        "text": f"[ERROR: {msg}]", "latency_s": 0, "prompt_tokens": 0,
        "response_tokens": 0, "tokens_per_sec": 0, "is_error": True,
    }


def call_model(host, model, prompt, keep_alive, max_tokens=1024, temperature=0.7,
               system=None, fmt=None, retries=2):
    for attempt in range(retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            if fmt is not None:
                payload["format"] = fmt
            t0 = time.time()
            r = requests.post(f"{host}/api/chat", json=payload, timeout=600)
            latency = round(time.time() - t0, 2)
            if r.status_code == 200:
                data = r.json()
                text = data.get("message", {}).get("content", "").strip()
                eval_count = data.get("eval_count", 0)
                return {
                    "text": text, "latency_s": latency,
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "response_tokens": eval_count,
                    "tokens_per_sec": round(eval_count / max(latency, 0.01), 1),
                    "is_error": False,
                }
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return _error_result(f"HTTP {r.status_code}: {r.text[:150]}")
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return _error_result("timeout")
        except Exception as e:
            return _error_result(str(e))
    return _error_result("max retries")


# ── One prompt, sequential across all models ─────────────────────────────

def get_responses_sequential(host, models, prompt, keep_alive, max_tokens, temperature, log):
    responses = {}
    for model in models:
        log(f"⏳ loading + generating: {model}")
        before = gpu_stats()
        r = call_model(host, model, prompt, keep_alive, max_tokens, temperature)
        after = gpu_stats()
        r["gpu_before"] = before
        r["gpu_after"] = after
        responses[model] = r
        status = "✅" if not r["is_error"] else "❌"
        log(f"  {status} {model} — {r['latency_s']}s, {r['tokens_per_sec']} tok/s"
            + ("" if not r["is_error"] else f" ({r['text']})"))
        log(f"⏏️  unloading {model} before next model loads")
        unload_model(host, model, log)
    return responses


def run_voting_sequential(host, models, prompt, responses, keep_alive, log):
    available = [m for m in models if not responses[m]["is_error"]]
    if len(available) < 3:
        log("  (skipping voting — need at least 3 non-errored models to judge pairwise)")
        return {}, (available[0] if available else None), {}

    votes = {}
    for voter in available:
        others = [m for m in available if m != voter]
        a_model, b_model = random.sample(others, 2)
        if random.random() < 0.5:
            a_model, b_model = b_model, a_model

        a_text = strip_thinking_blocks(responses[a_model]["text"])[:2000]
        b_text = strip_thinking_blocks(responses[b_model]["text"])[:2000]

        vote_prompt = (
            f'Prompt: "{prompt[:500]}"\n\n'
            "----------------------------------------\nResponse A:\n"
            f"{a_text}\n\n----------------------------------------\nResponse B:\n"
            f"{b_text}\n\n"
            "Choose the BETTER response based ONLY on: Accuracy, Completeness, Clarity, Helpfulness.\n"
            'Return: {"vote": "A"} or {"vote": "B"}'
        )

        log(f"⏳ reloading {voter} to judge {a_model} vs {b_model}")
        letter, raw = None, ""
        for attempt in range(3):
            result = call_model(
                host, voter, vote_prompt, keep_alive,
                max_tokens=512, temperature=0.0,
                system=JUDGE_SYSTEM_MSG, fmt=JSON_VOTE_SCHEMA,
            )
            raw = result["text"].strip()
            try:
                parsed = json.loads(raw)
                val = str(parsed.get("vote", "")).strip().upper()
                if val in ("A", "B"):
                    letter = val
                    break
            except json.JSONDecodeError:
                pass
            m = re.search(r'"vote"\s*:\s*["\']?([AB])["\']?', raw, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                break
            log(f"  ⚠️ {voter} gave an invalid vote (attempt {attempt+1}/3), retrying")
            time.sleep(0.5)

        if letter not in ("A", "B"):
            letter = "A" if random.random() < 0.5 else "B"
            log(f"  🚨 {voter} failed to vote validly — auto-assigned")

        voted_for = a_model if letter == "A" else b_model
        votes[voter] = {
            "voted_for": voted_for, "a_was": a_model, "b_was": b_model, "raw": raw,
        }
        log(f"  🗳 {voter} → {voted_for}")
        log(f"⏏️  unloading {voter}")
        unload_model(host, voter, log)

    tally = {}
    for v in votes.values():
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    winner = max(tally, key=tally.get) if tally else available[0]
    return votes, winner, tally


def process_one_prompt(host, models, prompt, keep_alive, max_tokens, temperature, log,
                        idx=1, total=1, filename=None):
    log(f"\n{'='*60}\n[{idx}/{total}] {prompt[:70]}\n{'='*60}")
    t0 = time.time()
    responses = get_responses_sequential(host, models, prompt, keep_alive, max_tokens, temperature, log)
    votes, winner, tally = run_voting_sequential(host, models, prompt, responses, keep_alive, log)
    total_time = round(time.time() - t0, 2)
    log(f"→ winner: {winner} | tally: {tally} | {total_time}s total")

    return {
        "id": idx,
        "prompt": prompt,
        "filename": filename,
        "responses": {
            m: {k: v for k, v in r.items() if k not in ("gpu_before", "gpu_after")}
            for m, r in responses.items()
        },
        "votes": votes,
        "vote_tally": tally,
        "winner": winner,
        "winning_response": responses.get(winner, {}).get("text", "") if winner else "",
        "total_time_s": total_time,
    }


# ── Gradio glue ───────────────────────────────────────────────────────────

def refresh_models(host):
    models = list_local_models(host)
    if not models:
        return gr.update(choices=[], value=[]), "⚠️ Couldn't reach Ollama at that host — is `ollama serve` running?"
    return gr.update(choices=models, value=models[: min(3, len(models))]), f"Found {len(models)} local models."


def format_result_markdown(result):
    lines = [f"### Prompt: {result['prompt'][:120]}"]
    if result.get("winner"):
        lines.append(f"**Winner: `{result['winner']}`**  — tally: {result['vote_tally']}\n")
    for model, r in result["responses"].items():
        tag = "❌ ERROR" if r["is_error"] else f"{r['latency_s']}s · {r['tokens_per_sec']} tok/s"
        lines.append(f"**{model}** ({tag})\n\n> {r['text'][:600]}\n")
    return "\n".join(lines)


def run_single(host, models, prompt, keep_alive, max_tokens, temperature):
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    if not models:
        yield "⚠️ Select at least one model.", "", None
        return
    if not prompt.strip():
        yield "⚠️ Enter a prompt.", "", None
        return

    yield "\n".join(log_lines), "Running…", None
    result = process_one_prompt(host, models, prompt, keep_alive, max_tokens, temperature, log)
    with open(RESULTS_PATH, "w") as f:
        json.dump([result], f, indent=2)
    yield "\n".join(log_lines), format_result_markdown(result), RESULTS_PATH


def run_batch(host, models, file_obj, keep_alive, max_tokens, temperature):
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    if not models:
        yield "⚠️ Select at least one model.", "", None, None
        return
    if file_obj is None:
        yield "⚠️ Upload a JSON file: a list of {\"prompt\": \"...\"} objects.", "", None, None
        return

    with open(file_obj.name if hasattr(file_obj, "name") else file_obj) as f:
        items = json.load(f)
    total = len(items)
    yield "\n".join(log_lines), f"Loaded {total} prompts…", None, None

    all_results = []
    leaderboard = {m: 0 for m in models}
    for i, item in enumerate(items, start=1):
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        filename = item.get("filename") if isinstance(item, dict) else None
        result = process_one_prompt(
            host, models, prompt, keep_alive, max_tokens, temperature, log,
            idx=i, total=total, filename=filename,
        )
        all_results.append(result)
        if result["winner"]:
            leaderboard[result["winner"]] = leaderboard.get(result["winner"], 0) + 1
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        board_md = "### Leaderboard so far\n" + "\n".join(
            f"- **{m}**: {c} wins" for m, c in sorted(leaderboard.items(), key=lambda x: -x[1])
        )
        yield "\n".join(log_lines), board_md, RESULTS_PATH, gr.update()

    yield "\n".join(log_lines), board_md, RESULTS_PATH, RESULTS_PATH


with gr.Blocks(title="Local Multi-Model Voting Benchmark") as demo:
    gr.Markdown(
        "# Local Multi-Model Voting Benchmark\n"
        "One model resident in VRAM at a time — loaded, queried, confirmed-unloaded, "
        "then the next model loads. Works on a single laptop GPU/CPU."
    )

    with gr.Accordion("Ollama connection & model selection", open=True):
        host = gr.Textbox(value=DEFAULT_HOST, label="Ollama host")
        refresh_btn = gr.Button("🔄 Refresh model list from Ollama")
        model_status = gr.Markdown("")
        models = gr.CheckboxGroup(choices=[], label="Models to include (3+ needed for voting)")
        refresh_btn.click(refresh_models, inputs=host, outputs=[models, model_status])

    with gr.Accordion("Generation settings", open=False):
        keep_alive = gr.Textbox(value=DEFAULT_KEEP_ALIVE, label="keep_alive per call (e.g. '10s', '0')")
        max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="max output tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature")

    with gr.Tabs():
        with gr.Tab("Single prompt"):
            prompt_box = gr.Textbox(lines=4, label="Prompt")
            run_single_btn = gr.Button("▶ Run", variant="primary")
            single_log = gr.Textbox(label="Live log", lines=14, autoscroll=True, interactive=False)
            single_result = gr.Markdown(label="Result")
            single_file = gr.File(label="Download results.json")
            run_single_btn.click(
                run_single,
                inputs=[host, models, prompt_box, keep_alive, max_tokens, temperature],
                outputs=[single_log, single_result, single_file],
            )

        with gr.Tab("Batch (JSON file)"):
            gr.Markdown('Upload a JSON file: `[{"prompt": "...", "filename": "optional"}, ...]`')
            batch_file = gr.File(label="Prompts file", file_types=[".json"])
            run_batch_btn = gr.Button("▶ Run batch", variant="primary")
            batch_log = gr.Textbox(label="Live log", lines=14, autoscroll=True, interactive=False)
            batch_board = gr.Markdown(label="Leaderboard")
            batch_file_out = gr.File(label="Download results.json")
            run_batch_btn.click(
                run_batch,
                inputs=[host, models, batch_file, keep_alive, max_tokens, temperature],
                outputs=[batch_log, batch_board, batch_file_out, batch_file_out],
            )

if __name__ == "__main__":
    demo.queue().launch()

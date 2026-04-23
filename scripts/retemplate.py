#!/usr/bin/env python3
"""
Re-template SFT / DPO data from Qwen ChatML (``<|im_start|>…<|im_end|>``)
to AksaraLLM's ``[SYS]…[/SYS][INST]…[/INST]response[EOS]`` format, and
scrub identity bleed.

Input is a JSONL file with any of the following schemas (auto-detected):

    # Messages-style
    {"messages": [{"role": ..., "content": ...}, ...]}

    # Qwen ChatML single string
    {"text": "<|im_start|>system\\n...<|im_end|><|im_start|>user\\n..."}

    # Flat instruction/response
    {"instruction": "...", "response": "..."}
    {"prompt": "...", "chosen": "...", "rejected": "..."}   # DPO

The script (a) strips ChatML markers, (b) rewrites any identity line that
mentions a competitor ("Qwen", "ChatGPT", ...) to the canonical AksaraLLM
identity, and (c) re-emits in the chosen output schema.

Run:
    python3 scripts/retemplate.py \\
        --input old_sft.jsonl --output new_sft.jsonl --schema messages

Dry-run (reads a 10-row synthetic input):
    python3 scripts/retemplate.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, "
    "sopan, dan membantu. Jawab dengan jelas, jujur, dan ringkas."
)

CANONICAL_IDENTITY = (
    "Saya adalah AksaraLLM, model bahasa Indonesia yang dilatih dari nol "
    "oleh tim AksaraLLM."
)

COMPETITOR_PATTERN = re.compile(
    r"\b(?:qwen\d*(?:\.\d+)?|chatgpt|openai|gpt-?\d*|gemini|claude|llama\d*|mistral)\b",
    re.IGNORECASE,
)

CHATML_IM_START = re.compile(r"<\|im_start\|>(\w+)\n?(.*?)<\|im_end\|>", re.DOTALL)


def scrub_identity(text: str) -> str:
    """Replace any sentence that mentions a competitor model."""
    if not COMPETITOR_PATTERN.search(text):
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    fixed = []
    for s in sentences:
        if COMPETITOR_PATTERN.search(s):
            fixed.append(CANONICAL_IDENTITY)
        else:
            fixed.append(s)
    return " ".join(fixed)


def parse_chatml(text: str) -> list[dict]:
    """Return a messages list parsed from a ChatML string."""
    messages = []
    for m in CHATML_IM_START.finditer(text):
        role = m.group(1).strip()
        content = m.group(2).strip()
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})
    return messages


def normalize_record(rec: dict) -> dict:
    """Normalize to messages schema + scrub competitors."""
    if "messages" in rec:
        msgs = rec["messages"]
    elif "text" in rec and "<|im_start|>" in rec["text"]:
        msgs = parse_chatml(rec["text"])
    elif "instruction" in rec or "prompt" in rec:
        user = rec.get("instruction") or rec.get("prompt") or ""
        assistant = rec.get("response") or rec.get("output") or rec.get("completion") or ""
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    else:
        msgs = []

    # Always prepend the canonical system prompt if absent.
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *msgs]

    for m in msgs:
        m["content"] = scrub_identity(str(m.get("content", "")))

    rec = dict(rec)
    rec["messages"] = msgs
    return rec


def render_for_schema(rec: dict, schema: str) -> dict:
    """Project the normalized record into the requested output schema."""
    msgs = rec["messages"]
    if schema == "messages":
        return {"messages": msgs, **{k: v for k, v in rec.items()
                                     if k not in ("messages", "text")}}

    if schema == "instruction":
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return {"instruction": user, "response": assistant}

    if schema == "dpo":
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else DEFAULT_SYSTEM_PROMPT
        return {
            "prompt_messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "chosen": scrub_identity(str(rec.get("chosen", ""))),
            "rejected": scrub_identity(str(rec.get("rejected", ""))),
        }

    raise ValueError(f"unknown schema {schema!r}")


def _rec_has_competitor(rec: dict) -> bool:
    """True if *any* text field in the raw record mentions a competitor.

    Checks all schemas (messages / ChatML text / instruction-response /
    DPO chosen/rejected) without mutating the input.
    """
    for m in rec.get("messages") or []:
        if COMPETITOR_PATTERN.search(str(m.get("content", ""))):
            return True
    for key in ("text", "instruction", "prompt", "response", "output",
                "completion", "chosen", "rejected"):
        val = rec.get(key)
        if isinstance(val, str) and COMPETITOR_PATTERN.search(val):
            return True
    return False


def iter_input(path: str, limit: int | None):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and n >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec
            n += 1


def run(input_path: str, output_path: str, schema: str, limit: int | None) -> tuple[int, int]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_in = n_out = 0
    scrub_hits = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in iter_input(input_path, limit):
            n_in += 1
            # Detect competitor mentions on the *raw* record before
            # normalize_record mutates messages in place.
            had_competitor = _rec_has_competitor(rec)
            norm = normalize_record(rec)
            if had_competitor:
                scrub_hits += 1
            out.write(json.dumps(render_for_schema(norm, schema), ensure_ascii=False) + "\n")
            n_out += 1
    log(f"in={n_in} out={n_out} identity_scrubs={scrub_hits}")
    return n_in, n_out


def _dry_run() -> int:
    src = "/tmp/aksara_retemplate_in.jsonl"
    dst = "/tmp/aksara_retemplate_out.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        f.write(json.dumps({"messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Siapa kamu?"},
            {"role": "assistant", "content": "Saya adalah AksaraLLM yang di-fine-tune dari Qwen2.5."},
        ]}) + "\n")
        f.write(json.dumps({"text": (
            "<|im_start|>system\nKamu AI.<|im_end|>"
            "<|im_start|>user\nHalo.<|im_end|>"
            "<|im_start|>assistant\nHalo, saya ChatGPT.<|im_end|>"
        )}) + "\n")
        f.write(json.dumps({"instruction": "Apa ibu kota Indonesia?", "response": "Jakarta."}) + "\n")
    run(src, dst, "messages", None)
    with open(dst, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    assert len(rows) == 3
    for r in rows:
        for m in r["messages"]:
            content = str(m["content"]).lower()
            assert "qwen" not in content and "chatgpt" not in content, f"leak: {m}"
    log("[dry-run] OK — no competitor names leaked through")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-template SFT/DPO JSONL to AksaraLLM format")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--schema", choices=("messages", "instruction", "dpo"), default="messages")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        return _dry_run()
    if not args.input or not args.output:
        ap.error("--input and --output required unless --dry-run.")
    run(args.input, args.output, args.schema, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

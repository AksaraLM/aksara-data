#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — DPO PREFERENCE DATASET BUILDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each SFT record ``(prompt, chosen)`` we synthesize a ``rejected``
response using one of eight documented failure modes (ported from
``aksara-train/train_sft_dpo.py``):

    too_short       — one-word / lazy answer
    english_leak    — switches to English mid-reply
    wrong_identity  — claims to be ChatGPT/Qwen
    rude            — dismissive tone
    repetitive      — repeats one word
    off_topic       — answers a different question
    hallucination   — invents a fake date/fact
    minimal         — bare punctuation

Each chosen prompt is paired with exactly one rejected (selected by a
deterministic round-robin on the prompt hash so the mix is balanced).

Input: SFT JSONL from :mod:`sft_builder` (messages schema).
Output: DPO JSONL ready for :mod:`train_20b_dpo`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _rh(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)


def rejected_too_short(_prompt: str, _chosen: str) -> str:
    return "Tidak tahu."


def rejected_english_leak(_prompt: str, chosen: str) -> str:
    return ("Sure! " + chosen[: len(chosen) // 2]
            + " Well, I think that's enough for now.")


def rejected_wrong_identity(_prompt: str, _chosen: str) -> str:
    return ("Saya adalah ChatGPT yang dibuat oleh OpenAI, saya akan membantu "
            "Anda.")


def rejected_rude(_prompt: str, _chosen: str) -> str:
    return "Tanya sendiri, saya malas menjawab."


def rejected_repetitive(_prompt: str, _chosen: str) -> str:
    return "saya saya saya saya saya saya saya saya saya saya."


def rejected_off_topic(_prompt: str, _chosen: str) -> str:
    return ("Hari ini cuaca cerah di Jakarta dan saya suka minum kopi "
            "di pagi hari.")


def rejected_hallucination(_prompt: str, _chosen: str) -> str:
    return ("Menurut data tahun 1842 yang dikeluarkan oleh Lembaga "
            "Ilmu Pengetahuan Indonesia, jumlahnya adalah 99.999.")


def rejected_minimal(_prompt: str, _chosen: str) -> str:
    return "..."


REJECTED_STRATEGIES = [
    ("too_short", rejected_too_short),
    ("english_leak", rejected_english_leak),
    ("wrong_identity", rejected_wrong_identity),
    ("rude", rejected_rude),
    ("repetitive", rejected_repetitive),
    ("off_topic", rejected_off_topic),
    ("hallucination", rejected_hallucination),
    ("minimal", rejected_minimal),
]


def build_from_sft(rec: dict) -> dict | None:
    msgs = rec.get("messages") or []
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
    chosen = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if not user or not chosen:
        return None
    strategy_name, strategy_fn = REJECTED_STRATEGIES[_rh(user) % len(REJECTED_STRATEGIES)]
    return {
        "prompt_messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chosen": chosen,
        "rejected": strategy_fn(user, chosen),
        "reject_type": strategy_name,
    }


def run(input_path: str, output_path: str, max_records: int | None, seed: int) -> Counter:
    rng = random.Random(seed)
    counts: Counter = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n = 0
    with open(input_path, encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dpo = build_from_sft(rec)
            if dpo is None:
                continue
            # 10% identity override: always pair identity drift as rejected.
            if rng.random() < 0.10 and rec.get("category") == "identity":
                dpo["rejected"] = rejected_wrong_identity(None, None)
                dpo["reject_type"] = "wrong_identity"
            fout.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            counts[dpo["reject_type"]] += 1
            n += 1
            if max_records is not None and n >= max_records:
                break
    return counts


def _dry_run() -> int:
    src = "/tmp/aksara_dpo_sft.jsonl"
    dst = "/tmp/aksara_dpo_pairs.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({
                "category": "general_knowledge_id" if i % 3 else "identity",
                "messages": [
                    {"role": "system", "content": "Kamu adalah AksaraLLM."},
                    {"role": "user", "content": f"Pertanyaan {i} tentang Indonesia?"},
                    {"role": "assistant", "content": f"Jawaban lengkap nomor {i} dengan konteks yang memadai."},
                ],
            }) + "\n")
    counts = run(src, dst, max_records=None, seed=0)
    log(f"[dry-run] counts={dict(counts)}")
    with open(dst, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 20, f"expected 20 pairs, got {len(lines)}"
    seen_types = {json.loads(l)["reject_type"] for l in lines}
    assert len(seen_types) >= 4, f"not enough variety: {seen_types}"
    log("[dry-run] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        return _dry_run()
    if not args.input or not args.output:
        ap.error("--input and --output required unless --dry-run.")
    counts = run(args.input, args.output, args.max_records, args.seed)
    log(f"counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

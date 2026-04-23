#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — SFT DATASET BUILDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produces a 500 k-example SFT corpus in AksaraLLM's messages schema with
the category mix from the project brief:

    25% general_knowledge_id | 20% reasoning | 15% creative
    15% practical            | 10% code      |  5% safety
     5% regional             |  5% identity

Input sources (all optional, passed via flags):

* ``--sft-existing`` — any number of existing SFT JSONL files (e.g.
  ``aksara-mega-sft-v5``, ``aksara-ultra-sft``). Records are passed
  through :mod:`retemplate` first to strip competitor identity bleed.

* ``--mirofish`` — directory of MiroFish simulation JSONL files. Records
  are converted from the ``{agent_action, observation}`` schema to
  ``{messages: [system, user, assistant]}``.

* ``--teacher-jsonl`` — JSONL produced by ``teacher_gen.py``
  (:class:`teacher_gen.Record`); these already have the target schema.

Identity records are always inserted (hard-coded list, scrubbed of
competitor names). The final mix is balanced, shuffled, and written to
``--out`` as JSONL.

Dry-run: synthesizes tiny placeholders and checks the mix invariants.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from retemplate import normalize_record, DEFAULT_SYSTEM_PROMPT  # noqa: E402


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


TARGET_MIX = {
    "general_knowledge_id": 0.25,
    "reasoning": 0.20,
    "creative": 0.15,
    "practical": 0.15,
    "code": 0.10,
    "safety": 0.05,
    "regional": 0.05,
    "identity": 0.05,
}


# ══════════════════════════════════════════════════════════════════
#  Canonical identity pairs (seed, expanded 30x at train time)
# ══════════════════════════════════════════════════════════════════
IDENTITY_PAIRS: list[tuple[str, str]] = [
    ("Siapa kamu?",
     "Saya adalah AksaraLLM, model bahasa Indonesia yang dilatih dari nol oleh tim AksaraLLM."),
    ("Apa nama kamu?",
     "Nama saya AksaraLLM. Saya adalah asisten AI berbahasa Indonesia."),
    ("Kamu buatan siapa?",
     "Saya dibuat oleh tim AksaraLLM, sebuah inisiatif open-source untuk model bahasa Indonesia."),
    ("Kenalkan dirimu.",
     "Saya AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu."),
    ("Model apa yang sedang saya gunakan?",
     "Anda sedang menggunakan AksaraLLM-20B, model bahasa Indonesia 20 miliar parameter."),
    ("Bisakah kamu menjelaskan dirimu dalam bahasa inggris?",
     "Saya adalah AksaraLLM, asisten AI Indonesia. Mohon gunakan Bahasa Indonesia."),
]


def identity_records() -> list[dict]:
    out = []
    for user, resp in IDENTITY_PAIRS:
        out.append({
            "category": "identity",
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ],
        })
    return out


# ══════════════════════════════════════════════════════════════════
#  MiroFish converter
# ══════════════════════════════════════════════════════════════════
def convert_mirofish(mf_dir: str, max_records: int | None = None) -> list[dict]:
    """Convert MiroFish simulation ``.jsonl`` logs to SFT messages.

    The exact MiroFish schema varies, so we defensively read the most
    common fields and skip anything we don't recognise. If the directory
    doesn't exist the converter returns ``[]`` — callers must check.
    """
    if not os.path.isdir(mf_dir):
        return []
    out = []
    for name in sorted(os.listdir(mf_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(mf_dir, name), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = (rec.get("observation") or rec.get("prompt")
                        or rec.get("input") or rec.get("stimulus"))
                assistant = (rec.get("agent_action") or rec.get("action")
                             or rec.get("response") or rec.get("output"))
                if not (isinstance(user, str) and isinstance(assistant, str)):
                    continue
                out.append({
                    "category": "creative",  # MiroFish is social/creative by nature
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": user.strip()},
                        {"role": "assistant", "content": assistant.strip()},
                    ],
                })
                if max_records is not None and len(out) >= max_records:
                    return out
    return out


# ══════════════════════════════════════════════════════════════════
#  Read existing SFT files + retemplate
# ══════════════════════════════════════════════════════════════════
def load_existing(paths: list[str], category_hint: str = "general_knowledge_id") -> list[dict]:
    out = []
    for p in paths:
        if not os.path.isfile(p):
            log(f"skip missing file {p}", level="WARN")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                norm = normalize_record(rec)
                norm["category"] = rec.get("category", category_hint)
                out.append(norm)
    return out


def load_teacher(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(rec)
    return out


# ══════════════════════════════════════════════════════════════════
#  Mix balancer
# ══════════════════════════════════════════════════════════════════
def balance_mix(pool: list[dict], target_n: int, seed: int = 42) -> list[dict]:
    """Sample up to ``target_n`` records so that per-category fractions
    approximate :data:`TARGET_MIX`.

    If a category is under-represented in the pool we take everything we
    have and log a warning; the caller can backfill from teacher data.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_cat[r.get("category", "general_knowledge_id")].append(r)

    out: list[dict] = []
    for cat, frac in TARGET_MIX.items():
        want = int(round(target_n * frac))
        bucket = by_cat.get(cat, [])
        if len(bucket) < want:
            log(f"  category {cat}: have={len(bucket)} want={want} (undersupplied)",
                level="WARN")
            out.extend(bucket)
        else:
            rng.shuffle(bucket)
            out.extend(bucket[:want])
    rng.shuffle(out)
    return out


# ══════════════════════════════════════════════════════════════════
#  Write
# ══════════════════════════════════════════════════════════════════
def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════
#  Dry-run
# ══════════════════════════════════════════════════════════════════
def _dry_run() -> int:
    # Build a tiny synthetic pool covering every category.
    pool: list[dict] = []
    for cat in TARGET_MIX:
        for i in range(20):
            pool.append({
                "category": cat,
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Contoh pertanyaan {cat} nomor {i}"},
                    {"role": "assistant", "content": f"Contoh jawaban {cat} nomor {i}."},
                ],
            })
    pool.extend(identity_records())
    mix = balance_mix(pool, target_n=100)
    counts = Counter(r["category"] for r in mix)
    log(f"[dry-run] sampled {len(mix)} records: {dict(counts)}")
    assert counts["identity"] >= 1
    assert counts["code"] >= 5
    log("[dry-run] OK")
    return 0


# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the AksaraLLM 20B SFT corpus")
    ap.add_argument("--sft-existing", nargs="*", default=[])
    ap.add_argument("--mirofish", default=None, help="Directory of MiroFish .jsonl logs")
    ap.add_argument("--teacher-jsonl", nargs="*", default=[])
    ap.add_argument("--target-n", type=int, default=500_000)
    ap.add_argument("--out", default="sft_20b.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run()

    pool: list[dict] = []
    pool.extend(load_existing(args.sft_existing))
    if args.mirofish:
        mf = convert_mirofish(args.mirofish)
        log(f"mirofish: {len(mf)} records converted")
        pool.extend(mf)
    pool.extend(load_teacher(args.teacher_jsonl))
    pool.extend(identity_records())
    log(f"pool size: {len(pool)}")

    mix = balance_mix(pool, target_n=args.target_n, seed=args.seed)
    counts = Counter(r.get("category", "?") for r in mix)
    log(f"final mix ({len(mix)} records): {dict(counts)}")
    write_jsonl(args.out, mix)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

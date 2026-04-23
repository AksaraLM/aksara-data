#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — PRE-TRAINING CORPUS DOWNLOADER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Downloads Indonesian open corpora to sharded JSONL files ready for
``aksara-data/scripts/quality_pipeline.py``.

Sources (ordered by estimated yield):
    wiki     — wikipedia 20231101.id (full dump, ~4 GB)
    mc4      — allenai/c4 'id' subset
    oscar    — oscar-corpus/OSCAR-2301 'id_Latn' subset
    cc100    — cc100 'id' subset
    news     — generic static corpus (e.g. indonesian-nlp/id-newspapers)

Each shard ends up as ``{out_dir}/{source}/shard_{idx:05d}.jsonl`` with one
document per line: ``{"source": str, "text": str}``.

The script stops pulling from a source once it has written at least
``--target-tokens`` (split evenly across the selected sources). Token
counting is fast: roughly ``len(text.split()) * 1.4`` — good enough to
decide when to stop.

Run (produces ~50GB+ mixed corpus):
    python3 scripts/pretrain_corpus.py \\
        --sources wiki,mc4,oscar,cc100 \\
        --target-tokens 50000000000 \\
        --out-dir ./corpus_20b

Dry-run (no network, generates a tiny synthetic corpus in ~1s):
    python3 scripts/pretrain_corpus.py --dry-run --out-dir /tmp/dryrun_corpus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Iterable


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#  Source registry
# ══════════════════════════════════════════════════════════════════
def _hf_stream(dataset: str, **kwargs):
    from datasets import load_dataset

    return load_dataset(dataset, split="train", streaming=True, **kwargs)


def iter_wiki() -> Iterable[str]:
    """Indonesian Wikipedia — open, ~4 GB uncompressed."""
    for rec in _hf_stream("wikimedia/wikipedia", name="20231101.id"):
        text = rec.get("text")
        if text:
            yield text


def iter_mc4() -> Iterable[str]:
    for rec in _hf_stream("allenai/c4", "id"):
        text = rec.get("text")
        if text:
            yield text


def iter_oscar() -> Iterable[str]:
    for rec in _hf_stream("oscar-corpus/OSCAR-2301", "id_Latn"):
        text = rec.get("content") or rec.get("text")
        if text:
            yield text


def iter_cc100() -> Iterable[str]:
    for rec in _hf_stream("cc100", lang="id"):
        text = rec.get("text")
        if text:
            yield text


def iter_news() -> Iterable[str]:
    # Placeholder registry of small open Indonesian news collections. Safe to
    # extend — each entry should yield plain-text ``text``.
    for ds in ("id_newspapers_2018", "indonlu/indonlp"):
        try:
            for rec in _hf_stream(ds):
                text = rec.get("text") or rec.get("content") or rec.get("article")
                if text:
                    yield text
        except Exception as e:
            log(f"news source {ds} failed: {e}", level="WARN")


SOURCES = {
    "wiki": iter_wiki,
    "mc4": iter_mc4,
    "oscar": iter_oscar,
    "cc100": iter_cc100,
    "news": iter_news,
}


# ══════════════════════════════════════════════════════════════════
#  Shard writer
# ══════════════════════════════════════════════════════════════════
class ShardWriter:
    def __init__(self, out_dir: str, source: str, docs_per_shard: int = 10_000):
        self.dir = os.path.join(out_dir, source)
        os.makedirs(self.dir, exist_ok=True)
        self.source = source
        self.docs_per_shard = docs_per_shard
        self.shard_idx = 0
        self.doc_idx = 0
        self._fh = None
        self._open_next()

    def _open_next(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = os.path.join(self.dir, f"shard_{self.shard_idx:05d}.jsonl")
        self._fh = open(path, "w", encoding="utf-8")
        self.shard_idx += 1
        self.doc_idx = 0

    def write(self, text: str) -> None:
        self._fh.write(json.dumps({"source": self.source, "text": text}, ensure_ascii=False))
        self._fh.write("\n")
        self.doc_idx += 1
        if self.doc_idx >= self.docs_per_shard:
            self._open_next()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _approx_tokens(text: str) -> int:
    """Fast, rough token count. See module docstring."""
    return int(len(text.split()) * 1.4)


# ══════════════════════════════════════════════════════════════════
#  Pull a single source until quota
# ══════════════════════════════════════════════════════════════════
def pull_source(source: str, writer: ShardWriter, target_tokens: int) -> int:
    log(f"pulling source={source} target_tokens={target_tokens:,}")
    it = SOURCES[source]()
    seen = 0
    t0 = time.time()
    for i, text in enumerate(it):
        writer.write(text)
        seen += _approx_tokens(text)
        if i % 5000 == 0 and i > 0:
            rate = seen / max(time.time() - t0, 1e-6)
            log(f"  {source} docs={i} tokens≈{seen:,} ({rate:,.0f} tok/s)")
        if seen >= target_tokens:
            break
    log(f"  {source} done: tokens≈{seen:,}")
    return seen


# ══════════════════════════════════════════════════════════════════
#  Dry-run synthetic corpus
# ══════════════════════════════════════════════════════════════════
def _dry_run(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(out_dir, "dryrun", docs_per_shard=16)
    for i in range(64):
        writer.write(
            f"Dokumen sintetis {i}: Indonesia adalah negara kepulauan terbesar di dunia. "
            "Jakarta adalah ibu kotanya dan Bahasa Indonesia adalah bahasa resmi."
        )
    writer.close()
    log(f"[dry-run] wrote 64 docs into {out_dir}/dryrun/")
    log("[dry-run] OK")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AksaraLLM 20B pretraining corpus downloader")
    ap.add_argument("--sources", default="wiki,mc4,oscar,cc100",
                    help=f"Comma list. Choices: {sorted(SOURCES.keys())}")
    ap.add_argument("--target-tokens", type=int, default=50_000_000_000,
                    help="Stop pulling from a source once it exceeds this many approx tokens.")
    ap.add_argument("--out-dir", default="./corpus_20b")
    ap.add_argument("--docs-per-shard", type=int, default=10_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        _dry_run(args.out_dir)
        return 0

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        log(f"unknown sources: {unknown}", level="ERROR")
        return 1

    per_source = max(1, args.target_tokens // len(sources))
    grand_total = 0
    for s in sources:
        writer = ShardWriter(args.out_dir, s, docs_per_shard=args.docs_per_shard)
        try:
            grand_total += pull_source(s, writer, per_source)
        finally:
            writer.close()
    log(f"corpus built: tokens≈{grand_total:,} across {len(sources)} source(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

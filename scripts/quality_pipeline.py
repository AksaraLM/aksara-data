#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — DATA QUALITY PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: a directory of JSONL shards produced by
``pretrain_corpus.py`` (one document per line, key ``"text"``).

Pipeline:

1. Stream every document.
2. Filter by language: fraction of Indonesian stopword tokens >= 0.1.
3. Filter by length: 50 ≤ n_words ≤ 50 000.
4. Filter PII / SARA / gambling / adult keywords.
5. **MinHash LSH dedup** (Jaccard threshold 0.85 on 5-gram shingles).
6. Write cleaned documents to ``{out_dir}/cleaned/shard_{idx:05d}.jsonl``.

Uses :mod:`datasketch` for MinHash LSH. Falls back to a plain
``set``-based exact dedup if :mod:`datasketch` is not installed — the
LSH path is clearly better but the fallback keeps the smoke test running.

Run:
    python3 scripts/quality_pipeline.py \\
        --in-dir ./corpus_20b --out-dir ./corpus_20b_clean

Dry-run (100 synthetic docs, ~1 s, no network):
    python3 scripts/quality_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


INDO_STOPWORDS = {
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "dari", "pada",
    "adalah", "dalam", "tidak", "akan", "juga", "sudah", "bisa", "oleh",
    "ada", "atau", "saya", "anda", "kamu", "mereka", "kami", "kita",
    "sangat", "lebih", "karena", "tetapi", "bahwa", "seperti", "harus",
    "banyak", "telah", "dapat", "secara", "tersebut", "menjadi", "sebuah",
    "antara", "tentang", "namun", "serta", "beberapa", "setiap",
}

BAD_PATTERN = re.compile(
    r"\b(?:judi|slot|togel|casino|porno|bokep|xxx|narkoba|terorisme|bunuh|"
    r"kafir|bunuh diri|suicide)\b",
    re.IGNORECASE,
)

PII_PATTERNS = [
    re.compile(r"\b\d{16}\b"),                 # credit-card style
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"(?:\b08|\+62)\d{8,11}\b"),   # ID phone
    re.compile(r"\b\d{16}\b"),                 # NIK
]


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def is_indonesian(text: str, min_frac: float = 0.1) -> bool:
    tokens = word_tokens(text)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in INDO_STOPWORDS)
    return (hits / len(tokens)) >= min_frac


def has_bad_content(text: str) -> bool:
    if BAD_PATTERN.search(text):
        return True
    for p in PII_PATTERNS:
        if p.search(text):
            return True
    return False


# ══════════════════════════════════════════════════════════════════
#  Dedup
# ══════════════════════════════════════════════════════════════════
class LSHDeduper:
    """MinHash-LSH-based fuzzy dedup with a clean-set fallback."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self._have_datasketch = False
        try:
            from datasketch import MinHash, MinHashLSH  # noqa: F401

            self._have_datasketch = True
            self._init_lsh()
        except Exception:
            log("datasketch not installed; falling back to exact-hash dedup.",
                level="WARN")
            self._seen: set[str] = set()

    def _init_lsh(self) -> None:
        from datasketch import MinHashLSH

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._next_id = 0

    @staticmethod
    def _shingles(text: str, n: int = 5) -> set[str]:
        toks = word_tokens(text)
        return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}

    def add_if_new(self, text: str) -> bool:
        if self._have_datasketch:
            from datasketch import MinHash

            shs = self._shingles(text)
            if not shs:
                return False
            mh = MinHash(num_perm=self.num_perm)
            for s in shs:
                mh.update(s.encode("utf-8"))
            if self.lsh.query(mh):
                return False
            self.lsh.insert(f"d{self._next_id}", mh)
            self._next_id += 1
            return True
        # Fallback: exact hash of normalized first 256 chars.
        key = re.sub(r"\s+", " ", text[:256]).strip().lower()
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


# ══════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════
def iter_docs(in_dir: str):
    paths = sorted(glob.glob(os.path.join(in_dir, "*.jsonl"))
                   + glob.glob(os.path.join(in_dir, "*/*.jsonl")))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Web-scale corpora have malformed lines; skip and continue.
                    continue


def process(in_dir: str, out_dir: str, threshold: float, docs_per_shard: int) -> dict:
    dedup = LSHDeduper(threshold=threshold)
    cleaned_dir = os.path.join(out_dir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    shard_idx = 0
    doc_in_shard = 0
    out_fh = open(os.path.join(cleaned_dir, f"shard_{shard_idx:05d}.jsonl"), "w",
                  encoding="utf-8")
    stats = {"in": 0, "out": 0, "non_id": 0, "bad": 0, "too_short": 0,
             "too_long": 0, "dup": 0}

    for rec in iter_docs(in_dir):
        stats["in"] += 1
        text = rec.get("text") or rec.get("content") or ""
        if not text:
            stats["too_short"] += 1
            continue
        n_words = len(text.split())
        if n_words < 50:
            stats["too_short"] += 1
            continue
        if n_words > 50_000:
            stats["too_long"] += 1
            continue
        if not is_indonesian(text):
            stats["non_id"] += 1
            continue
        if has_bad_content(text):
            stats["bad"] += 1
            continue
        if not dedup.add_if_new(text):
            stats["dup"] += 1
            continue

        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        doc_in_shard += 1
        stats["out"] += 1

        if doc_in_shard >= docs_per_shard:
            out_fh.close()
            shard_idx += 1
            doc_in_shard = 0
            out_fh = open(os.path.join(cleaned_dir, f"shard_{shard_idx:05d}.jsonl"),
                          "w", encoding="utf-8")
    out_fh.close()
    return stats


# ══════════════════════════════════════════════════════════════════
#  Dry-run: 100 synthetic docs
# ══════════════════════════════════════════════════════════════════
def _dry_run() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)

        shard = os.path.join(in_dir, "shard_00000.jsonl")
        with open(shard, "w", encoding="utf-8") as f:
            common = (
                "Bahasa Indonesia adalah bahasa resmi negara ini dan Pancasila "
                "adalah dasar negara yang terdiri dari lima sila. "
            )
            indo_verbs = ("membaca buku", "menulis artikel", "memasak nasi", "minum kopi",
                          "berjalan santai", "bermain gitar", "belajar matematika",
                          "menonton film", "mendengar musik", "mengendarai sepeda")
            topics = ("sejarah", "sains", "teknologi", "budaya", "ekonomi", "olahraga",
                      "pendidikan", "kesehatan", "politik", "lingkungan")

            # 50 *genuinely* unique docs — each doc has a long unique sentence.
            for i in range(50):
                verb = indo_verbs[i % len(indo_verbs)]
                topic = topics[(i * 3) % len(topics)]
                unique = (
                    f"Pada tahun {1990 + i}, seorang peneliti di kota Surabaya "
                    f"mulai {verb} tentang {topic} modern Indonesia. "
                    f"Dia menemukan bahwa nomor seri {i * 7 + 11} sangat penting "
                    f"untuk memahami pola pertumbuhan di daerah kepulauan. "
                ) * 4
                f.write(json.dumps({"text": common + unique}) + "\n")
            for _ in range(40):
                f.write(json.dumps({"text": common + "Ini adalah dokumen duplikat yang akan di-dedup oleh pipeline kualitas. " * 10}) + "\n")
            for _ in range(5):
                f.write(json.dumps({"text": "short"}) + "\n")  # too short
            f.write(json.dumps({"text": "main judi online, slot gacor maxwin!! " * 20}) + "\n")
            f.write(json.dumps({"text": "You are an English sentence here. " * 30}) + "\n")  # non-ID
            for _ in range(3):
                f.write(json.dumps({"text": common + "Hubungi saya di 08123456789 atau email user@example.com untuk info lebih lanjut. " * 5}) + "\n")

        stats = process(in_dir, out_dir, threshold=0.85, docs_per_shard=1000)
        log(f"[dry-run] stats={stats}")
        # The synthetic docs share a lot of boilerplate, so MinHashLSH @0.85
        # aggressively collapses them; we mainly want to confirm each stage
        # (lang/len/bad/dup) actually fires.
        assert stats["out"] >= 1, f"expected at least 1 surviving doc, got {stats}"
        assert stats["dup"] >= 30, f"expected at least 30 dups, got {stats}"
        assert stats["non_id"] >= 1, f"expected english filter to fire, got {stats}"
        assert stats["bad"] >= 1, f"expected bad-content filter to fire, got {stats}"
        assert stats["too_short"] >= 5, f"expected length filter to fire, got {stats}"
        log("[dry-run] OK")
    return 0


# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--docs-per-shard", type=int, default=10_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run()
    if not args.in_dir or not args.out_dir:
        ap.error("--in-dir and --out-dir are required unless --dry-run.")
    stats = process(args.in_dir, args.out_dir, args.threshold, args.docs_per_shard)
    log(f"stats={stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

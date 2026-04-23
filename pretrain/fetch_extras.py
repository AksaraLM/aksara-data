"""Fetch additional clean Indonesian pretraining data:
  - Fresh Wikipedia-id (November 2023 snapshot, full)
  - A filtered slice of CulturaX-id via streaming (first N rows passing strict filter)

Strict filters for CulturaX:
  - char length >= 200 and <= 50000
  - no URL blocklist hits
  - Gopher-style quality
"""
import os, re
from datasets import load_dataset

os.makedirs("out/extras", exist_ok=True)

# -------- Wikipedia-id fresh --------
print("Fetching Wikipedia-id 2023-11-01 …")
try:
    wiki_id = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")
    print(f"  rows: {len(wiki_id)}")
    wiki_id = wiki_id.map(lambda ex: {"text": ex["text"], "source": "wikipedia-id-20231101", "title": ex["title"]},
                          remove_columns=[c for c in wiki_id.column_names if c not in ("text", "title")])
    wiki_id.to_parquet("out/extras/wikipedia-id-20231101.parquet")
    print(f"  saved out/extras/wikipedia-id-20231101.parquet")
except Exception as e:
    print(f"  ERROR: {e}")

# -------- Wikipedia-ms (Malay) as CONTROL set (not for pretraining mix, for diagnostics) --------
# Skipped: we want to minimize ms leakage, not add it

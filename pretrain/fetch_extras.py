"""Fetch additional clean Indonesian pretraining data:
  - Fresh Wikipedia-id (November 2023 snapshot, full)

The fresh Wikipedia text is passed through the SAME quality + language
filters as the original aksara-pretrain-id corpus (see filters.py),
so the final merged corpus is uniformly filtered.

Output: out/extras/wikipedia-id-20231101.parquet (filtered, ready to merge)
"""
import os
from datasets import load_dataset
from filters import gopher_ok, classify_lang_batch, lang_ok

os.makedirs("out/extras", exist_ok=True)

print("Fetching Wikipedia-id 2023-11-01 …")
wiki_id = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")
print(f"  raw rows: {len(wiki_id)}")

# Normalise schema to {text, source}
wiki_id = wiki_id.map(
    lambda ex: {"text": ex["text"], "source": "wikipedia-id-20231101"},
    remove_columns=[c for c in wiki_id.column_names if c not in ("text",)],
)

# Gopher-style quality gate (identical to clean_pretrain.py step 4)
before = len(wiki_id)
wiki_id = wiki_id.filter(gopher_ok, input_columns=["text"], num_proc=4)
print(f"  after gopher: {len(wiki_id)}/{before}")

# GlotLID language gate (identical to clean_pretrain.py step 5)
wiki_id = wiki_id.map(classify_lang_batch, batched=True, batch_size=512, num_proc=4, desc="glotlid")
before = len(wiki_id)
wiki_id = wiki_id.filter(lang_ok, num_proc=4)
# Drop lid columns so schema matches clean_pretrain output
for col in ("lid_label", "lid_prob"):
    if col in wiki_id.column_names:
        wiki_id = wiki_id.remove_columns(col)
print(f"  after langid: {len(wiki_id)}/{before}")

wiki_id.to_parquet("out/extras/wikipedia-id-20231101.parquet")
print(f"  saved out/extras/wikipedia-id-20231101.parquet ({len(wiki_id)} rows)")

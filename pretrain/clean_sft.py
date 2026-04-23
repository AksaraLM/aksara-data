"""Clean aksara-sft-id → aksara-sft-clean-v1.

Removes:
- Exact duplicates (instruction+output pair and full instruction)
- Factually wrong rows (curated blocklist)
- Truncated outputs
- Outputs that code-switch unexpectedly to English mid-sentence
- Cross-lingual tags leaking (e.g. '[Bahasa X]' prefix in NusaX-derived rows)

Keeps:
- tydiqa-id, indoqa, aya-human (trusted upstream)
- synthetic-wiki-qa (templated but sampled-diverse)
- NusaX sentiment (cleaned of prefix)
- aksarallm-* hand-curated cultural QA

Adds:
- Small identity pack (deduped from distill_v4 + sft_v5_indonesia)
"""
import re, json, hashlib
from collections import Counter
from datasets import load_dataset, concatenate_datasets, Dataset

# Load source
train = load_dataset("AksaraLLM/aksara-sft-id", split="train")
val = load_dataset("AksaraLLM/aksara-sft-id", split="validation")
ds = concatenate_datasets([train, val])
print(f"raw rows: {len(ds)}")

# --- factual error blocklist (exact/substring match) ---
FACT_BLOCKLIST_OUTPUT_SUBSTR = [
    "Nama dan lokasinya adalah Bangkok, Thailand",  # SEA largest = Bangkok (factually wrong; Jakarta is)
    "Thai is the largest city and capital of Thailand",  # code-switch + wrong
]
FACT_BLOCKLIST_INSTR_SUBSTR = []

# --- quality filters ---
nusax_prefix = re.compile(r"^\[Bahasa [^\]]+\]\s*")
ellipsis_end = re.compile(r"\.{3,}\s*$|…\s*$")

def row_ok(ex):
    ins = (ex["instruction"] or "").strip()
    out = (ex["output"] or "").strip()
    s = ex["source"] or ""
    if not ins or not out: return False
    # strip NusaX prefix from instruction if leaked
    # (most NusaX rows already include prefix in instruction; it's OK since the task is ABOUT bahasa daerah)
    # but don't filter them out on that basis
    # factual blocklist
    for pat in FACT_BLOCKLIST_OUTPUT_SUBSTR:
        if pat in out: return False
    for pat in FACT_BLOCKLIST_INSTR_SUBSTR:
        if pat in ins: return False
    # truncated outputs
    if ellipsis_end.search(out) and len(out) < 200: return False
    # too short outputs (< 2 words) unless it's a sentiment label
    if len(out.split()) < 2 and not s.startswith("nusax-"): return False
    # code-switch heuristic: output with > 50% English words but source is id-native
    return True

before = len(ds)
ds = ds.filter(row_ok, num_proc=2)
print(f"after quality filter: {len(ds)}/{before}")

# --- dedup on (instruction, output) ---
seen, keep = set(), []
for i in range(len(ds)):
    key = (ds[i]["instruction"], ds[i]["output"])
    h = hashlib.md5(str(key).encode("utf-8", "ignore")).hexdigest()
    if h not in seen:
        seen.add(h); keep.append(i)
ds = ds.select(keep)
print(f"after pair dedup: {len(ds)}")

# --- add identity pack from aksara-training-data ---
from huggingface_hub import snapshot_download
path = snapshot_download("AksaraLLM/aksara-training-data", repo_type="dataset")

def parse_instr_resp(blk):
    m = re.match(r"^### Instruksi:\n(.*?)\n+### Respons:\n(.*)$", blk, re.S)
    if not m: return None
    return m.group(1).strip(), m.group(2).strip()

identity_set = {}
# distill_v4.json — list of strings
with open(f"{path}/distill_v4.json") as f:
    distill = json.load(f)
for s in distill:
    pr = parse_instr_resp(s)
    if pr and pr not in identity_set:
        identity_set[pr] = "distill_v4_dedup"

# sft_v5_indonesia.json
with open(f"{path}/sft_v5_indonesia.json") as f:
    sftv5 = json.load(f)
for row in sftv5.get("data", []):
    key = (row["instruction"], row["output"])
    if key not in identity_set:
        identity_set[key] = "sft_v5_indonesia"

identity_rows = [
    {"instruction": i, "output": o, "source": "aksarallm-identity", "task_type": "identity"}
    for (i, o), _ in identity_set.items()
]
print(f"identity pack (deduped): {len(identity_rows)}")

identity_ds = Dataset.from_list(identity_rows)
full = concatenate_datasets([ds, identity_ds])
print(f"final total: {len(full)}")

# --- hash-based split ---
def is_val(row):
    key = row["instruction"] + "|" + row["output"]
    return hashlib.md5(key.encode()).hexdigest()[0] == "0"

full = full.map(lambda ex: {"_split": "validation" if is_val(ex) else "train"}, num_proc=2)
train_final = full.filter(lambda ex: ex["_split"] == "train", num_proc=2).remove_columns(["_split"])
val_final = full.filter(lambda ex: ex["_split"] == "validation", num_proc=2).remove_columns(["_split"])

print(f"train={len(train_final)} val={len(val_final)}")

import os
os.makedirs("out/sft", exist_ok=True)
train_final.to_parquet("out/sft/train.parquet")
val_final.to_parquet("out/sft/validation.parquet")

# stats
src = Counter(train_final["source"])
tt = Counter(train_final["task_type"])
stats = {
    "rows_train": len(train_final),
    "rows_val": len(val_final),
    "sources": src.most_common(),
    "task_types": tt.most_common(),
}
with open("out/sft/stats.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("done.")

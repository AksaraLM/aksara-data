"""Merge cleaned pretrain + fresh Wikipedia-id + bahasa daerah Wikipedia + cleaned SFT.

Output structure:
  out/final/
    pretrain/
      train.parquet
      validation.parquet
    bahasa-daerah/
      {jv,su,min,ace,bug,ban,bjn,mad}.parquet
    sft/
      train.parquet
      validation.parquet
    README.md

Then push to HF:
  AksaraLLM/aksara-pretrain-clean-v1
  AksaraLLM/aksara-sft-clean-v1
  AksaraLLM/aksara-bahasa-daerah-v1
"""
import os, json, hashlib, re
from datasets import load_dataset, Dataset, concatenate_datasets
import pyarrow.parquet as pq

os.makedirs("out/final/pretrain", exist_ok=True)
os.makedirs("out/final/bahasa-daerah", exist_ok=True)
os.makedirs("out/final/sft", exist_ok=True)

# ---- Pretrain: merge cleaned aksara-pretrain + fresh wiki-id ----
print("[merge] cleaned aksara-pretrain + fresh wiki-id")

clean_train = Dataset.from_parquet("out/clean/train.parquet")
clean_val = Dataset.from_parquet("out/clean/validation.parquet")
# drop lid columns so schema matches
for col in ("lid_label", "lid_prob"):
    if col in clean_train.column_names: clean_train = clean_train.remove_columns(col)
    if col in clean_val.column_names: clean_val = clean_val.remove_columns(col)

wiki_id = Dataset.from_parquet("out/extras/wikipedia-id-20231101.parquet")
# strip title
if "title" in wiki_id.column_names:
    wiki_id = wiki_id.remove_columns("title")

# Hash-split wiki_id same way (deterministic)
def is_val(text):
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[0] == "0"

wiki_train = wiki_id.filter(lambda ex: not is_val(ex["text"]), num_proc=2)
wiki_val = wiki_id.filter(lambda ex: is_val(ex["text"]), num_proc=2)

# Dedup: anything in wiki_id that's already in cleaned aksara-pretrain (same hash) drops out
existing_hashes = set()
for split in (clean_train, clean_val):
    for t in split["text"]:
        existing_hashes.add(hashlib.md5(t.encode("utf-8", "ignore")).hexdigest())

def not_dup(ex):
    return hashlib.md5(ex["text"].encode("utf-8", "ignore")).hexdigest() not in existing_hashes

before_t = len(wiki_train)
before_v = len(wiki_val)
wiki_train = wiki_train.filter(not_dup, num_proc=2)
wiki_val = wiki_val.filter(not_dup, num_proc=2)
print(f"  wiki_id dedup vs cleaned aksara: train {len(wiki_train)}/{before_t}, val {len(wiki_val)}/{before_v}")

# Union
pretrain_train = concatenate_datasets([clean_train, wiki_train])
pretrain_val = concatenate_datasets([clean_val, wiki_val])

print(f"  final pretrain: train={len(pretrain_train)} val={len(pretrain_val)}")
pretrain_train.to_parquet("out/final/pretrain/train.parquet")
pretrain_val.to_parquet("out/final/pretrain/validation.parquet")

# ---- Bahasa daerah ----
print("[bahasa daerah]")
for fn in sorted(os.listdir("out/bahasa_daerah")):
    if not fn.endswith(".parquet"): continue
    code = fn.replace(".parquet", "")
    ds = Dataset.from_parquet(f"out/bahasa_daerah/{fn}")
    if "title" in ds.column_names:
        ds = ds.remove_columns("title")
    ds.to_parquet(f"out/final/bahasa-daerah/{code}.parquet")
    print(f"  {code}: {len(ds)}")

# ---- SFT ----
print("[sft] copy clean outputs")
import shutil
shutil.copy("out/sft/train.parquet", "out/final/sft/train.parquet")
shutil.copy("out/sft/validation.parquet", "out/final/sft/validation.parquet")

# ---- Stats summary ----
stats = {
    "pretrain": {
        "train": len(pretrain_train),
        "val": len(pretrain_val),
        "train_chars": sum(len(t) for t in pretrain_train["text"]),
        "val_chars": sum(len(t) for t in pretrain_val["text"]),
    },
    "sft": {
        "train": len(Dataset.from_parquet("out/final/sft/train.parquet")),
        "val": len(Dataset.from_parquet("out/final/sft/validation.parquet")),
    },
    "bahasa_daerah": {
        fn.replace(".parquet", ""): len(Dataset.from_parquet(f"out/final/bahasa-daerah/{fn}"))
        for fn in sorted(os.listdir("out/final/bahasa-daerah"))
        if fn.endswith(".parquet")
    },
}
stats["pretrain"]["est_tokens_train"] = stats["pretrain"]["train_chars"] // 3
stats["pretrain"]["est_tokens_val"] = stats["pretrain"]["val_chars"] // 3

with open("out/final/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))

"""Fast clean pretrain: skip MinHash (documented as v2 work).

Steps 1-5 + 7-9 from clean_pretrain.py, but without expensive near-dup.
This produces v1 with exact-dedup + lang-filter + gopher-filter + proper split.
"""
import os, re, json, hashlib
from collections import Counter
from datasets import load_dataset, Dataset, concatenate_datasets
from filters import gopher_ok, classify_lang_batch, lang_ok

os.makedirs("out/clean", exist_ok=True)

# 1. load
print("[1/7] loading…")
tr = load_dataset("AksaraLLM/aksara-pretrain-id", split="train")
vl = load_dataset("AksaraLLM/aksara-pretrain-id", split="validation")
ds = concatenate_datasets([tr, vl])
print(f"  raw rows: {len(ds)}")

stats = {"raw_rows": len(ds)}

# 2. NusaX prefix strip
print("[2/7] NusaX prefix strip…")
prefix_re = re.compile(r"^\[Bahasa [^\]]+\]\s*")
def strip_prefix(ex):
    t = ex["text"]
    if (ex["source"] or "").startswith("nusax-"):
        t = prefix_re.sub("", t)
    return {"text": t, "source": ex["source"]}
ds = ds.map(strip_prefix, num_proc=4, desc="strip")

# 3. exact dedup
print("[3/7] exact dedup…")
seen, keep = set(), []
for i, t in enumerate(ds["text"]):
    h = hashlib.md5((t or "").encode("utf-8", "ignore")).hexdigest()
    if h not in seen:
        seen.add(h); keep.append(i)
ds = ds.select(keep)
stats["after_exact_dedup"] = len(ds)
print(f"  kept {len(ds)}")

# 4. Gopher filter
print("[4/7] Gopher filter…")
before = len(ds)
ds = ds.filter(gopher_ok, input_columns=["text"], num_proc=4)
stats["after_gopher"] = len(ds)
print(f"  kept {len(ds)}/{before}")

# 5. GlotLID
print("[5/7] GlotLID…")
ds = ds.map(classify_lang_batch, batched=True, batch_size=512, num_proc=4, desc="glotlid")

before = len(ds)
ds = ds.filter(lang_ok, num_proc=4)
stats["after_langid"] = len(ds)
print(f"  kept {len(ds)}/{before}")
lbl_dist = Counter(ds["lid_label"])
stats["lang_dist_after"] = lbl_dist.most_common(20)
print("  lang dist:", lbl_dist.most_common(10))

# 6. hash-split
print("[6/7] hash-split…")
def is_val_text(t):
    return hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()[0] == "0"

ds = ds.map(lambda ex: {"_split": "validation" if is_val_text(ex["text"]) else "train"}, num_proc=4)
train = ds.filter(lambda ex: ex["_split"] == "train", num_proc=4).remove_columns(["_split", "lid_label", "lid_prob"])
val = ds.filter(lambda ex: ex["_split"] == "validation", num_proc=4).remove_columns(["_split", "lid_label", "lid_prob"])

stats["final_train"] = len(train)
stats["final_val"] = len(val)
print(f"  train={len(train)} val={len(val)}")

# 7. save
print("[7/7] saving…")
train.to_parquet("out/clean/train.parquet")
val.to_parquet("out/clean/validation.parquet")

tc = sum(len(t) for t in train["text"])
vc = sum(len(t) for t in val["text"])
stats["final_chars_train"] = tc
stats["final_chars_val"] = vc
stats["est_tokens_train"] = tc // 3
stats["est_tokens_val"] = vc // 3
stats["source_train"] = Counter(train["source"]).most_common()
stats["source_val"] = Counter(val["source"]).most_common()

with open("out/clean/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("done.")
print(json.dumps({k:v for k,v in stats.items() if not isinstance(v, list)}, indent=2))

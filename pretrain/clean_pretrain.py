"""Fast clean pretrain: skip MinHash (documented as v2 work).

Steps 1-5 + 7-9 from clean_pretrain.py, but without expensive near-dup.
This produces v1 with exact-dedup + lang-filter + gopher-filter + proper split.
"""
import os, re, json, hashlib
from collections import Counter
from datasets import load_dataset, Dataset, concatenate_datasets
import fasttext

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
nav_re = re.compile(r"(Log in|Sign up|Sign in|Cookies|Terms of Service|Privacy Policy|Daftar isi|Halaman utama)", re.I)
url_re = re.compile(r"https?://([^/\s)]+)")
BLOCK = {
    "arenasbo88.com", "malehealthcenter.com", "wearebrewstuds.com",
    "hargano.com", "gpgo.in", "159.65.11.81",
    "sbobet.com", "sbobet88.com", "sbobet365.com", "bola88.com",
    "togel.com", "hongkongpools.com", "prediksisgp.com",
}
def gopher_ok(t):
    if not t or len(t) < 80: return False
    lines = t.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty: return False
    avg_w = sum(len(l.split()) for l in non_empty) / len(non_empty)
    if avg_w < 3: return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha / len(t) < 0.65: return False
    ell = sum(1 for l in non_empty if l.rstrip().endswith("…") or l.rstrip().endswith("..."))
    if ell / len(non_empty) > 0.3: return False
    for dom in url_re.findall(t):
        d = dom.lower().lstrip("www.")
        if d in BLOCK: return False
    nh = len(nav_re.findall(t))
    if nh > 5 and nh / max(len(non_empty), 1) > 0.05: return False
    return True

before = len(ds)
ds = ds.filter(gopher_ok, input_columns=["text"], num_proc=4)
stats["after_gopher"] = len(ds)
print(f"  kept {len(ds)}/{before}")

# 5. GlotLID
print("[5/7] GlotLID…")
lid = fasttext.load_model("glotlid.bin")
def classify(batch):
    preds, probs = [], []
    for t in batch["text"]:
        x = (t or "").replace("\n", " ").replace("\r", " ")[:2000].strip()
        if not x:
            preds.append("none"); probs.append(0.0); continue
        lbl, p = lid.predict(x, k=1)
        preds.append(lbl[0].replace("__label__", ""))
        probs.append(float(p[0]))
    return {"lid_label": preds, "lid_prob": probs}
ds = ds.map(classify, batched=True, batch_size=512, num_proc=4, desc="glotlid")

def lang_ok(ex):
    s = ex["source"] or ""
    lbl = ex["lid_label"]; p = ex["lid_prob"]
    if s.startswith("nusax-"): return True
    if lbl == "ind_Latn" and p >= 0.60: return True
    return False

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

---
language:
  - id
size_categories:
  - 1M<n<10M
task_categories:
  - text-generation
  - fill-mask
license: cc-by-sa-4.0
tags:
  - indonesian
  - pretrain
  - wikipedia
  - culturax
  - clean
---

# AksaraLLM Pretrain Clean v1

**Versi clean dari [`AksaraLLM/aksara-pretrain-id`](https://huggingface.co/datasets/AksaraLLM/aksara-pretrain-id), ditambah Wikipedia-id (Nov 2023 fresh dump).**

## Changes dari aksara-pretrain-id v4

| Fix | Before | After |
|---|---|---|
| Exact duplicate rows | 48,009 (5.84%) | 0 |
| Train/val leakage | 11.76% | 0% (hash-based split) |
| NusaX `[Bahasa X]` prefix | 5,388 rows contaminated | 0 (stripped) |
| Malay rows labeled as Indonesian | ~25% | ≤ 5% (GlotLID P≥0.60 filter) |
| Gopher-style quality filter | No | Yes |
| URL blocklist (judi/spam) | No | Yes |
| Fresh Wikipedia-id 2023-11-01 dump | No | Yes (~665k articles) |

## Schema

```
{
  "text": string,       # document content
  "source": string      # one of: wikipedia-id, culturax-id, wikipedia-indonesia-topik,
                        # wikipedia-id-20231101, nusax-{ace,ban,bbc,bjn,bug,ind,jav,mad,min,nij,sun}
}
```

## Methodology

1. Load train + validation dari `aksara-pretrain-id`.
2. Strip `[Bahasa X]` prefix dari rows NusaX.
3. Exact MD5 dedup.
4. Gopher-style filters:
   - Min 80 chars, avg words per line ≥ 3
   - Alphabetic char fraction ≥ 0.65
   - Fraction of lines ending with "…" ≤ 0.30
   - Nav boilerplate density ≤ 5%
   - URL blocklist (hand-curated: gambling, spam domains)
5. **GlotLID** language classifier (`cis-lmu/glotlid`):
   - For id-partition sources: keep only `ind_Latn` with P ≥ 0.60
   - For NusaX sources: keep all (FT may not support those labels)
6. **MinHash** near-dedup (Jaccard threshold 0.85, num_perm=128, 5-gram shingles).
7. Merge fresh Wikipedia-id (wikimedia/wikipedia 20231101.id), deduped vs existing.
8. Deterministic hash-based train/val split (last hex = "0" → val, no leakage possible).

## Stats (v1.0 — April 2026)

| Stage | Rows |
|---|---|
| Raw (`aksara-pretrain-id` train + val) | 839,366 |
| After exact MD5 dedup | 789,368 |
| After Gopher quality filter | 770,310 |
| After GlotLID language filter | 758,642 |
| **+ Fresh Wikipedia-id (2023-11-01, deduped)** | +104,473 |
| **Final total** | **863,115** |
| → train split | 808,886 |
| → validation split | 54,229 |

**Estimated tokens** (chars/3 heuristic):
- Train: ~500 M
- Validation: ~34 M

**Language distribution (after filter):**
- `ind_Latn`: 99.5% (754,973 / 758,642)
- `jav_Latn`, `sun_Latn`, `min_Latn`, `ban_Latn`, `bjn_Latn`, `ace_Latn`, `bug_Latn`, `mad_Latn`, `bbc_Latn`: ~350–400 each (= NusaX sentiment subsets, kept despite low confidence because source label is trusted)

## Split

| Split | Strategy |
|---|---|
| train | MD5(text)[0] != "0" |
| validation | MD5(text)[0] == "0" (~6% of data) |

Karena deterministic by text hash, **tidak mungkin ada leakage** saat dataset di-update di masa depan: document yang sama akan selalu masuk ke split yang sama.

## Licensing

**IMPORTANT:** Dataset ini adalah gabungan dari multiple sources dengan license yang berbeda:

| Source | License |
|---|---|
| `wikipedia-id` + `wikipedia-id-20231101` + `wikipedia-indonesia-topik` | CC-BY-SA 4.0 |
| `culturax-id` | ODC-BY (dari OSCAR/mC4) |
| NusaX subsets | CC-BY-SA 4.0 |

Secara keseluruhan, license paling ketat adalah **CC-BY-SA 4.0**, dan dataset ini dirilis under that license. Pastikan kamu mematuhi terms untuk hilirnya.

## Reproduce

Lihat `scripts/clean_pretrain.py` di repo `AksaraLLM/aksara-data` (cabang `clean-v1`).

```bash
pip install datasets fasttext-wheel datasketch "numpy<2.0"
wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin -O glotlid.bin
python scripts/clean_pretrain.py
```

## Citation

Kalau pakai dataset ini, mohon cite:

```
@misc{aksarallm_pretrain_clean_v1,
  author       = {AksaraLLM Community},
  title        = {AksaraLLM Pretrain Clean v1},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/AksaraLLM/aksara-pretrain-clean-v1}}
}
```

Plus upstream sources:
- Nguyen et al. 2023 (CulturaX)
- Winata et al. 2023 (NusaX)
- Wikimedia Foundation (Wikipedia)

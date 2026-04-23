# Pretrain Data Pipeline (v1)

Cleaning pipeline dan sumber pretraining baru untuk AksaraLLM. Produces three HuggingFace datasets:

| Dataset | Rows | Size | Description |
|---|---|---|---|
| [`AksaraLLM/aksara-pretrain-clean-v1`](https://huggingface.co/datasets/AksaraLLM/aksara-pretrain-clean-v1) | 863,115 | ~1.5 GB | Cleaned & merged Indonesian pretraining corpus |
| [`AksaraLLM/aksara-sft-clean-v1`](https://huggingface.co/datasets/AksaraLLM/aksara-sft-clean-v1) | 33,593 | 8 MB | De-duplicated, hallucination-free SFT data |
| [`AksaraLLM/aksara-bahasa-daerah-v1`](https://huggingface.co/datasets/AksaraLLM/aksara-bahasa-daerah-v1) | 424,658 | 79 MB | 8 bahasa daerah (jv/su/min/ace/bug/ban/bjn/mad) |

## Motivation

Data audit of existing datasets (`aksara-pretrain-id`, `aksara-sft-id`, `aksara-training-data`) revealed:

- **11.76% train/val leakage** in pretrain (all reported validation metrics overestimated)
- **~25% Malay contamination** mislabeled as Indonesian
- **NusaX `[Bahasa X]` prefix** leaked 100% of rows (model learning to emit metadata as tokens)
- **5.84% cross-source exact duplicates**
- **64% exact dupes** in `distill_v4.json`
- **~2,900 hallucinated SFT rows** in `mega_dataset_v5` and `mega_distill_progress` (non-words like `jenjib`, `era-kerae`, `upayawan`)
- 1 row teaching factual error (Bangkok as largest SEA city)

See [`docs/AKSARALLM_DATA_AUDIT.md`](../docs/AKSARALLM_DATA_AUDIT.md) for full scorecard + reproducible findings.

## Pipeline

```
1. clean_pretrain.py       → clean aksara-pretrain-id
2. fetch_extras.py          → add fresh Wikipedia-id 2023-11-01
3. fetch_bahasa_daerah.py   → 8 regional Wikipedia dumps
4. clean_sft.py             → clean aksara-sft-id + identity pack
5. merge_and_push.py        → dedup across sources, hash-split
6. push_to_hf.py            → upload to HuggingFace
```

### Steps applied in `clean_pretrain.py`

1. **NusaX prefix strip** — remove leaked `[Bahasa X]` metadata from 5,387 rows
2. **Exact MD5 dedup** — catches all byte-identical rows
3. **Gopher-style quality filter** — min length 80, min 3 words/line, alpha ratio ≥ 0.65, ellipsis ratio ≤ 30%, nav-boilerplate ratio ≤ 5%, URL blocklist (gambling/spam domains)
4. **GlotLID language filter** — keep `ind_Latn` with P ≥ 0.60, trust NusaX sources for bahasa daerah
5. **Deterministic hash-split** — MD5(text)[0]=="0" → val (zero-leakage guarantee for future updates)

Not included in v1 (deferred to v1.1): MinHash near-dedup (already pipelined separately, takes longer).

### Stats

```
Raw:        839,366
→ dedup:    789,368  (-50,000 exact dups)
→ gopher:   770,310  (-19,058 low-quality/boilerplate)
→ langid:   758,642  (-11,668 wrong-language)
+ fresh wiki-id 2023-11-01: +104,473 unique
Final:      863,115  (808,886 train / 54,229 val)
≈ 500M pretraining tokens (chars/3 heuristic)
```

## Running

```bash
pip install -r requirements.txt
# HF_TOKEN required for push step only
export HF_TOKEN=hf_xxxxx

# Download GlotLID model (~1.7 GB) first
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('cis-lmu/glotlid', 'model_v3.bin', local_dir='.', local_dir_use_symlinks=False)"
mv model_v3.bin glotlid.bin

python pretrain/clean_pretrain.py         # ~15 min on 4 CPU
python pretrain/fetch_extras.py           # ~3 min
python pretrain/fetch_bahasa_daerah.py    # ~5 min
python pretrain/clean_sft.py              # ~1 min
python pretrain/merge_and_push.py         # ~2 min
python pretrain/push_to_hf.py             # ~1 min (depends on bandwidth)
```

## License

Same as parent repo: Apache 2.0. Derived datasets inherit original licenses (mostly CC-BY-SA 4.0 from Wikipedia & CulturaX).

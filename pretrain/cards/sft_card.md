---
language:
  - id
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
license: cc-by-sa-4.0
tags:
  - indonesian
  - sft
  - instruction-tuning
---

# AksaraLLM SFT Clean v1

**Versi clean dari [`AksaraLLM/aksara-sft-id`](https://huggingface.co/datasets/AksaraLLM/aksara-sft-id), dengan distill_v4 identity pack yang sudah dededup.**

## Changes dari aksara-sft-id v5

| Fix | Before | After |
|---|---|---|
| Factual error rows (e.g. "Bangkok is SEA largest city") | present | removed |
| Truncated / ellipsis-cut outputs | present | filtered |
| Exact pair duplicates | 7 | 0 |
| Identity pack from distill_v4 | 542 rows, 64% dup | 211 deduped unique items |

## What's INCLUDED

| Source | Rows (train) | Task type |
|---|---|---|
| synthetic-wiki-qa | ~17,000 | qa_templated |
| tydiqa-id | ~4,900 | qa |
| indoqa | ~2,800 | qa_knowledge |
| nusax (11 bahasa daerah) | ~4,700 | sentiment_bahasa_daerah |
| aya-human | ~650 | general |
| aksarallm-provinsi / kerajaan / agama / pahlawan / etc. | ~370 | budaya/sejarah/... |
| aksarallm-identity (merged from distill_v4 + sft_v5) | 211 | identity |

## What's EXPLICITLY EXCLUDED

- **`mega_dataset_v5.json`** (1,675 items): contains hallucinated output from a weak distiller ("jenjib", "era-kerae", "upayawan"). **Do not use for SFT.**
- **`mega_distill_progress.json`** (1,222 items): same issue.
- Rows with `"Nama dan lokasinya adalah Bangkok, Thailand"` in output (factually wrong).
- Rows with truncated outputs ending in `...` under 200 chars.

## Schema

```
{
  "instruction": string,
  "output": string,
  "source": string,
  "task_type": string
}
```

## Split

Hash-based deterministic: MD5(instruction||output)[0] == "0" → val (~6%), else train.

## Recommended next steps (v2)

Berikut task yang belum tercover dan perlu ditambah:

- **Reasoning/CoT** — 3–5k rows
- **Math** — 2–5k rows (currently 10)
- **Code** — 2–5k rows (currently 0)
- **Multi-turn dialog** — 3–10k rows (currently 10)
- **Summarization** — 2k rows
- **Translation ID↔EN** — 5k rows (Aya English subset)
- **Translation ID↔bahasa daerah** — 1–2k per language
- **Safety refusal** — 1–2k rows (currently 2)
- **Creative writing** (puisi, esai, email) — 2k rows

## License

CC-BY-SA 4.0 (mengikuti upstream paling ketat).

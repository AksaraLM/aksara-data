---
language:
  - jv
  - su
  - min
  - ace
  - bug
  - ban
  - bjn
  - mad
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
  - fill-mask
license: cc-by-sa-4.0
tags:
  - indonesian
  - regional-language
  - nusantara
  - wikipedia
configs:
  - config_name: jv
    data_files: jv.parquet
  - config_name: su
    data_files: su.parquet
  - config_name: min
    data_files: min.parquet
  - config_name: ace
    data_files: ace.parquet
  - config_name: bug
    data_files: bug.parquet
  - config_name: ban
    data_files: ban.parquet
  - config_name: bjn
    data_files: bjn.parquet
  - config_name: mad
    data_files: mad.parquet
---

# AksaraLLM Bahasa Daerah v1

Korpus pretraining untuk **8 bahasa daerah Indonesia**, dihimpun dari Wikipedia snapshot Nov 2023.

| Bahasa | ISO code | Rows (articles) | Notes |
|---|---|---|---|
| Bahasa Jawa | jv | 73,380 | Substantial |
| Bahasa Sunda | su | 61,555 | Substantial |
| Bahasa Minangkabau | min | 227,143 | Largest; ⚠️ contains many bot-generated stub articles |
| Bahasa Aceh | ace | 13,003 | Moderate |
| Bahasa Bugis | bug | 15,880 | Moderate |
| Bahasa Bali | ban | 20,986 | Moderate |
| Bahasa Banjar | bjn | 10,519 | Small |
| Bahasa Madura | mad | 1,192 | Small |

## Motivation

`aksara-pretrain-id` original hanya punya **5,388 rows × ~170 chars ≈ 900 KB** total untuk 11 bahasa daerah (dari NusaX sentiment dataset, yang sebenarnya customer reviews, bukan pretraining corpus, dengan prefix `[Bahasa X]` literal bocor).

Dataset ini menggantikan itu dengan:
- **~424k articles** dari Wikipedia per bahasa (bukan review).
- **Domain diversity** (sejarah, budaya, tokoh, tempat, konsep).
- **Format pretraining yang benar**: no metadata prefix, no classification format.

## ⚠️ Caveats

- **Wiki-min** (Minangkabau Wikipedia) banyak mengandung **bot-generated stub articles** tentang spesies biologi, dll — **bukan asli dari penutur**. Untuk training, disarankan filter yang agresif (misal buang artikel < 300 kata atau yang mengandung pola `"adalah sebuah spesies ..."`).
- **Bahasa Batak Toba (bbc)** dan **Ngaju (nij)** belum termasuk — Wikipedia untuk bahasa tersebut tidak tersedia di snapshot 20231101. Untuk v2, pertimbangkan sumber alternatif (NusaCrowd).
- Beberapa artikel mungkin code-switch ke Indonesian di section tertentu.

## Schema

```
{
  "text": string,
  "source": string   # e.g. "wikipedia-jv"
}
```

## Usage untuk pretraining

```python
from datasets import load_dataset, concatenate_datasets

# Load per language
jv = load_dataset("AksaraLLM/aksara-bahasa-daerah-v1", "jv", split="train")
su = load_dataset("AksaraLLM/aksara-bahasa-daerah-v1", "su", split="train")
# ... combine, interleave with Indonesian pretrain corpus
```

Recommended sampling strategy saat pretraining:
- Indonesian: 80-85% of tokens
- Bahasa daerah (gabungan 8 bahasa): 10-15%
- English code/docs: 5% (optional)

## License

CC-BY-SA 4.0 (Wikipedia).

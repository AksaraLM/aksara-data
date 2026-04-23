"""Fetch additional bahasa daerah pretraining data from public HF Wikipedia dumps.

Languages covered:
  id  Indonesian      (reference / baseline)
  jv  Javanese
  su  Sundanese
  min Minangkabau
  ms  Malay            (useful contrast — for mixed ID/MS detection)
  ace Acehnese         (Wikipedia available, small)
  bug Buginese         (Wikipedia available, tiny)
  ban Balinese         (Wikipedia available, small)
  bjn Banjar           (Wikipedia available, small)
  mad Madurese         (Wikipedia available, tiny)

Note: wikipedia 'bbc' (Batak Toba) and 'nij' (Ngaju) may not be on wikimedia/wikipedia;
will fallback to alternatives where possible.

Output per language:
  out/bahasa_daerah/<lang>.parquet
"""
import os
from datasets import load_dataset

os.makedirs("out/bahasa_daerah", exist_ok=True)

# wikimedia/wikipedia uses 'YYYYMMDD.LANG' config names with snapshots
# Use 20231101 snapshot (broadly available)
LANGS = {
    "jv": "jav_Latn",
    "su": "sun_Latn",
    "min": "min_Latn",
    "ace": "ace_Latn",
    "bug": "bug_Latn",
    "ban": "ban_Latn",
    "bjn": "bjn_Latn",
    "mad": "mad_Latn",  # may not exist
}

SNAPSHOT = "20231101"

for code, glotlid_label in LANGS.items():
    try:
        ds = load_dataset("wikimedia/wikipedia", f"{SNAPSHOT}.{code}", split="train")
        n = len(ds)
        print(f"[wiki/{code}] {n} rows")
        ds = ds.map(lambda ex: {"text": ex["text"], "source": f"wikipedia-{code}", "title": ex["title"]},
                    remove_columns=[c for c in ds.column_names if c not in ("text", "title")])
        ds.to_parquet(f"out/bahasa_daerah/{code}.parquet")
        print(f"  saved out/bahasa_daerah/{code}.parquet")
    except Exception as e:
        print(f"[wiki/{code}] ERROR: {type(e).__name__}: {str(e)[:200]}")

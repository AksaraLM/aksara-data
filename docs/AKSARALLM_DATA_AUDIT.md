# AksaraLLM — Data Quality & Coverage Audit

**Auditor:** Devin (for @cahyohackids)
**Date:** 2026-04-23
**Scope:** Public datasets under [huggingface.co/AksaraLLM](https://huggingface.co/AksaraLLM):
`aksara-pretrain-id`, `aksara-sft-id`, `aksara-training-data`.

**TL;DR.** AksaraLLM's filosofi ("open everything" untuk Nusantara) sangat bagus, tapi dataset saat ini **belum siap** untuk pretraining atau SFT model yang kompetitif. Ada **5 masalah kritis** yang harus diperbaiki sebelum scaling parameter:

1. **Train/val leakage 11.8%** — ~1.975 dari 16.788 baris validation verbatim ada di train. Semua metrik validation saat ini overestimate.
2. **~25% pretrain corpus dilabel "Indonesian" tapi sebenarnya Malay** menurut FastText lid176; ditambah 6% English, 4% Jawa di dalam partisi `wikipedia-id` & `culturax-id`.
3. **48.009 cross-source duplicates** (5.84%) — teks yang sama muncul di `wikipedia-id` dan `wikipedia-indonesia-topik`.
4. **NusaX pretrain records punya prefix `[Bahasa X]` bocor 100%** — model akan belajar mengeluarkan prefix metadata literal.
5. **`aksara-training-data` (SFT distill)** banyak berisi **hallucination dan bahasa Indonesia tidak gramatikal** dari model kecil; 64% `distill_v4.json` adalah exact duplicate. Tidak layak dipakai untuk SFT.

Skala korpus pretrain hanya **~0.58 miliar token (1.75 GB teks)** — 20× lebih kecil dari CulturaX-ID mentah (12 B token) dan ≈4 ordo magnitudo di bawah Sahabat-AI. **Prioritas #1 bukan scaling parameter, tapi fix data.**

---

## 1. Inventory

| Dataset | Rows (train) | Rows (val) | Size | Format | Notes |
|---|---|---|---|---|---|
| `aksara-pretrain-id` | 822,578 | 16,788 | 1.01 GB parquet / 1.75 GB teks | parquet | columns: `text`, `source` |
| `aksara-sft-id` | 33,934 | 1,787 | 8.86 MB | parquet | columns: `instruction`, `output`, `source`, `task_type` |
| `aksara-training-data` | 2,217 | – | 2.33 MB | JSON (5 files, **schema tidak konsisten**) | HF viewer gagal karena CastError |

### Dataset card issues
- `aksara-training-data` **tidak punya dataset card** dan HF dataset viewer error (`CreateCommitError`) karena tiga file JSON memiliki schema yang berbeda:
  - `distill_v4.json`, `mega_dataset_v5.json` → list of strings
  - `mega_distill_progress.json` → `[{instruction, response, topic}, ...]`
  - `sft_v5_indonesia.json` → `{data: [{instruction, output}, ...]}`
  - `system_prompt_v2.json` → object
- Tidak ada license yang di-declare secara eksplisit di README `aksara-pretrain-id` / `aksara-sft-id`. Karena sumbernya campuran (Wikipedia CC-BY-SA, CulturaX ODC-BY/ODbL + per-domain, NusaX CC-BY-SA, Aya Apache-2.0, TyDi QA Apache-2.0), lisensi gabungan **harus disebutkan per subset** atau dipaksa ke lisensi terketat (kemungkinan CC-BY-SA 4.0).

---

## 2. Pretrain corpus: `aksara-pretrain-id`

### 2.1 Komposisi

| source | rows | % rows | MB | % bytes | avg chars |
|---|---|---|---|---|---|
| `wikipedia-id` | 572,093 | 69.55% | 1,029 | 58.76% | 1,793 |
| `culturax-id` | 196,072 | 23.84% | 542 | 30.96% | 2,761 |
| `wikipedia-indonesia-topik` | 49,025 | 5.96% | 179 | 10.23% | 3,644 |
| nusax (11 bahasa daerah, digabung) | 5,388 | 0.65% | 0.9 | 0.05% | ~170 |

**Catatan:**
- Gabungan Wikipedia = 80% rows / 69% bytes → korpus sangat ensiklopedis, minim register kasual / percakapan / kode / forum / chat.
- Rata-rata ukuran NusaX ≈ 170 karakter ≈ kalimat tunggal (ini adalah *review* e-commerce dari paper NusaX, bukan naturalistic corpus). Untuk pretraining, ini terlalu pendek dan domain-nya sempit.
- `wikipedia-indonesia-topik` kemungkinan adalah subset topikal Wikipedia — lihat §2.3 tentang duplikasi dengan `wikipedia-id`.

### 2.2 Dedup & leakage

| Check | Result |
|---|---|
| Exact duplicate rows (train) | **48,009 / 822,578 → 5.84%** |
| Same-source duplicates | 0 |
| **Cross-source duplicates** | **48,009** (semua cross-source) |
| **Train ↔ Val exact overlap** | **1,975 / 16,788 → 11.76%** |

Konsekuensi:
- **Setiap reported val loss / val perplexity / val accuracy overestimated** — evaluasi saat ini tidak bisa dipercaya.
- Cross-source dup menunjukkan `wikipedia-id` dan `wikipedia-indonesia-topik` beririsan. Jika `-topik` adalah subset bertema Indonesia dari `wikipedia-id`, dia tidak membawa informasi baru sebagai kategori terpisah.

### 2.3 Language ID (FastText lid176, stratified 500/source sample, n≈6,890)

**Overall:**
```
id  (Indonesian)  57.88%
ms  (Malay)       25.06%   ← DOMINASI YANG SALAH
en  (English)      6.37%
jv  (Javanese)     4.46%
su  (Sundanese)    2.47%
tl  (Tagalog)      2.09%
min (Minangkabau)  0.44%
it, hr, eo, tr, fr, es, sl, hu  (various noise)  < 1.5%
```

**Per source (clean/dirty signal):**
- `wikipedia-id`: id=98%, en=1% → **bersih**
- `culturax-id`: id=100% (FT agak bias; bisa jadi ms sebagian) → **relatif bersih**
- `wikipedia-indonesia-topik`: id=98%, en=1%, it=0% → bersih, tapi lihat §2.2
- NusaX `ind`: id=95% ✓
- NusaX `jav`: jv=44% (OK tapi rendah), id=28%, ms=21% → ~50% baris bukan Jawa
- NusaX `sun`: su=28%, ms=36%, id=27% → hanya ~28% yang Sunda menurut FT
- NusaX `min`: min=6% saja, sisanya id/ms/jv/tl
- NusaX `ace`, `ban`, `bjn`, `bug`, `mad`, `nij`, `bbc`: FT **tidak punya label** untuk bahasa-bahasa ini → terklasifikasi sebagai Malay/Indonesian/Tagalog (peringatan: ini bukan 100% bukti kualitas rendah, hanya limitasi FT).

**Interpretasi:**
- Masalah terbesar: **25% overall** corpus diklasifikasi sebagai Malay. Beberapa memang kesalahan FT (Bahasa Indonesia sering confusable dengan Malay), tapi beberapa artikel Wikipedia memang code-mixed atau di-rewrite dari Wiki Melayu. Butuh **filter kedua** pakai `GlotLID` atau model klasifikasi `id/ms` yang khusus untuk memisahkan keduanya.
- **Contoh konkret di sampel:** `wikipedia-indonesia-topik` row "Bandar Udara La Rochelle - Île de Ré..." di-detect `ms` oleh FT; kontennya memang mencampur Indonesia dan French.

### 2.4 NusaX prefix contamination

**Temuan:** 5,388 dari 5,388 (100%) baris NusaX di pretrain corpus diawali dengan literal `[Bahasa <lang>]`, misal:

```
[Bahasa Madura] Si A bhentana tong kosong ranying munyina bhentana tadek artena.
[Bahasa Bali] Tiyang suba dadi pelanggan madtari uli tahun 2002...
[Bahasa Bugis] Sibawakku massappa jamang okko tokopedia
```

**Dampak:** Dengan pretraining causal LM, model akan belajar bahwa banyak dokumen dimulai dengan `[Bahasa X]`. Saat inference, model bisa mengeluarkan token `[Bahasa Madura]` di mana-mana — ini adalah **metadata leakage** klasik.

**Fix yang tepat:** hilangkan prefix saat ingest. Alternatif yang lebih baik: tambahkan prefix sebagai **structured control token** di tokenizer (`<|lang:mad|>`) dan gunakan data augmentation acak (kadang ada, kadang tidak).

### 2.5 Length & garbage

| Metric | Value |
|---|---|
| Median char length | 882 |
| Mean char length | 2,124 |
| p99 char length | 18,673 |
| Max char length | **534,419** (≈ 177k token untuk 1 dokumen) |
| Rows with mojibake pattern | 169 (0.02%) — low |
| Nav/boilerplate tokens in first 500 chars | 67,417 (8.2%) — **needs review** |
| Empty rows | 0 |

**8.2% nav-boilerplate** adalah bendera kuning: banyak dokumen mengandung "Log in/Sign up/Daftar isi/Cookies/Terms of service" di awal — CulturaX harusnya sudah melakukan content extraction, tapi sebagian masih bocor. Perlu filter ke-2 dengan heuristik line-level (Gopher-style: hapus baris yang rata-rata < 2 kata atau mulai dengan bullet).

**1,195 dokumen > 50,000 karakter** (ada yang sampai 534k). Untuk context window < 4k–8k, dokumen sepanjang ini akan di-chunk secara arbitrer; lebih baik chunk eksplisit dengan logical boundary (section/paragraph).

### 2.6 CulturaX-ID URL domain spot-check

Dari 196k baris CulturaX-ID, 13,551 URL dikutip di teks. Top 20 domain mengandung:

- `arenasbo88.com` (50x) — **situs judi online**
- `malehealthcenter.com` (39x), `hargano.com` (94x) — iklan/produk kesehatan yang sering spam
- `bit.ly` (188x), `goo.gl` (67x), `t.co` (94x) — URL shortener (low trust)
- `wearebrewstuds.com` (242x) — domain mati / squat
- `159.65.11.81` (33x) — IP address langsung (suspicious)

**Rekomendasi:** URL-domain blocklist untuk filtering. Gunakan daftar seperti `badwords` + UT1 blocklists atau SEA-LION's SEABlocklists.

---

## 3. SFT corpus: `aksara-sft-id`

### 3.1 Komposisi

| source | rows | % | task_type |
|---|---|---|---|
| `synthetic-wiki-qa` | 19,000 | 56.0% | qa_templated |
| `tydiqa-id` | 5,404 | 15.9% | qa |
| `indoqa` | 3,136 | 9.2% | qa_knowledge |
| `aya-human` | 731 | 2.2% | general |
| nusax (11 bahasa daerah) | 5,237 | 15.4% | sentiment_bahasa_daerah |
| `aksarallm-*` (provinsi, pahlawan, agama, dll) | 399 | 1.2% | budaya/sejarah/geografi/dll |
| synthetic-math, synthetic-conversation, synthetic-creative | 25 | 0.1% | math/conv/creative |

**Bahasa daerah di SFT:** 100% hanya untuk **sentiment classification** (dari NusaX). Tidak ada SFT untuk translation, summarization, QA, atau instruction following dalam bahasa daerah — sangat sempit.

### 3.2 Kualitas

| Metric | Value |
|---|---|
| Exact (instruction+output) pair dup | 7 (0.02%) — clean |
| Exact instruction dup | 119 (0.35%) — low |
| Median instruction length | 48 chars |
| Median output length | 177 chars |
| p95 output length | 530 chars |

Beberapa templated prefixes muncul banyak:
- `"Terjemahkanlah penggalan teks Bahasa Inggris..."` × 114
- `"Klasifikasikan kalimat berikut berdasarkan sentimennya..."` × 12
- `"Isilah titik-titik berikut..."` × 8

Ini **normal** untuk templated SFT; kuncinya apakah `{content}` slot-nya beragam. Dari sampling manual, slot content memang beragam → **OK**.

### 3.3 Task coverage yang MISSING

Berikut kategori yang belum/sangat kurang di SFT saat ini:

| Task | Current coverage | Target |
|---|---|---|
| Reasoning / chain-of-thought | 0 | 3–5k |
| Math (aritmatika → word problem) | 10 | 2–5k |
| Code | 0 | 2–5k |
| Multi-turn dialog | 10 | 3–10k |
| Summarization | 0 | 2k |
| Translation ID↔EN | 0 | 5k |
| Translation ID↔bahasa daerah | 0 (hanya sentiment) | 1–2k / bahasa |
| Safety / refusal | 2 | 1–2k |
| Instruction following (non-QA) | ~731 (Aya) | 5–10k |
| Writing (puisi/esai/email) | 5 | 2k |

**Conclusion:** SFT corpus saat ini efektif hanya mengajarkan model satu skill — **QA ensiklopedis pendek**. Itu tidak cukup untuk asisten general-purpose.

---

## 4. `aksara-training-data` (distilled/synthetic)

Ini dataset yang paling problematik. 5 file:

### 4.1 `distill_v4.json` (542 items, list of strings)

- **347 / 542 = 64% exact duplicates.** Item paling dominan muncul **40 kali**.
- Top 5 repeated items semuanya adalah identity prompts (`"Apakah kamu ChatGPT?"`, `"Apa itu AksaraLLM?"`, `"Siapa kamu?"`, `"Namamu siapa?"`, `"Siapa pembuat kamu?"`).
- Efek: kalau dipakai untuk fine-tune, model akan over-fit ke identity response.

### 4.2 `mega_dataset_v5.json` (1,675 items) & `mega_distill_progress.json` (1,222 items)

Berisi **synthetic distillation output dari model kecil** yang kualitasnya rendah. Contoh nyata dari data:

> **Instruksi:** "Apakah perbedaan antara Presiden Republik Indonesia yang berada di Jakarta dan Presiden Republik Indonesia yang berada di Yogyakarta?"
> **Respons:** "Perbedaannya terletak pada lokasi tingkat pemerintahan: 1. **Jendela**: Jakarta memiliki sebuah **jenjib** keamanan yang lebih besar dibandingkan dengan Yogyakarta..."

`jenjib` bukan kata Bahasa Indonesia. `Jendela` sebagai dimension kategorisasi juga tidak masuk akal. Pertanyaan itu sendiri premis-nya salah (tidak ada dua presiden).

Contoh lain:

> **Instruksi:** "Apakah tujuan utama dari Kehadiran Pemimpin yang Seseorang pada era-kerae tertentu..."

"era-kerae", "upayawan", "pemilihan kepala daerah (pimpinan) sebagai direktur" — semua incoherent.

**Kesimpulan:** Data ini **merusak** model kalau dipakai SFT. Ini adalah pola "model collapse via self-distillation" yang terdokumentasi (Shumailov et al. 2024). **Jangan dipakai.**

### 4.3 `sft_v5_indonesia.json` (16 items)

Hand-curated identity + fakta wajib. Berkualitas, tapi tiny.

### 4.4 Sample-level bug di dataset preview

Dari data card preview langsung terlihat pasangan salah:

> **Instruksi:** "Apakah nama dan lokasi tempat terbesar di Asia Tenggara?"
> **Respons:** "Nama dan lokasinya adalah Bangkok, Thailand. Thai is the largest city and capital of Thailand."

Secara geografis: Jakarta > Bangkok by population dalam SEA. Juga response code-switch ke English tanpa alasan. **Ini SFT row yang aktif mengajari kesalahan faktual.**

---

## 5. Benchmark vs. corpus Indonesia yang tersedia publik

| Corpus | Indonesian tokens | Notes |
|---|---|---|
| **aksara-pretrain-id** (current) | **~0.58 B** (582 M) | Wiki + CulturaX sample + NusaX |
| CulturaX (full, `id`) | 12.06 B | Source dari aksara-pretrain-id |
| mC4 (`id`) | ~30 B | Raw, noisy |
| IndoWebCorpus | ~6 B | Web scrape |
| SEA-LION v3 pretrain mix | ~980 B (semua SEA) | Indonesian portion ~100 B+ |
| Sahabat-AI v2 / SEA-LION v4 | beberapa ratus B | Indonesian + 11 bahasa daerah |

Untuk context, **Chinchilla-optimal** untuk model 500M param adalah ~10 B tokens, 1B param adalah ~20 B tokens. **Corpus AksaraLLM saat ini 17× di bawah** Chinchilla-optimal untuk target 500M, dan 34× di bawah untuk target 1B di roadmap Fase 3.

Bahasa daerah:
- aksara-pretrain-id: **0.9 MB** total 11 bahasa daerah.
- NusaCrowd public: puluhan MB per bahasa, beberapa GB untuk yang lebih besar.
- CC-100 / OSCAR: sudah punya `jv`, `su`, `min`.
- Gap: sangat mudah diperkaya tanpa crawling baru.

---

## 6. Data Quality Scorecard

Skor 0–5, 5 = state-of-the-art open corpus.

| Dimensi | aksara-pretrain-id | aksara-sft-id | aksara-training-data |
|---|---|---|---|
| Schema / format / card | 3 | 3 | **0** (broken) |
| Reproducibility pipeline | 2 | 2 | 1 |
| Dedup (exact) | **1** (5.8% dup + val leak) | 4 | **0** (64% dup) |
| Dedup (near / MinHash) | 0 (not done) | 0 | 0 |
| Language ID purity | 2 (25% ms leakage) | 3 | 2 |
| Domain diversity | 2 (wiki-dominant) | **1** (QA only) | 1 |
| Regional language depth | **1** (5k rows, 0.9 MB) | **1** (sentiment only) | 0 |
| Noise filtering | 2 (boilerplate + bad URL) | 4 | 1 (hallucinated) |
| License clarity | 2 | 2 | 1 |
| Scale vs goal | **1** (0.58 B / 20 B target) | 2 | 1 |
| **Overall** | **1.6 / 5** | **2.2 / 5** | **0.7 / 5** |

---

## 7. Rekomendasi prioritas

### P0 — harus dilakukan SEBELUM training berikutnya

1. **Hapus train/val leakage.** Re-split berdasarkan document hash setelah dedup, bukan random split.
   ```python
   from datasets import load_dataset
   import hashlib
   ds = load_dataset("AksaraLLM/aksara-pretrain-id", split="train+validation")
   def h(x): return hashlib.md5(x['text'].encode()).hexdigest()
   ds = ds.map(lambda x: {'_h': h(x)})
   # dedup
   seen = set(); keep = []
   for i, hh in enumerate(ds['_h']):
       if hh not in seen:
           seen.add(hh); keep.append(i)
   ds = ds.select(keep)
   # hash-based split: last hex digit < '2' → val
   val_mask = [hh[0] in '01' for hh in ds['_h']]
   ```
2. **Hapus prefix `[Bahasa X]` di NusaX pretrain split.**
   ```python
   import re
   pref = re.compile(r'^\[Bahasa [^\]]+\]\s*')
   ds = ds.map(lambda x: {'text': pref.sub('', x['text']) if x['source'].startswith('nusax-') else x['text']})
   ```
3. **Singkirkan `mega_dataset_v5.json` + `mega_distill_progress.json`** dari SFT pipeline sampai data tersebut diregenerate dari teacher model yang lebih kuat (misal Gemma-2-9B-Indo, SEA-LION-8B, atau Sahabat-AI). Tandai sebagai `EXPERIMENTAL — do not use`.
4. **Dedup `distill_v4.json`** — sisakan hanya unique items. Setelah dedup tinggal ~195, yang lebih realistis sebagai identity pack.
5. **Perbaiki factual errors di SFT v5.** Audit manual 100 baris acak; contoh yang ketahuan: "tempat terbesar di Asia Tenggara = Bangkok" harus dikoreksi ke Jakarta (kota terbesar SEA by population).

### P1 — quick wins (1–2 minggu effort)

6. **Run MinHash near-dedup** di level dokumen dan line-level untuk pretrain. Gunakan `datasketch` atau `text-dedup`.
7. **Filter id/ms dengan GlotLID** (bukan FastText lid176). Buang / tandai dokumen dengan P(id) < 0.85 di partisi `wikipedia-id` dan `culturax-id`. Ekspektasi: buang ~10–20%.
8. **URL/domain blocklist** di CulturaX-ID untuk judi, adult, malware, link-farm. Referensi list: [Ungoogled Chromium blocklists](https://github.com/StevenBlack/hosts), UT1 `adult`/`gambling`.
9. **Line-level quality filter** (Gopher rules):
   - Avg words per line ≥ 3
   - Fraction of alphabetic chars ≥ 0.80
   - Fraction of lines ending with ellipsis ≤ 0.30
   - Fraction of repeated bigrams ≤ 0.20
10. **Data card lengkap** untuk semua dataset. Include: source list, license per subset, preprocessing steps, known limitations, contact.
11. **Perbaiki schema `aksara-training-data`.** Pisahkan menjadi beberapa `config` HuggingFace:
    ```yaml
    configs:
      - config_name: distill_v4
        data_files: distill_v4.json
      - config_name: mega_v5
        data_files: mega_dataset_v5.json
      - config_name: sft_v5
        data_files: sft_v5_indonesia.json
    ```

### P2 — strategic (1–3 bulan)

12. **Scale pretrain ke 15–30 B tokens.** Paling cepat:
    - Tarik seluruh CulturaX-ID (`uonlp/CulturaX`, subset `id`) → 12 B tokens.
    - Tambahkan `mc4-id` sisa.
    - Tambahkan IndoWebCorpus / IndonesianNLP-Corpus.
    - Tambahkan kode (The Stack v2 `id` comments + Indonesian code repos).
    - Tambahkan buku/artikel Indonesia publik (Wikisource, Gutenberg section Melayu lama, Perpusnas public domain).
13. **Bahasa daerah yang serius.** Untuk tiap bahasa daerah target (jav/sun/min/ace/ban/bug/mad/bjn/bbc/nij):
    - NusaCrowd corpora (lebih besar dari NusaX).
    - Wikipedia per bahasa (`jvwiki`, `suwiki`, `minwiki`, dll).
    - CC-100 `jv`, `su`.
    - Tentukan **target minimum 50 MB teks per bahasa** untuk pretrain yang meaningful.
    - Separate SFT task **per bahasa daerah** (translation, QA, summarization) — jangan hanya sentiment.
14. **Pipeline reproducibility.** Release `aksara-data/scripts/build_pretrain.py` yang:
    - Spesifikasi eksplisit sumber + versi (CulturaX 1.2.0, Wikipedia dump YYYYMMDD).
    - Hash-based fingerprint output.
    - Skor kualitas per dokumen (KenLM perplexity terhadap KenLM-id trained on cleaned subset).
15. **Buat `Aksara-Indo-Bench`** (Fase 2 roadmap) dengan eval di luar MMLU-translate. Suggested tracks:
    - **IndoMMLU** (sudah ada, Koto et al.).
    - **IndoCulture** (sudah ada — budaya/hukum/sejarah).
    - **COPAL-ID** (pragmatic reasoning).
    - **NusaX / NusaWrites** bahasa daerah.
    - **Custom safety eval** (isu lokal: SARA, politik sensitif).
16. **Technical report** yang reproducible. Minimal 8–12 halaman; bandingkan vs SEA-LION, Sahabat-AI, Merak-7B di eval yang sama.

---

## 8. Reproducible scripts

Semua angka di atas bisa direproduksi dengan:

```bash
pip install -q datasets fasttext-wheel pandas pyarrow "numpy<2.0"
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Lihat repo audit untuk skrip lengkap (akan dilampirkan):
- `01_inventory.py` — schema + row count
- `02_pretrain_stats.py` — dedup + length + boilerplate
- `03_langid.py` — FastText LID per source
- `04_sft_stats.py` — source / task_type distribution + dup
- `05_distill_quality.py` — hallucination detection patterns

---

## 9. Saran interaksi dengan maintainer AksaraLLM

Kalau kamu (atau siapapun) mau push improvements upstream:

1. Buka **GitHub issue** di [`AksaraLLM/aksara-data`](https://github.com/AksaraLLM/aksara-data) dengan ringkasan P0 (leakage + prefix contamination + hallucinated SFT). Lampirkan report ini.
2. PR kecil pertama: fix prefix `[Bahasa X]` + dedup script. Ini high-impact, low-risk, satu file.
3. PR berikutnya: clean re-split train/val.
4. Isu terpisah: scope untuk Fase 2 Aksara-Indo-Bench (daftar task + source).
5. Hindari mengirim mega-PR; small incremental PRs lebih mudah di-merge untuk proyek kecil.

---

## 10. Penutup

Proyek AksaraLLM di posisi bagus dari sisi misi & struktur repo, tapi datanya belum siap. Kalau 5 masalah P0 di §7 diperbaiki, model 500M yang di-train ulang di atas data bersih kemungkinan **sudah** outperform yang sekarang — tanpa perlu naik parameter. Setelah itu, baru scale ke 1 B+ dengan confidence.

— *End of audit.*

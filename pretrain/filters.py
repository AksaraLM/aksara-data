"""Shared quality + language filters for the pretrain pipeline.

Used by both clean_pretrain.py (original corpus) and fetch_extras.py (fresh
wiki-id), so all text that lands in the final dataset goes through the
same cleaning steps.
"""
import re
from functools import lru_cache

# Gopher-style quality filter =====================================

_nav_re = re.compile(
    r"(Log in|Sign up|Sign in|Cookies|Terms of Service|Privacy Policy|Daftar isi|Halaman utama)",
    re.I,
)
_url_re = re.compile(r"https?://([^/\s)]+)")

BLOCK_DOMAINS = {
    # spam / SEO / bogus "health" domains seen in audit
    "arenasbo88.com", "malehealthcenter.com", "wearebrewstuds.com",
    "hargano.com", "gpgo.in", "159.65.11.81",
    # gambling / togel / illegal betting
    "sbobet.com", "sbobet88.com", "sbobet365.com", "bola88.com",
    "togel.com", "hongkongpools.com", "prediksisgp.com",
}


def gopher_ok(t: str) -> bool:
    """Return True if `t` passes a Chinchilla/Gopher-style quality gate.

    Applied identically to:
      - original aksara-pretrain-id rows in clean_pretrain.py
      - fresh Wikipedia-id rows in fetch_extras.py
    """
    if not t or len(t) < 80:
        return False
    lines = t.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False
    avg_w = sum(len(l.split()) for l in non_empty) / len(non_empty)
    if avg_w < 3:
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha / len(t) < 0.65:
        return False
    ell = sum(
        1 for l in non_empty
        if l.rstrip().endswith("…") or l.rstrip().endswith("...")
    )
    if ell / len(non_empty) > 0.3:
        return False
    for dom in _url_re.findall(t):
        d = dom.lower().removeprefix("www.")
        if d in BLOCK_DOMAINS:
            return False
    nh = len(_nav_re.findall(t))
    if nh > 5 and nh / max(len(non_empty), 1) > 0.05:
        return False
    return True


# GlotLID language filter ==========================================

@lru_cache(maxsize=1)
def _load_lid(model_path="glotlid.bin"):
    import fasttext
    return fasttext.load_model(model_path)


def classify_lang_batch(batch, model_path="glotlid.bin"):
    """Return {lid_label: [...], lid_prob: [...]} for a datasets batch."""
    lid = _load_lid(model_path)
    preds, probs = [], []
    for t in batch["text"]:
        x = (t or "").replace("\n", " ").replace("\r", " ")[:2000].strip()
        if not x:
            preds.append("none")
            probs.append(0.0)
            continue
        lbl, p = lid.predict(x, k=1)
        preds.append(lbl[0].replace("__label__", ""))
        probs.append(float(p[0]))
    return {"lid_label": preds, "lid_prob": probs}


def lang_ok(ex, min_prob: float = 0.60) -> bool:
    """Keep row if Indonesian (GlotLID `ind_Latn` with P >= `min_prob`),
    or explicitly trust NusaX sources (which have many short regional-language
    examples that GlotLID isn't always confident on but are labeled at source).
    """
    s = (ex.get("source") or "")
    if s.startswith("nusax-"):
        return True
    if ex.get("lid_label") == "ind_Latn" and ex.get("lid_prob", 0.0) >= min_prob:
        return True
    return False

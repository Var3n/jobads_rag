"""Step 2: per-region quality metrics and flags.

The post-corrected `text` is usually much better than the raw OCR, but the correction model
sometimes loops, duplicates whole ads, or writes fluent text where the OCR was noise. Those
cases are flagged here, together with regions that are not job ads at all (entries of the
Vienna death register, 1850–1899). Raw metrics are stored too, so thresholds can be changed
later without recomputing.

Thresholds were calibrated on the Wiener Zeitung sample (see the step 2 notes in the README).
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

from hisrag.normalize.text import normalize_text

# A 4-word sequence occurring this often, and at least REPEAT_FACTOR times as often as in the
# OCR, means the post-correction looped or duplicated text.
REPEAT_MIN = 4
REPEAT_FACTOR = 2
# Post-correction this much longer than the OCR: text was duplicated or added.
EXPANDED_LEN_RATIO = 1.3
# Share of the post-correction's character trigrams that also occur in the OCR. Trimming or
# fixing OCR keeps this near 1; below this the text is largely reconstructed from noise and
# often invented ("Gute Köchin sucht Stelle…" from "öGSCαααπππ…"). Low sim_ocr alone is not
# a signal: it mostly means the correction dropped fragments of neighbouring notices.
MIN_OCR_SUPPORT = 0.6
TOO_SHORT_CHARS = 30

_WORD = re.compile(r"\w+")
_NON_ALNUM = re.compile(r"[^a-zäöüß0-9]+")
_WORDLIKE = re.compile(r"^[A-Za-zÄÖÜäöüſßéèàçò]{3,}[.,;:!?]?$")

# Death register entry: "Kraft Vincenz, Hausdiener, 48 J., VIII., Lerchenfelderstraße 56, Lungenschwindsucht."
_AGE = r"\d{1,3}\s?[½¼¾]?\s?(?:J\.|Mon\.|Mt\.|T\.|W\.|St\.)"
_DEATH_ENTRY = re.compile(_AGE + r"\s?,?\s?(?:[IVX1l]{1,5}\.,|[A-ZÄÖÜ][\wſäöü.-]+,\s)")
_DEATH_NAME = re.compile(r"(?:^|[.)]\s)[A-ZÄÖÜ][\wſäöü-]+\s[A-ZÄÖÜ][\wſäöü]+,\s[^,]{2,40},\s" + _AGE)
_SEARCH_VERB = re.compile(r"[sſ]ucht\s+(?:Stelle|Po[sſ]ten|Platz|Dien[sſ]t|Be[sſ]ch)|wün[sſ]cht|bittet|empfiehlt", re.I)

_STOP = {
    "de": re.compile(r"\b(?:der|die|das|und|mit|für|sich|ein|eine|einen|wird|ist|in|zu|bei|als|von|auf|an)\b", re.I),
    "it": re.compile(r"\b(?:il|della|delle|di|per|che|nel|alla|dei|sono|con|gli|posto)\b"),
    "fr": re.compile(r"\b(?:le|les|des|et|pour|une|dans|du|est|avec|qui|sa|leur)\b"),
}


def max_ngram_repeat(text: str, n: int = 4) -> int:
    words = _WORD.findall(text.lower())
    if len(words) < 2 * n:
        return 0
    return max(Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1)).values())


def _trigrams(text: str) -> set[str]:
    s = _NON_ALNUM.sub(" ", normalize_text(text).lower())
    return {s[i:i + 3] for i in range(len(s) - 2) if " " not in s[i:i + 3]}


def ocr_support(text: str, ocr: str) -> float:
    """Share of the corrected text's trigrams found in the OCR (1.0 = fully supported)."""
    grams = _trigrams(text)
    return len(grams & _trigrams(ocr)) / len(grams) if grams else 1.0


def wordlike_share(text: str) -> float:
    tokens = text.split()
    return sum(bool(_WORDLIKE.match(t)) for t in tokens) / len(tokens) if tokens else 0.0


def death_register_entries(text: str) -> int:
    if _SEARCH_VERB.search(text):
        return 0
    return max(len(_DEATH_ENTRY.findall(text)), len(_DEATH_NAME.findall(text)))


def guess_language(text_norm: str) -> str:
    counts = {lang: len(p.findall(text_norm)) for lang, p in _STOP.items()}
    foreign = max(("it", "fr"), key=counts.get)
    if counts[foreign] >= 2 and counts[foreign] > counts["de"]:
        return foreign
    if counts["de"] == 0 and sum(c.isalpha() for c in text_norm) < 10:
        return "unknown"
    return "de"


def assess(ads: pd.DataFrame) -> pd.DataFrame:
    """ads needs: ad_id, newspaper, year, label, text, text_ocr, heading_text."""
    text, ocr = ads["text"].fillna(""), ads["text_ocr"].fillna("")
    out = pd.DataFrame({"ad_id": ads["ad_id"], "newspaper": ads["newspaper"], "year": ads["year"]})
    out["text_norm"] = text.map(normalize_text)
    out["heading_norm"] = ads["heading_text"].map(normalize_text).replace("", None)
    out["lang"] = out["text_norm"].map(guess_language)

    out["sim_ocr"] = [fuzz.ratio(a, b) / 100 for a, b in zip(text, ocr)]
    out["len_ratio"] = text.str.len() / ocr.str.len().clip(lower=1)
    out["repeat_max"] = text.map(max_ngram_repeat)
    out["repeat_max_ocr"] = ocr.map(max_ngram_repeat)
    out["ocr_wordlike"] = ocr.map(wordlike_share)
    out["ocr_support"] = [ocr_support(a, b) for a, b in zip(text, ocr)]
    out["death_entries"] = text.map(death_register_entries)

    out["flag_pc_repetition"] = (out["repeat_max"] >= REPEAT_MIN) & \
        (out["repeat_max"] >= REPEAT_FACTOR * out["repeat_max_ocr"].clip(lower=1))
    out["flag_pc_expanded"] = out["len_ratio"] > EXPANDED_LEN_RATIO
    out["flag_pc_unsupported"] = out["ocr_support"] < MIN_OCR_SUPPORT
    out["flag_too_short"] = (out["text_norm"].str.len() < TOO_SHORT_CHARS) & (ads["label"] != "heading").to_numpy()
    out["flag_death_register"] = out["death_entries"] > 0

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["n_flags"] = out[flag_cols].sum(axis=1).astype("int8")
    return out


def summarize(q: pd.DataFrame, labels: pd.Series) -> dict:
    flag_cols = [c for c in q.columns if c.startswith("flag_")]
    return {
        "rows": len(q),
        "flags": {c.removeprefix("flag_"): int(q[c].sum()) for c in flag_cols},
        "rows_with_any_flag": int((q["n_flags"] > 0).sum()),
        "death_register_by_label": labels[q["flag_death_register"].to_numpy()].value_counts().to_dict(),
        "lang": q["lang"].value_counts().to_dict(),
    }


SCHEMA = pa.schema(
    [
        ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()),
        ("text_norm", pa.string()), ("heading_norm", pa.string()), ("lang", pa.string()),
        ("sim_ocr", pa.float32()), ("len_ratio", pa.float32()),
        ("repeat_max", pa.int16()), ("repeat_max_ocr", pa.int16()),
        ("ocr_wordlike", pa.float32()), ("ocr_support", pa.float32()), ("death_entries", pa.int16()),
        ("flag_pc_repetition", pa.bool_()), ("flag_pc_expanded", pa.bool_()), ("flag_pc_unsupported", pa.bool_()),
        ("flag_too_short", pa.bool_()), ("flag_death_register", pa.bool_()), ("n_flags", pa.int8()),
    ]
)

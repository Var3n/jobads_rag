"""Step 3: group repeated printings of the same ad.

Ads were printed several times (official notices usually three times, private ads for days or
weeks), each time with different OCR errors and region boundaries. Candidate pairs come from
MinHash LSH on character 5-grams, restricted to the same newspaper within WINDOW_DAYS. Each
candidate is then verified word by word, because similarity alone cannot separate reprints
from template notices: two vacancy notices for different schools can be 95 % identical and
differ only in the place name. Two regions are the same ad when

  * the matching words cover at least MIN_COVER of the shorter text, and
  * no passage was *substituted*: a stretch where both texts have different content that
    does not resemble each other (not an OCR variant) and contains a capitalized word or a
    number on both sides (a place, a date, a salary, a subject).

Extra text on one side only (a region starting a line earlier, an appended ad number) does
not count. Accepted pairs are joined into clusters; the best-quality member is canonical.
Thresholds were calibrated on the Wiener Zeitung sample.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

WINDOW_DAYS = 60
NUM_PERM = 64
BANDS = 16                 # 16 bands × 4 rows: candidate from ~50 % Jaccard similarity on 5-grams
SHINGLE = 5
MIN_WORDS = 8              # shorter texts are too generic to match reliably
MIN_LEN_RATIO = 0.5        # a fragment is not matched against a full ad
MIN_COVER = 0.7
VARIANT_SIM = 80           # differing words at least this similar (0–100) are OCR variants

_NON_ALNUM = re.compile(r"[^a-zäöüß0-9]+")
_WORD = re.compile(r"\w+")
_rng = np.random.default_rng(20240501)
_A = _rng.integers(1, 2**63, NUM_PERM, dtype=np.uint64) | np.uint64(1)
_B = _rng.integers(0, 2**63, NUM_PERM, dtype=np.uint64)
_POW = np.array([31**k for k in range(SHINGLE - 1, -1, -1)], dtype=np.uint64)


def minhash(text: str) -> np.ndarray:
    """MinHash signature over character 5-grams (multiply-shift hashing, vectorized)."""
    s = _NON_ALNUM.sub(" ", text.lower()).strip()
    codes = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if len(codes) < SHINGLE:
        codes = np.pad(codes, (0, SHINGLE - len(codes)))
    shingles = np.unique(np.lib.stride_tricks.sliding_window_view(codes, SHINGLE) @ _POW)
    with np.errstate(over="ignore"):
        return ((_A[:, None] * shingles[None, :] + _B[:, None]) >> np.uint64(32)).min(axis=1)


def candidate_pairs(signatures: np.ndarray, days: np.ndarray, groups: np.ndarray) -> set[tuple[int, int]]:
    """Index pairs sharing an LSH band, in the same group (newspaper) and within WINDOW_DAYS."""
    rows = NUM_PERM // BANDS
    pairs: set[tuple[int, int]] = set()
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = np.ascontiguousarray(signatures[:, band * rows:(band + 1) * rows])
        for i, key in enumerate(map(bytes, block)):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda i: days[i])
            for a, i in enumerate(members):
                for j in members[a + 1:]:
                    if days[j] - days[i] > WINDOW_DAYS:
                        break
                    if groups[i] == groups[j]:
                        pairs.add((min(i, j), max(i, j)))
    return pairs


def _strong(word: str) -> bool:
    """Content that tells two notices apart: names and nouns, numbers, longer lowercase words
    ("deutsche" vs "croatische" Sprache)."""
    return (len(word) >= 3 and (word[0].isupper() or word.isdigit())) or len(word) >= 6


def _unmatched(words: list[str], others: list[str]) -> list[str]:
    """Strong words without an OCR-variant counterpart among `others`."""
    lowered = [o.lower() for o in others]
    return [w for w in words if _strong(w)
            and not any(fuzz.ratio(w.lower(), o) >= VARIANT_SIM for o in lowered)]


def substitutions(a: str, b: str) -> tuple[float, list[str], list[str]]:
    """Word alignment of two texts: (cover, substituted words in a, substituted words in b).

    cover is the share of the shorter text's words that match. Substituted words are content
    words (see _strong) that one side has and the other lacks, without an OCR-variant
    counterpart; differences are only a substitution when both sides have such words.
    """
    wa, wb = _WORD.findall(a), _WORD.findall(b)
    ops = difflib.SequenceMatcher(None, [w.lower() for w in wa], [w.lower() for w in wb],
                                  autojunk=False).get_opcodes()
    cover = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal") / max(min(len(wa), len(wb)), 1)

    # Text on one side only at the very start or end is a region-boundary difference: ignore it.
    diff = [op for op in ops if op[0] != "equal"]
    if diff and diff[0][0] in ("insert", "delete") and (diff[0][1] == 0 or diff[0][3] == 0):
        diff = diff[1:]
    if diff and diff[-1][0] in ("insert", "delete") and (diff[-1][2] == len(wa) or diff[-1][4] == len(wb)):
        diff = diff[:-1]
    # Substitutions often come out as a deletion plus an insertion around a shared word, so all
    # remaining differences are pooled per side.
    only_a = [w for _, i1, i2, _, _ in diff for w in wa[i1:i2]]
    only_b = [w for _, _, _, j1, j2 in diff for w in wb[j1:j2]]
    left, right = _unmatched(only_a, only_b), _unmatched(only_b, only_a)
    return (cover, left, right) if left and right else (cover, [], [])


def same_ad(a: str, b: str) -> bool:
    """Verification of a candidate pair (see module docstring)."""
    shorter, longer = sorted((len(_WORD.findall(a)), len(_WORD.findall(b))))
    if shorter < MIN_WORDS or shorter / max(longer, 1) < MIN_LEN_RATIO:
        return False
    cover, left, _ = substitutions(a, b)
    return cover >= MIN_COVER and not left


def _canonical(regions: pd.DataFrame, root: np.ndarray) -> np.ndarray:
    """Row index of each row's canonical member: fewest quality flags, best OCR support, earliest."""
    ranked = pd.DataFrame({"root": root, "flags": regions["n_flags"].to_numpy(),
                           "support": -regions["ocr_support"].to_numpy(),
                           "date": pd.to_datetime(regions["date"]).to_numpy()})
    best = ranked.sort_values(["root", "flags", "support", "date"]).drop_duplicates("root")
    return pd.Series(best.index.to_numpy(), index=best["root"].to_numpy()).loc[root].to_numpy()


def _clusters(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def deduplicate(regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """regions needs: ad_id, newspaper, year, date, text_norm, eligible, n_flags, ocr_support.

    Ineligible regions (headings, death register, too short) become single-member clusters.
    """
    regions = regions.reset_index(drop=True)
    elig = regions.index[regions["eligible"]].to_numpy()
    sub = regions.loc[elig]
    days = (pd.to_datetime(sub["date"]) - pd.Timestamp("1800-01-01")).dt.days.to_numpy()
    groups = sub["newspaper"].to_numpy()
    texts = sub["text_norm"].tolist()

    sigs = np.vstack([minhash(t) for t in texts]) if texts else np.empty((0, NUM_PERM), np.uint64)
    cands = candidate_pairs(sigs, days, groups)
    edges = [(i, j) for i, j in cands if same_ad(texts[i], texts[j])]

    root = np.arange(len(regions))
    root[elig] = elig[_clusters(len(elig), edges)]
    canonical = _canonical(regions, root)

    # Chains can link two different ads through a third region (often a generic fragment of a
    # multi-part notice). A member with substituted content relative to its canonical is detached.
    texts_all = regions["text_norm"].tolist()
    members = np.flatnonzero(canonical != np.arange(len(regions)))
    detached = [i for i in members if substitutions(texts_all[i], texts_all[canonical[i]])[1]]
    if detached:
        root[detached] = detached
        canonical = _canonical(regions, root)

    out = regions[["ad_id", "newspaper", "year", "date"]].copy()
    out["_root"] = root
    out["dup_cluster_id"] = out["ad_id"].to_numpy()[canonical]
    out["is_canonical"] = out.index.to_numpy() == canonical
    grp = out.groupby("_root")
    out["dup_cluster_size"] = grp["ad_id"].transform("size").astype("int32")
    dates = pd.to_datetime(out["date"])
    out["run_first_date"] = dates.groupby(out["_root"]).transform("min").dt.date
    out["run_last_date"] = dates.groupby(out["_root"]).transform("max").dt.date
    out["run_days"] = ((pd.to_datetime(out["run_last_date"]) - pd.to_datetime(out["run_first_date"])).dt.days
                       .astype("int32"))

    sizes = out.loc[out["is_canonical"], "dup_cluster_size"]
    report = {
        "regions": len(out),
        "eligible": int(len(elig)),
        "candidate_pairs": len(cands),
        "accepted_pairs": len(edges),
        "detached_from_chains": len(detached),
        "distinct_ads": int(out["is_canonical"].sum()),
        "regions_in_multi_clusters": int((out["dup_cluster_size"] > 1).sum()),
        "cluster_sizes": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
    }
    return out.drop(columns=["_root"]), report


SCHEMA = pa.schema(
    [
        ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()),
        ("dup_cluster_id", pa.string()), ("dup_cluster_size", pa.int32()), ("is_canonical", pa.bool_()),
        ("run_first_date", pa.date32()), ("run_last_date", pa.date32()), ("run_days", pa.int32()),
    ]
)

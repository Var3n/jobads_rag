"""Text normalization for indexing and search.

Only used for the index and for matching: display and span offsets always use the original
`text`. Deliberately conservative: historical spellings (Wirthschafterin, Commis, Classe) are
kept, because rewriting them blindly damages other words (Theater, Mathematik); spelling
variants are handled on the query side instead.
"""

from __future__ import annotations

import re
import unicodedata

# Words after which a trailing hyphen is an elision ("Lehrer⸗ und Lehrerinstelle",
# "Zwirn⸗ und Wollhandlung"), not a line-break hyphen, so the hyphen must stay.
_CONJ = r"(?:und|oder|bezw|bzw|resp|respective|beziehungsweise|sowie|u|od|als|bis|wie|or|and|e|o|et|ou)\b"
_LOWER = r"[a-zäöüßſ]"

_CHAR_MAP = str.maketrans({
    "ſ": "s", "ꝛ": "r", "ẞ": "SS",
    "„": '"', "“": '"', "”": '"', "»": '"', "«": '"',
    "‚": "'", "‘": "'", "’": "'", "›": "'", "‹": "'",
    "⸗": "-", "¬": "-",  # line-break hyphens left over after the rules below
})

# Hyphen + line break + lowercase continuation that is not a conjunction: one word split over two lines.
_JOIN_LINEBREAK = re.compile(rf"([^\W\d_])[⸗¬-]\s*\n\s*(?!{_CONJ})(?={_LOWER})")
# Hyphen + line break before a conjunction: elision, keep hyphen and space ("Lehrer- oder").
_ELISION_LINEBREAK = re.compile(rf"([^\W\d_])[⸗¬-]\s*\n\s*(?={_CONJ})")
# Hyphen + line break before a capital or digit: compound split over two lines, keep the hyphen.
_KEEP_LINEBREAK = re.compile(r"([^\W\d_])[⸗¬-]\s*\n\s*")
# The post-correction often turned line breaks into spaces ("Re⸗ gimente"): same rule on spaces.
_JOIN_SPACE = re.compile(rf"([^\W\d_])[⸗¬-] +(?!{_CONJ})(?={_LOWER})")
_WS = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not isinstance(text, str) or not text:  # None, NaN, pd.NA
        return ""
    s = unicodedata.normalize("NFC", text)
    s = _JOIN_LINEBREAK.sub(r"\1", s)
    s = _ELISION_LINEBREAK.sub(r"\1- ", s)
    s = _KEEP_LINEBREAK.sub(r"\1-", s)
    s = _JOIN_SPACE.sub(r"\1", s)
    s = s.translate(_CHAR_MAP)  # after the hyphen rules, so ſ-words still count as lowercase there
    return _WS.sub(" ", s).strip()

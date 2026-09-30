"""Step 5: requirement tags.

Every distinct phrase of the requirement columns is sent once to the LLM and mapped to zero or
more tags (dimension, value, detail) from the controlled vocabulary in vocab/requirements.yaml.
The vocabulary is part of the prompt, and the prompt version is derived from the file's content,
so editing the vocabulary means the next run re-maps all phrases.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import yaml
from pydantic import BaseModel, Field, create_model
from rapidfuzz import fuzz

from hisrag.config import REPO_ROOT
from hisrag.llm.client import DHClient
from hisrag.normalize.batching import run_batches
from hisrag.normalize.text import normalize_text

COLUMNS = ("job_specific", "background", "language", "activity", "attitude_towards_work", "interpersonal")
VOCAB_PATH = REPO_ROOT / "vocab" / "requirements.yaml"


class Vocabulary:
    def __init__(self, path: Path = VOCAB_PATH):
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        self.version = f"requirements-v{raw.get('version', 1)}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        self.dimensions: dict[str, dict] = {}
        self.group_of: dict[str, str] = {}
        for group, dims in raw["groups"].items():
            for name, spec in dims.items():
                self.dimensions[name] = spec
                self.group_of[name] = group

    def closed_values(self, dimension: str) -> set[str] | None:
        values = self.dimensions[dimension].get("values")
        return {v.lower() for v in values} if values else None

    def prompt_section(self) -> str:
        lines = []
        for group in dict.fromkeys(self.group_of.values()):
            lines.append(f"\n{group}:")
            for name, spec in self.dimensions.items():
                if self.group_of[name] != group:
                    continue
                line = f"- {name}: {spec['description']}"
                if spec.get("values"):
                    line += f". Werte (genau einer davon): {', '.join(spec['values'])}"
                if spec.get("suggested_values"):
                    line += f". Bevorzugte Werte: {', '.join(spec['suggested_values'])}"
                if spec.get("value_rule"):
                    line += f". Wertformat: {spec['value_rule']}"
                if spec.get("examples"):
                    line += ". Beispiele: " + "; ".join(f'"{k}" → {v}' for k, v in spec["examples"].items())
                lines.append(line)
        return "\n".join(lines)


def build_models(vocab: Vocabulary):
    """Pydantic models whose `dimension` is restricted to the vocabulary (enforced by guided decoding)."""
    Dimension = Literal[tuple(vocab.dimensions)]  # type: ignore[valid-type]
    Tag = create_model("Tag", dimension=(Dimension, ...), value=(str, ...), detail=(str | None, None))
    Item = create_model("RequirementItem", i=(int, Field(description="Nummer des Ausdrucks")),
                        tags=(list[Tag], Field(description="leer, wenn der Ausdruck keine Information enthält")))
    Batch = create_model("RequirementBatch", items=(list[Item], ...))
    return Tag, Item, Batch


def system_prompt(vocab: Vocabulary) -> str:
    return f"""Du ordnest Ausdrücke aus historischen Stellenanzeigen der Wiener Zeitung (1850–1950) einem festen Vokabular zu.
Die Ausdrücke wurden automatisch aus OCR-Text extrahiert: Anforderungen an Bewerber, Angaben von Stellensuchenden über sich selbst, Tätigkeiten, Bedingungen. Manche sind Bruchstücke ohne Information.

Für jeden Ausdruck gibst du eine Liste von Tags zurück, je Tag: dimension (aus der Liste unten), value (kurz, modernes Deutsch), detail (optional: Niveau, Anzahl, Betrag, "bevorzugt" …, sonst null).
Regeln:
- Tagge nur, was im Ausdruck selbst steht. Der Kontext hilft nur zu entscheiden, WELCHE Dimension gemeint ist (z. B. ob "deutscher" zu "Unterrichtssprache" gehört); er liefert keine zusätzlichen Tags, Werte oder Details. "gründlich" ergibt nur gründlich, auch wenn im Kontext "gründlich und schnell" steht; "Kenntnis der deutschen Sprache" hat kein detail.
- Stehen mehrere Kontexte (getrennt durch ‖) dabei, kommt der Ausdruck in verschiedenen Anzeigen vor: wähle die Deutung, die allgemein zutrifft, nicht die eines einzelnen Kontexts.
- Ein Tag pro Information; Aufzählungen ergeben mehrere Tags ("Buchhaltung, Correspondenz und Stenographie" → drei Tags fachkenntnisse).
- Leere Liste für Bruchstücke ohne erkennbaren Inhalt ("welches fähig ist", "ord", "ge", "kundig" allein).
- Werte aus den Beispielen und bevorzugte Werte haben Vorrang, wenn sie die Bedeutung treffen ("gehörig instruierten" → vorschriftsmäßiges Gesuch); sonst das eigene Wort des Ausdrucks, z. B. "sympathisch".
- Werte immer in moderner Schreibung (documentirt → dokumentiert, Correspondenz → Korrespondenz); nur Sprachbezeichnungen bleiben historisch (siehe unten).
- Die angegebene Spalte ist die automatische Einordnung und kann falsch sein; entscheide nach dem Inhalt.
- Bei Dimensionen mit festen Werten verwendest du genau einen dieser Werte. Die Schreibung wird modernisiert (Correspondenz → Korrespondenz), Sprachbezeichnungen bleiben aber wie in der Quelle (böhmisch → Böhmisch, ruthenisch → Ruthenisch, tschechisch → Tschechisch).
- Erfinde nichts: verstümmelte Wörter, die du nicht sicher erkennst, lässt du weg.

Dimensionen:{vocab.prompt_section()}

Antworte mit einem JSON-Objekt {{"items": [...]}} mit genau einem Element pro Ausdruck, "i" = Nummer des Ausdrucks."""


def phrase_key(text: str) -> str:
    key = normalize_text(text).lower()
    key = re.sub(r"^[^\wäöü]+|[^\wäöü]+$", "", key)
    return re.sub(r"\s+", " ", key)


def _snippet(text: str, start: int, end: int, width: int = 50) -> str:
    left = normalize_text(text[max(0, start - width):start])
    right = normalize_text(text[end:end + width])
    return f"…{left} [{normalize_text(text[start:end])}] {right}…"


# Frequent phrases occur in different kinds of ads ("deutsche Sprache" as a skill or as a
# taught subject); they get several contexts from different years so the model picks the
# reading that fits the phrase in general, not the one of a single example.
MULTI_CONTEXT_MIN_COUNT = 3
MAX_CONTEXTS = 3
CONTEXT_SEPARATOR = " ‖ "


def collect_phrases(spans: pd.DataFrame) -> pd.DataFrame:
    """One row per (column, phrase key). spans: ad_id, column, text, start, end, phrase (year optional)."""
    spans = spans.assign(pkey=spans["phrase"].map(phrase_key))
    spans = spans[spans["pkey"] != ""]
    if "year" in spans:
        spans = spans.sort_values("year", kind="stable")
    rows = []
    for (column, pkey), grp in spans.groupby(["column", "pkey"], sort=False):
        grp = grp.drop_duplicates("ad_id")
        n = len(grp)
        picks = [n // 2] if n < MULTI_CONTEXT_MIN_COUNT else sorted({0, n // 2, n - 1})[:MAX_CONTEXTS]
        rows.append({
            "key": f"{column}|{pkey}", "column": column, "phrase_key": pkey, "count": int(n),
            "surface": normalize_text(Counter(grp["phrase"]).most_common(1)[0][0]).strip(" .,;:-"),
            "context": CONTEXT_SEPARATOR.join(_snippet(grp.iloc[i]["text"], grp.iloc[i]["start"], grp.iloc[i]["end"])
                                              for i in picks),
        })
    return pd.DataFrame(rows).sort_values(["column", "phrase_key"]).reset_index(drop=True)


_DETAIL_WORD = re.compile(r"[^\W\d_]{4,}|\d{2,}")

# Standard details the model writes for a historical wording that shares no letters with them.
DETAIL_CUES = {
    "bevorzugt": ("bevorzug", "vorzug", "möglichst", "womöglich", "eventuell", "erwünscht", "wünschenswert"),
    "abgeschlossen": ("absolv", "abgelegt", "beendet", "vollendet", "zurückgelegt"),
}


def _fold(text: str) -> str:
    """Lower case without diacritics and with historical spellings unified (Correcte → korrekte,
    Theil → teil, nähe → nahe, familières → familieres), applied to both sides of a comparison."""
    text = unicodedata.normalize("NFKD", normalize_text(text).lower().replace("ß", "ss"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"c(?=[eiy])", "z", text.replace("ck", "k").replace("th", "t"))
    return text.replace("c", "k")


def detail_is_grounded(detail: str, phrase: str) -> bool:
    """A detail must restate the phrase, not the context: every content word of the detail
    (letters ≥ 4, or a number) has to occur in the phrase, allowing spelling variants
    (Correspondenz / Korrespondenz, vollkommen / vollkommene, 14 Jahre / 14jährigen) and the
    cue words of DETAIL_CUES (möglichst → bevorzugt)."""
    text = _fold(phrase_key(phrase))
    for word in _DETAIL_WORD.findall(normalize_text(detail).lower()):
        cues = DETAIL_CUES.get(word, ())
        if any(_fold(c) in text for c in cues):
            continue
        w = _fold(word)
        if not (w in text or fuzz.partial_ratio(w, text) >= 80):
            return False
    return True


def value_is_grounded(value: str, phrase: str) -> bool:
    """Diagnostic only: does any content word of the value occur in the phrase? False for values
    taken from the context ("ger" → Buchführung), but also for modern paraphrases
    ("hiesigen Platz" → Ortskenntnis), so it flags tags for review and does not drop them."""
    text = _fold(phrase_key(phrase))
    words = [_fold(w) for w in _DETAIL_WORD.findall(normalize_text(value).lower())]
    return not words or any(w in text or fuzz.partial_ratio(w, text) >= 80 for w in words)


class RequirementMapper:
    def __init__(self, vocab: Vocabulary | None = None):
        self.vocab = vocab or Vocabulary()
        self.Tag, self.Item, self.Batch = build_models(self.vocab)
        self.system = system_prompt(self.vocab)

    def messages(self, batch: pd.DataFrame) -> list[dict]:
        lines = [f'{n}. [{r.column}] "{r.surface}" — Kontext{"e" if CONTEXT_SEPARATOR in r.context else ""}: {r.context}'
                 for n, r in enumerate(batch.itertuples(), 1)]
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": "Ausdrücke:\n" + "\n".join(lines)}]

    def run(self, client: DHClient, phrases: pd.DataFrame, *, batch_size: int = 40,
            progress: bool = True) -> tuple[dict, dict]:
        return run_batches(client, phrases, job="requirements", batch_size=batch_size, progress=progress,
                           build_messages=self.messages, result_model=self.Batch,
                           empty_item=lambda: self.Item(i=1, tags=[]))

    def to_dictionary(self, phrases: pd.DataFrame, results: dict, model: str) -> pd.DataFrame:
        d = phrases.copy()
        tags = []
        self.details_dropped = 0
        for key, surface in zip(d["key"], d["surface"]):
            if key not in results:
                tags.append(None)
                continue
            out = []
            for t in results[key].tags:
                closed = self.vocab.closed_values(t.dimension)
                value = t.value.strip()
                detail = (t.detail or "").strip() or None
                if detail and not detail_is_grounded(detail, surface):
                    detail = None
                    self.details_dropped += 1
                out.append({"dimension": t.dimension, "group": self.vocab.group_of[t.dimension], "value": value,
                            "detail": detail, "detail_raw": (t.detail or "").strip() or None,
                            "in_vocab": closed is None or value.lower() in closed, "verified": None})
            tags.append(out)
        d["tags"] = tags
        d["is_informative"] = d["tags"].map(lambda t: bool(t) if t is not None else None)
        d["prompt_version"] = self.vocab.version
        d["model"] = model
        return d[DICT_SCHEMA.names]


# The mapping sees example ads, and for short or frequent phrases it tags what one of those ads
# says ("verheirathet" → kinderlos, "absolvirter" → Bergschule). Tags whose value does not occur
# in the phrase are therefore checked a second time, without any context: does the phrase alone
# state this? Rejected tags stay in the dictionary (verified = false) but give no ad rows.
VERIFY_BATCH_SIZE = 60


def verify_system_prompt(vocab: Vocabulary) -> str:
    dims = "\n".join(f"- {name}: {spec['description']}" for name, spec in vocab.dimensions.items())
    return f"""Du prüfst Tags, die Ausdrücken aus historischen Stellenanzeigen der Wiener Zeitung (1850–1950) zugeordnet wurden.
Ein Ausdruck kommt in vielen Anzeigen vor; ein Tag darf nur sagen, was der Ausdruck selbst aussagt, in jeder dieser Anzeigen. Du siehst deshalb nur den Ausdruck, keinen Kontext.

Für jeden Tag: stated = true, wenn der Ausdruck selbst diese Information enthält. Alte Schreibung, OCR-Fehler, Abkürzungen, andere Sprachen und moderne Umschreibungen im Wert sind in Ordnung:
- "gehörig instruierten" → bewerbungsformalitaeten = vorschriftsmäßiges Gesuch: true
- "in besten Jahren" → alter = mittleres Alter: true
- "Franzose" → herkunft = Frankreich: true
stated = false, wenn der Tag mehr oder anderes sagt als der Ausdruck, also aus einer einzelnen Anzeige stammen muss:
- "verheiratet" → kinder = kinderlos: false (nur familienstand = verheiratet stünde im Ausdruck)
- "absolvirter" → bildung = Bergschule: false (welche Schule, sagt der Ausdruck nicht)
- "Deutsch und" → unterrichtsfach = Englisch: false
Bei verstümmelten Ausdrücken zählt nur, was eindeutig erkennbar ist. Im Zweifel false.

Dimensionen:
{dims}

Antworte mit einem JSON-Objekt {{"items": [...]}} mit genau einem Element pro Tag, "i" = Nummer des Tags."""


class VerifyItem(BaseModel):
    i: int
    stated: bool


class VerifyBatch(BaseModel):
    items: list[VerifyItem]


def verification_candidates(dictionary: pd.DataFrame) -> pd.DataFrame:
    """One row per tag whose value does not occur in its phrase; key = "<phrase key>#<tag index>"."""
    rows = []
    for r in dictionary.itertuples():
        for n, t in enumerate(r.tags if r.tags is not None else []):
            if not value_is_grounded(t["value"], r.surface):
                rows.append({"key": f"{r.key}#{n}", "column": r.column, "surface": r.surface, "count": r.count,
                             "context": r.context, "dimension": t["dimension"], "value": t["value"],
                             "detail_raw": t["detail_raw"]})
    cols = ["key", "column", "surface", "count", "context", "dimension", "value", "detail_raw"]
    return pd.DataFrame(rows, columns=cols)


class TagVerifier:
    def __init__(self, vocab: Vocabulary | None = None):
        self.vocab = vocab or Vocabulary()
        self.system = verify_system_prompt(self.vocab)

    def messages(self, batch: pd.DataFrame) -> list[dict]:
        lines = [f'{n}. "{r.surface}" → {r.dimension} = {r.value}' for n, r in enumerate(batch.itertuples(), 1)]
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": "Tags:\n" + "\n".join(lines)}]

    def run(self, client: DHClient, candidates: pd.DataFrame, *, batch_size: int = VERIFY_BATCH_SIZE,
            progress: bool = True) -> tuple[dict, dict]:
        # A tag whose check failed is kept (stated=True), like before the check existed.
        return run_batches(client, candidates, job="requirements_verify", batch_size=batch_size, progress=progress,
                           build_messages=self.messages, result_model=VerifyBatch,
                           empty_item=lambda: VerifyItem(i=1, stated=True))


def apply_verification(dictionary: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Set `verified` on checked tags; a phrase whose tags were all rejected is no longer informative."""
    d = dictionary.copy()
    tags = []
    for key, row_tags in zip(d["key"], d["tags"]):
        if row_tags is None:
            tags.append(None)
            continue
        tags.append([{**t, "verified": results[f"{key}#{n}"].stated if f"{key}#{n}" in results else t["verified"]}
                     for n, t in enumerate(row_tags)])
    d["tags"] = tags
    d["is_informative"] = d["tags"].map(
        lambda t: any(x["verified"] is not False for x in t) if t is not None else None)
    return d


# detail_raw keeps the model's detail before the grounding check, so dropped details can be audited.
# verified: None = value occurs in the phrase (not checked), True/False = result of the context-free check.
TAG = pa.struct([("dimension", pa.string()), ("group", pa.string()), ("value", pa.string()),
                 ("detail", pa.string()), ("detail_raw", pa.string()), ("in_vocab", pa.bool_()),
                 ("verified", pa.bool_())])
DICT_SCHEMA = pa.schema([
    ("key", pa.string()), ("column", pa.string()), ("phrase_key", pa.string()), ("surface", pa.string()),
    ("count", pa.int32()), ("context", pa.string()), ("tags", pa.list_(TAG)), ("is_informative", pa.bool_()),
    ("prompt_version", pa.string()), ("model", pa.string()),
])
AD_REQUIREMENTS_SCHEMA = pa.schema([
    ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()), ("column", pa.string()),
    ("span_start", pa.int32()), ("span_end", pa.int32()), ("phrase", pa.string()),
    ("dimension", pa.string()), ("group", pa.string()), ("value", pa.string()), ("detail", pa.string()),
    ("in_vocab", pa.bool_()),
])


def ad_requirements(spans: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """One row per (phrase mention, tag). spans: ad_id, newspaper, year, column, start, end, phrase."""
    m = spans.assign(key=spans["column"] + "|" + spans["phrase"].map(phrase_key))
    d = dictionary.loc[dictionary["is_informative"].fillna(False), ["key", "tags"]]
    m = m.merge(d, on="key").explode("tags", ignore_index=True)
    keep = pd.Series([t.get("verified") is not False for t in m["tags"]], index=m.index, dtype=bool)
    m = m[keep].reset_index(drop=True)
    out = pd.DataFrame({"ad_id": m["ad_id"], "newspaper": m["newspaper"], "year": m["year"], "column": m["column"],
                        "span_start": m["start"], "span_end": m["end"], "phrase": m["phrase"]})
    for f in ("dimension", "group", "value", "detail", "in_vocab"):
        out[f] = m["tags"].map(lambda t, f=f: t[f])
    return out


def summarize(dictionary: pd.DataFrame, ad_req: pd.DataFrame) -> dict:
    tags = dictionary.explode("tags").dropna(subset=["tags"])
    t = pd.DataFrame(list(tags["tags"])) if len(tags) else pd.DataFrame(columns=["dimension", "value", "in_vocab"])
    return {
        "phrases": len(dictionary),
        "phrases_mapped": int(dictionary["is_informative"].notna().sum()),
        "phrases_with_tags": int(dictionary["is_informative"].fillna(False).sum()),
        "tags_checked_without_context": int(t["verified"].notna().sum()) if "verified" in t else 0,
        "tags_rejected_by_check": int((t["verified"] == False).sum()) if "verified" in t else 0,  # noqa: E712
        "tag_mentions": len(ad_req),
        "ads_with_tags": int(ad_req["ad_id"].nunique()) if len(ad_req) else 0,
        "mentions_by_dimension": ad_req["dimension"].value_counts().to_dict() if len(ad_req) else {},
        "values_outside_closed_lists": {f"{dim}: {val}": int(n) for (dim, val), n in
                                        t.loc[~t["in_vocab"].astype(bool), ["dimension", "value"]]
                                        .value_counts().head(15).items()} if len(t) else {},
    }


def pilot_sample(phrases: pd.DataFrame, per_column: int = 20, seed: int = 0) -> pd.DataFrame:
    """Per column the most frequent half and a random half, sorted like the full run."""
    parts = []
    for _, grp in phrases.groupby("column"):
        top = grp.nlargest(per_column // 2, "count")
        rest = grp.drop(top.index)
        parts += [top, rest.sample(min(per_column - len(top), len(rest)), random_state=seed)]
    return pd.concat(parts).sort_values(["column", "phrase_key"]).reset_index(drop=True)


def format_tags(item) -> str:
    return "; ".join(f"{t.dimension}={t.value}" + (f" ({t.detail})" if t.detail else "") for t in item.tags) or "—"


def format_tag_dicts(tags: list[dict]) -> str:
    return "; ".join(f"{t['dimension']}={t['value']}" + (f" ({t['detail']})" if t["detail"] else "") for t in tags) or "—"


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)

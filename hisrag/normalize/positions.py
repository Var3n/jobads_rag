"""Step 4: position dictionary.

Collects every distinct position form (extracted `positions` spans plus headings, which are
usually the job title of the notice below), sends them to the LLM in small batches with one
context snippet each, and stores one dictionary row per form. The LLM sees each distinct form
once, never each ad, so the cost grows with the vocabulary, not with the corpus.

Per form the dictionary holds zero or more entries (a form like "Schulleiters- und
Lehrerstellen" names two positions, "eine" or "Bewerberinnen" none):

  term         historical title as written, singular nominative, without "-stelle"
               (Unterlehrerin, Wirtschafterin, Commis)
  lemma        gender-neutral base form used for grouping (Unterlehrer, Wirtschafter, Commis)
  modern       modern German equivalent (Commis → Handlungsgehilfe)
  gender_form  gender of the wording itself: m, f, m/f (both named), n (neutral wording)
  category     one of CATEGORIES (coarse, until HISCO codes are added)

`hisco_code` is left empty for the HISCO matching that is being built separately.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, Field

from hisrag.llm.client import DHClient
from hisrag.normalize.text import normalize_text

PROMPT_VERSION = "positions-v1"

CATEGORIES = (
    "Haushalt/Dienstboten",
    "Erziehung/Unterricht",
    "Handel/Verkauf",
    "Büro/Kanzlei (privat)",
    "Öffentliche Verwaltung",
    "Justiz/Recht",
    "Post/Bahn/Verkehr",
    "Handwerk",
    "Industrie/Technik",
    "Land-/Forstwirtschaft",
    "Gesundheit/Pflege",
    "Gastgewerbe",
    "Kunst/Musik/Unterhaltung",
    "Kirche/Religion",
    "Militär/Sicherheit/Aufsicht",
    "Sonstiges",
)
Category = Literal[CATEGORIES]  # type: ignore[valid-type]


class Entry(BaseModel):
    term: str
    lemma: str
    modern: str
    gender_form: Literal["m", "f", "m/f", "n"]
    category: Category


class FormResult(BaseModel):
    i: int = Field(description="Nummer der Form aus der Liste")
    entries: list[Entry] = Field(description="leer, wenn keine Berufs-/Stellenbezeichnung")
    confidence: Literal["high", "medium", "low"]


class BatchResult(BaseModel):
    items: list[FormResult]


SYSTEM_PROMPT = f"""Du normalisierst Berufs- und Stellenbezeichnungen aus historischen Stellenanzeigen der Wiener Zeitung (1850–1950).
Die Formen wurden automatisch extrahiert; manche sind keine Berufsbezeichnungen (Füllwörter, Adjektive, Satzteile).

Für jede Form gibst du eine Liste von Einträgen zurück:
- leer, wenn die Form keine konkrete Berufs- oder Stellenbezeichnung ist ("eine", "mit", "sucht", "Stelle", "Posten", "Bewerberinnen", "Classe", Ortsnamen);
- ein Eintrag pro genanntem Beruf; zwei, wenn zwei verschiedene Berufe genannt sind ("Schulleiters- und Lehrerstellen").
Die Geschlechtsvarianten desselben Berufs ("Lehrer- oder Lehrerinstelle") sind EIN Eintrag mit gender_form "m/f".

Felder eines Eintrags:
- term: die historische Bezeichnung im Nominativ Singular, ohne "-stelle"/"-posten", mit dem Geschlecht der Form. Die Schreibung wird nur modernisiert, wenn es dasselbe Wort bleibt (Wirthschafterin → Wirtschafterin, Secretär → Sekretär); historische Berufswörter bleiben (Commis, Supplent, Kanzlist, Diurnist).
- lemma: die geschlechtsneutrale Grundform für Gruppierungen, in der Regel die männliche Form (Lehrerin → Lehrer, Köchin → Koch, Wirtschafterin → Wirtschafter).
- modern: die heutige deutsche Entsprechung, gleiches Geschlecht wie term (Commis → Handlungsgehilfe, Diurnist → Schreibkraft (Tagelöhner), Gouvernante → Hauslehrerin, Supplent → Vertretungslehrer). Wenn der Beruf heute gleich heißt, dasselbe Wort.
- gender_form: Geschlecht der Wortform selbst: "m", "f", "m/f" (beide genannt) oder "n" (neutral formuliert, z. B. "Lehrstelle", "Lehrkraft").
- category: genau eine von: {", ".join(CATEGORIES)}.

Achtung: In österreichischen Ausschreibungen bedeutet "Lehrstelle" meist eine Stelle als Lehrer (nicht Ausbildungsplatz), "Lehrkanzel" eine Professur. Nutze den Kontext.
confidence: "low", wenn die Form verstümmelt oder mehrdeutig ist.

Beispiele:
- "Unterlehrer-" (Kontext: "Die Unterlehrer-, resp. Unterlehrerinstelle an der Volksschule") → [{{"term": "Unterlehrer", "lemma": "Unterlehrer", "modern": "Grundschullehrer", "gender_form": "m", "category": "Erziehung/Unterricht"}}]
- "Lehrerinstelle" → [{{"term": "Lehrerin", "lemma": "Lehrer", "modern": "Lehrerin", "gender_form": "f", "category": "Erziehung/Unterricht"}}]
- "Wirthschafterin" → [{{"term": "Wirtschafterin", "lemma": "Wirtschafter", "modern": "Hauswirtschafterin", "gender_form": "f", "category": "Haushalt/Dienstboten"}}]
- "Commis" → [{{"term": "Commis", "lemma": "Commis", "modern": "Handlungsgehilfe", "gender_form": "m", "category": "Handel/Verkauf"}}]
- "Kanzlistenstelle" → [{{"term": "Kanzlist", "lemma": "Kanzlist", "modern": "Verwaltungsangestellter", "gender_form": "m", "category": "Öffentliche Verwaltung"}}]
- "Bedienter" → [{{"term": "Bedienter", "lemma": "Bedienter", "modern": "Hausdiener", "gender_form": "m", "category": "Haushalt/Dienstboten"}}]
- "eine" → []

Antworte mit einem JSON-Objekt {{"items": [...]}} mit genau einem Element pro Form, "i" = Nummer der Form."""


def form_key(text: str) -> str:
    """Grouping key for surface forms: normalized, lowercase, outer punctuation stripped."""
    key = normalize_text(text).lower()
    key = re.sub(r"^[^\wäöü]+|[^\wäöü]+$", "", key)
    key = re.sub(r"^(?:\d+[.)]?|[a-z]\))\s+", "", key)  # enumerations: "1. Lehrstelle", "2) ...", "a) ..."
    return re.sub(r"\s+", " ", key)


def _snippet(text: str, start: int, end: int, width: int = 70) -> str:
    left = normalize_text(text[max(0, start - width):start])
    right = normalize_text(text[end:end + width])
    return f"…{left} [{normalize_text(text[start:end])}] {right}…"


def collect_forms(spans: pd.DataFrame, headings: pd.DataFrame) -> pd.DataFrame:
    """One row per form key with frequency, the most common surface form and one context.

    spans: ad_id, text, start, end, form, gender   (one row per extracted position span)
    headings: ad_id, heading, ad_text               (heading text and the start of its ad)
    """
    rows = []
    spans = spans.assign(key=spans["form"].map(form_key))
    for key, grp in spans.groupby("key", sort=False):
        ex = grp.iloc[len(grp) // 2]
        rows.append({
            "key": key, "n_spans": len(grp), "n_headings": 0,
            "surface": normalize_text(Counter(grp["form"]).most_common(1)[0][0]).strip(" .,;:-"),
            "context": _snippet(ex["text"], ex["start"], ex["end"]),
            "extracted_gender": json.dumps(dict(Counter(grp["gender"].fillna("unknown")))),
        })
    forms = pd.DataFrame(rows).set_index("key")

    headings = headings.assign(key=headings["heading"].map(form_key))
    for key, grp in headings.groupby("key", sort=False):
        if key in forms.index:
            forms.loc[key, "n_headings"] = len(grp)
            continue
        ex = grp.iloc[0]
        ctx = normalize_text(ex["ad_text"])[:140] if isinstance(ex["ad_text"], str) else ""
        forms.loc[key] = {"n_spans": 0, "n_headings": len(grp), "surface": normalize_text(ex["heading"]).strip(" .,;:-"),
                          "context": f"Überschrift über: {ctx}…" if ctx else "Überschrift", "extracted_gender": "{}"}
    forms = forms[forms.index != ""]
    forms["count"] = forms["n_spans"] + forms["n_headings"]
    return forms.reset_index().sort_values("key").reset_index(drop=True)


def _messages(batch: pd.DataFrame) -> list[dict]:
    lines = [f'{n}. "{r.surface}" — Kontext: {r.context}' for n, r in enumerate(batch.itertuples(), 1)]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Formen:\n" + "\n".join(lines)}]


def normalize_batch(client: DHClient, batch: pd.DataFrame, *, thinking: bool = False) -> dict[str, FormResult]:
    """Normalize one batch; items the model skipped are retried one by one."""
    result = client.chat_json(_messages(batch), BatchResult, thinking=thinking,
                              max_tokens=16000 if thinking else 6000)
    by_i = {item.i: item for item in result.items if 1 <= item.i <= len(batch)}
    out = {}
    for n, key in enumerate(batch["key"], 1):
        if n in by_i:
            out[key] = by_i[n]
        else:
            single = client.chat_json(_messages(batch.iloc[[n - 1]]), BatchResult, thinking=thinking,
                                      max_tokens=4000)
            out[key] = single.items[0] if single.items else FormResult(i=1, entries=[], confidence="low")
    return out


def run(client: DHClient, forms: pd.DataFrame, *, batch_size: int = 25, thinking: bool = False,
        progress: bool = True) -> tuple[dict[str, FormResult], dict]:
    batches = [forms.iloc[i:i + batch_size] for i in range(0, len(forms), batch_size)]
    t0 = time.monotonic()
    with client.job_scope(f"positions{'-thinking' if thinking else ''}"):
        parts = client.map(lambda b: normalize_batch(client, b, thinking=thinking), batches,
                           desc="positions" if progress else None, return_exceptions=True)
    results, failed = {}, []
    for batch, part in zip(batches, parts):
        if isinstance(part, Exception):
            failed.append(f"{batch['key'].iloc[0]}…: {type(part).__name__}: {str(part)[:200]}")
        else:
            results.update(part)
    stats = {"forms": len(forms), "batches": len(batches), "failed_batches": failed,
             "seconds": round(time.monotonic() - t0, 1)}
    return results, stats


DICT_SCHEMA = pa.schema([
    ("key", pa.string()), ("surface", pa.string()), ("count", pa.int32()), ("n_spans", pa.int32()),
    ("n_headings", pa.int32()), ("context", pa.string()), ("extracted_gender", pa.string()),
    ("is_position", pa.bool_()), ("confidence", pa.string()),
    ("entries", pa.list_(pa.struct([("term", pa.string()), ("lemma", pa.string()), ("modern", pa.string()),
                                    ("gender_form", pa.string()), ("category", pa.string())]))),
    ("hisco_code", pa.string()), ("prompt_version", pa.string()), ("model", pa.string()),
])


def to_dictionary(forms: pd.DataFrame, results: dict[str, FormResult], model: str) -> pd.DataFrame:
    d = forms.copy()
    d["entries"] = [[e.model_dump() for e in results[k].entries] if k in results else None for k in d["key"]]
    d["confidence"] = [results[k].confidence if k in results else None for k in d["key"]]
    d["is_position"] = d["entries"].map(lambda e: bool(e) if e is not None else None)
    d["hisco_code"] = None
    d["prompt_version"] = PROMPT_VERSION
    d["model"] = model
    return d[DICT_SCHEMA.names]


def consistency_report(dictionary: pd.DataFrame) -> dict:
    """Lemmas that received different categories in different batches."""
    e = dictionary.explode("entries").dropna(subset=["entries"])
    if e.empty:
        return {"lemmas": 0, "lemmas_with_conflicting_category": 0, "examples": {}}
    e = pd.DataFrame({"lemma": e["entries"].map(lambda x: x["lemma"]), "category": e["entries"].map(lambda x: x["category"]),
                      "count": e["count"]})
    cats = e.groupby("lemma")["category"].nunique()
    conflicts = cats[cats > 1].index
    examples = {lem: e[e.lemma == lem].groupby("category")["count"].sum().to_dict() for lem in list(conflicts)[:10]}
    return {"lemmas": int(len(cats)), "lemmas_with_conflicting_category": int(len(conflicts)), "examples": examples}


def pilot_sample(forms: pd.DataFrame, n_frequent: int = 40, n_random: int = 40, seed: int = 0) -> pd.DataFrame:
    """The most frequent forms plus a random sample of the rest, sorted by key like the full run."""
    frequent = forms.nlargest(n_frequent, "count")
    rest = forms.drop(frequent.index)
    rnd = rest.sample(min(n_random, len(rest)), random_state=seed)
    return pd.concat([frequent, rnd]).sort_values("key").reset_index(drop=True)


def compare_runs(a: dict[str, FormResult], b: dict[str, FormResult]) -> dict:
    """Agreement between two runs on the same forms (e.g. without and with reasoning)."""
    keys = [k for k in a if k in b]
    lem = lambda r: sorted(e.lemma.lower() for e in r.entries)
    cat = lambda r: sorted(e.category for e in r.entries)
    return {
        "forms": len(keys),
        "same_is_position": round(sum(bool(a[k].entries) == bool(b[k].entries) for k in keys) / max(len(keys), 1), 3),
        "same_lemmas": round(sum(lem(a[k]) == lem(b[k]) for k in keys) / max(len(keys), 1), 3),
        "same_categories": round(sum(cat(a[k]) == cat(b[k]) for k in keys) / max(len(keys), 1), 3),
    }


AD_POSITIONS_SCHEMA = pa.schema([
    ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()), ("source", pa.string()),
    ("span_start", pa.int32()), ("span_end", pa.int32()), ("surface", pa.string()), ("key", pa.string()),
    ("extracted_gender", pa.string()), ("term", pa.string()), ("lemma", pa.string()), ("modern", pa.string()),
    ("gender_form", pa.string()), ("category", pa.string()), ("confidence", pa.string()),
])


def ad_positions(spans: pd.DataFrame, headings: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """One row per (position mention, dictionary entry) for mentions that name a position.

    spans: ad_id, newspaper, year, start, end, form, gender;  headings: ad_id, newspaper, year, heading.
    Heading mentions are attached to the ad below the heading, with source "heading".
    """
    s = pd.DataFrame({"ad_id": spans["ad_id"], "newspaper": spans["newspaper"], "year": spans["year"],
                      "source": "span", "span_start": spans["start"], "span_end": spans["end"],
                      "surface": spans["form"], "extracted_gender": spans["gender"]})
    h = pd.DataFrame({"ad_id": headings["ad_id"], "newspaper": headings["newspaper"], "year": headings["year"],
                      "source": "heading", "span_start": None, "span_end": None,
                      "surface": headings["heading"], "extracted_gender": None})
    mentions = pd.concat([s, h], ignore_index=True)
    mentions["key"] = mentions["surface"].map(form_key)
    d = dictionary.loc[dictionary["is_position"].fillna(False), ["key", "entries", "confidence"]]
    m = mentions.merge(d, on="key").explode("entries", ignore_index=True)
    for f in ("term", "lemma", "modern", "gender_form", "category"):
        m[f] = m["entries"].map(lambda e, f=f: e[f])
    return m[AD_POSITIONS_SCHEMA.names]

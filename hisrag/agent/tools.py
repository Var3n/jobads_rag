"""Step 10: the tools the agent (step 11) calls, as OpenAI function specs plus their implementations.

  search_ads      semantic or keyword search in the index (step 9), with filters; short hit records
  get_ad          full records of up to 10 ads: text, positions, requirements, pay, printings, clipping link
  aggregate       SQL templates over ad_clean: count, share or pay per group; counts only countable ads
  expand_concept  looks a word up in the dictionaries of steps 4–5: lemmas, historical spellings, categories,
                  requirement tags, each with its number of ads, so filters can be exact

Every result is a JSON-able dict; errors come back as {"error": …} so the model can correct its call.
Descriptions are German like the agent's prompt. Arguments are validated with pydantic; the specs are
plain JSON schema (no anyOf/null, no titles), which Qwen's tool calling handles reliably.
"""

from __future__ import annotations

import json
import re
import threading
from functools import cached_property
from typing import Any, Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from hisrag.config import Config, load_config
from hisrag.data import connect
from hisrag.index import AdIndex, Filters
from hisrag.index.build import REQUIREMENT_TAGS_SQL
from hisrag.normalize.clean import BENEFITS
from hisrag.normalize.positions import CATEGORIES
from hisrag.normalize.text import fold_spelling

LABELS = ("job_offer", "job_search", "service_offer", "vermittlung")
FULL_TEXT_CHARS = 4000  # get_ad's safeguard; the longest ad has ~2,400
MAX_GROUPS = 40       # largest groups kept for lemmas, values, …; years and decades are always complete
COUNTABLE_NOTE = ("Gezählt werden nur zählbare Anzeigen: je Anzeige ein Abdruck (Wiederholungen nicht), ohne Texte, "
                  "die die Nachkorrektur stark verändert oder erfunden hat.")

# ------------------------------------------------------------------ arguments


class FilterArgs(BaseModel):
    """Restrictions shared by search and aggregate; mapped to hisrag.index.Filters."""
    year_from: int | None = Field(None, description="frühestes Jahr (einschließlich)")
    year_to: int | None = Field(None, description="spätestes Jahr (einschließlich)")
    labels: list[Literal[LABELS]] = Field(  # type: ignore[valid-type]
        [], description="Art der Anzeige: job_offer = Stellenangebot, job_search = Stellengesuch, "
                        "service_offer = Dienstleistungsangebot, vermittlung = Stellenvermittlung")
    position_categories: list[Literal[CATEGORIES]] = Field(  # type: ignore[valid-type]
        [], description="Berufsfeld der genannten Stelle (eines davon)")
    position_lemmas: list[str] = Field(
        [], description="Berufe als Lemma (männliche Grundform, z. B. 'Koch' für Koch und Köchin), wie "
                        "expand_concept sie liefert (einer davon)")
    position_gender: list[Literal["f", "m", "mixed", "neutral"]] = Field(
        [], description="grammatisches Geschlecht der Berufsbezeichnung in der Anzeige")
    requirement_tags: list[str] = Field(
        [], description="Anforderungen als 'dimension:wert', z. B. 'sprachkenntnisse:Böhmisch', wie expand_concept "
                        "sie liefert (eine davon)")
    has_pay: bool | None = Field(None, description="true: nur Anzeigen mit Lohnangabe; false: nur ohne")
    benefits: list[Literal[BENEFITS]] = Field(  # type: ignore[valid-type]
        [], description="Naturalleistungen, die alle genannt sein müssen (wohnung, kost, …)")

    def filters(self, **extra) -> Filters:
        return Filters(year_from=self.year_from, year_to=self.year_to, labels=self.labels or None,
                       position_categories=self.position_categories or None,
                       position_lemmas=self.position_lemmas or None, position_gender=self.position_gender or None,
                       requirement_tags=self.requirement_tags or None, has_pay=self.has_pay,
                       benefits=list(self.benefits), **extra)


class AggFilterArgs(FilterArgs):
    keyword: str | None = Field(None, description="nur Anzeigen, die diese Wörter enthalten (Syntax wie search_ads "
                                                  "im Modus keyword)")


class SearchArgs(BaseModel):
    query: str = Field(description="semantic: Frage oder Thema in heutigem Deutsch; keyword: genaue Wörter")
    mode: Literal["semantic", "keyword"] = Field(
        "semantic", description="semantic findet Anzeigen nach Bedeutung, auch in historischer Wortwahl; keyword "
                                "findet genau diese Wörter (Namen, Orte, feste Begriffe) in jeder historischen "
                                "Schreibung. keyword-Syntax: alle Wörter müssen vorkommen, \"…\" ist eine Phrase, "
                                "wort* ein Wortanfang (Krakau* für Krakau, Krakauer), -wort schließt aus")
    filters: FilterArgs = Field(default_factory=FilterArgs)
    k: int = Field(10, ge=1, le=25, description="Anzahl der Treffer (höchstens 25)")


class GetAdArgs(BaseModel):
    ad_ids: list[str] = Field(min_length=1, max_length=10, description="IDs aus search_ads (höchstens 10)")


class AggregateArgs(BaseModel):
    measure: Literal["count", "share", "pay"] = Field(
        "count", description="count: Zahl der Anzeigen je Gruppe; share: Anteil (in Prozent) der Anzeigen aus "
                             "'filters', die zusätzlich 'subset' erfüllen; pay: Hauptlohn (Quartile) je Gruppe, getrennt nach "
                             "Währung, Währungsstandard und Zeitraum, nominal")
    group_by: Literal["none", "decade", "year", "label", "newspaper", "position_category", "position_lemma",
                      "position_modern", "position_gender", "requirement_dimension", "requirement_value",
                      "benefit"] = Field(
        "decade", description="Gruppierung. Bei Listenfeldern (Berufe, Anforderungen, Leistungen) zählt eine Anzeige "
                              "in jeder ihrer Gruppen")
    dimension: str | None = Field(None, description="nur bei group_by=requirement_value: die Anforderungsdimension, "
                                                    "z. B. sprachkenntnisse")
    filters: AggFilterArgs = Field(default_factory=AggFilterArgs, description="die Grundmenge der Anzeigen")
    subset: AggFilterArgs | None = Field(None, description="nur bei share: die zusätzliche Bedingung")


class ExpandArgs(BaseModel):
    term: str = Field(description="ein Beruf, eine Anforderung oder ein Wort, heutig oder historisch")


# ------------------------------------------------------------------ JSON schema for the specs


def _clean_schema(node: Any, defs: dict) -> Any:
    """Inline $refs, turn `X | None` (anyOf with null) into X, drop titles."""
    if isinstance(node, list):
        return [_clean_schema(n, defs) for n in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return _clean_schema(defs[node["$ref"].split("/")[-1]], defs) | (
            {"description": node["description"]} if "description" in node else {})
    if "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(options) == 1:
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            return _clean_schema(options[0] | rest, defs)
    return {k: _clean_schema(v, defs) for k, v in node.items() if k not in ("title", "$defs")}


def json_schema(model: type[BaseModel]) -> dict:
    s = model.model_json_schema()
    out = _clean_schema(s, s.get("$defs", {}))
    out.pop("default", None)
    return out


SPECS = {
    "search_ads": (SearchArgs, "Sucht Anzeigen und liefert die Treffer mit vollständigem Text (ID, Datum, Art, Beruf, Text). Jeder "
                               "Treffer ist eine Anzeige; Wiederholungsabdrucke sind zusammengefasst (printings). Im "
                               "Modus keyword wird auch die Gesamtzahl der Treffer gemeldet."),
    "get_ad": (GetAdArgs, "Liefert vollständige Anzeigen: Text, Berufe, Anforderungen, Lohn, Abdrucke, Link zum Bild "
                          "des Originals."),
    "aggregate": (AggregateArgs, "Zählt Anzeigen, berechnet Anteile oder Lohnstatistiken je Gruppe (z. B. Jahrzehnt). "
                                 + COUNTABLE_NOTE),
    "expand_concept": (ExpandArgs, "Schlägt ein Wort in den Wörterbüchern der Berufe und Anforderungen nach: Lemmata, "
                                   "historische Schreibungen, Berufsfelder und Anforderungs-Tags mit der Zahl ihrer "
                                   "Anzeigen. Vor Filtern auf Berufe oder Anforderungen aufrufen."),
}


# ------------------------------------------------------------------ the tools


def _date(x) -> str | None:
    return None if x is None or pd.isna(x) else str(x)[:10]


def _list(x) -> list:
    return [] if x is None or (not hasattr(x, "__len__") and pd.isna(x)) else list(x)


def _num(x):
    return None if x is None or pd.isna(x) else round(float(x), 2)


class Tools:
    def __init__(self, cfg: Config | None = None, index: AdIndex | None = None, client=None):
        self.cfg = cfg or load_config()
        self.index = index or AdIndex(self.cfg, client)
        self.con = connect(self.cfg)
        self._lock = threading.Lock()  # one DuckDB connection; the agent pilot asks questions in parallel
        self.con.execute(f"CREATE TEMP VIEW base AS SELECT *, {REQUIREMENT_TAGS_SQL} AS requirement_tags "
                         "FROM ad_clean WHERE searchable")

    def specs(self) -> list[dict]:
        return [{"type": "function", "function": {"name": name, "description": desc, "parameters": json_schema(model)}}
                for name, (model, desc) in SPECS.items()]

    def call(self, name: str, arguments: str | dict) -> dict:
        """Run a tool call from the model; any problem is returned as {"error": …} for the model to read."""
        if name not in SPECS:
            return {"error": f"unbekanntes Tool {name!r}; vorhanden: {', '.join(SPECS)}"}
        try:
            args = SPECS[name][0].model_validate(json.loads(arguments) if isinstance(arguments, str) else arguments)
        except json.JSONDecodeError as exc:
            return {"error": f"ungültige Argumente: kein JSON ({exc})"}
        except ValidationError as exc:
            return {"error": "ungültige Argumente: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_url=False))}
        try:
            with self._lock:
                return getattr(self, name)(args)
        except (ValueError, duckdb.Error) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -------------------------------------------------------------- search_ads

    def search_ads(self, a: SearchArgs) -> dict:
        f = a.filters.filters()
        out: dict = {"mode": a.mode, "query": a.query}
        if a.mode == "semantic":
            hits = self.index.semantic(a.query, f, k=a.k)
        else:
            hits = self.index.keyword(a.query, f, k=None)
            out["total_matches"] = len(hits)
            out["total_countable"] = int(hits["countable"].sum())
            if hits.attrs.get("expanded"):
                out["expanded"] = {k: v[:20] for k, v in hits.attrs["expanded"].items()}
            hits = hits.head(a.k)
        out["results"] = [self._hit(r) for r in hits.itertuples()]
        return out

    @staticmethod
    def _hit(r) -> dict:
        hit = {"ad_id": r.ad_id, "date": _date(r.date), "label": r.label, "positions": _list(r.position_modern),
               "text": r.text, "printings": int(r.dup_cluster_size)}
        if isinstance(r.quality_warning, str):
            hit["warning"] = r.quality_warning
        return hit

    # -------------------------------------------------------------- get_ad

    def get_ad(self, a: GetAdArgs) -> dict:
        rows = self.con.execute("SELECT * FROM ad_clean WHERE ad_id IN (SELECT unnest(?))", [a.ad_ids]).df()
        by_id = {r["ad_id"]: r for _, r in rows.iterrows()}
        return {"ads": [self._full(by_id[i]) if i in by_id else {"ad_id": i, "error": "nicht gefunden"}
                        for i in a.ad_ids]}

    @staticmethod
    def _full(r) -> dict:
        text = (f"{r['heading_text']}\n" if isinstance(r["heading_text"], str) else "") + (r["text_norm"] or "")
        ad = {"ad_id": r["ad_id"], "newspaper": r["newspaper"], "date": _date(r["date"]), "page": int(r["page"]),
              "label": r["label"], "lang": r["lang"],
              "text": text if len(text) <= FULL_TEXT_CHARS else text[:FULL_TEXT_CHARS] + " …",
              "positions_historical": _list(r["position_terms"]), "positions_modern": _list(r["position_modern"]),
              "position_categories": _list(r["position_categories"]),
              "requirements": [{k: v for k, v in t.items() if k in ("dimension", "value", "detail") and v}
                               for t in _list(r["requirements"])],
              "benefits": [b for b in BENEFITS if r[f"benefit_{b}"]],
              "printings": {"count": int(r["dup_cluster_size"]), "first": _date(r["run_first_date"]),
                            "last": _date(r["run_last_date"])},
              "countable": bool(r["countable"]), "searchable": bool(r["searchable"]), "image": r["iiif_link"]}
        if not pd.isna(r["pay_min"]):
            ad["pay"] = {"min": _num(r["pay_min"]), "max": _num(r["pay_max"]), "currency": r["pay_currency"],
                         "standard": r["pay_standard"], "period": r["pay_period"]}
        ad["amounts"] = [{k: (_num(v) if k.startswith("amount") else v) for k, v in s.items() if v is not None}
                         for s in _list(r["salary_amounts"])]
        if isinstance(r["quality_warning"], str):
            ad["warning"] = r["quality_warning"]
        return ad

    # -------------------------------------------------------------- aggregate

    GROUPS = {
        "none": "'alle'", "decade": "decade", "year": "year", "label": "label", "newspaper": "newspaper",
        "position_category": "unnest(position_categories)", "position_lemma": "unnest(position_lemmas)",
        "position_modern": "unnest(position_modern)", "position_gender": "position_gender",
        "requirement_dimension": "unnest(requirement_dimensions)",
        "requirement_value": "unnest(list_filter(requirement_tags, x -> starts_with(x, $dim || ':')))",
        "benefit": "unnest(list_filter([" + ", ".join(f"CASE WHEN benefit_{b} THEN '{b}' END" for b in BENEFITS)
                   + "], x -> x IS NOT NULL))",
    }
    ORDERED = {"decade", "year"}  # sorted by group; the others by size

    def _where(self, f: AggFilterArgs, name: str) -> str:
        """SQL condition for one filter set; a keyword condition becomes the set of matching printing clusters."""
        parts = [f.filters().sql() or "true"]
        if f.keyword:
            clusters = self.index.keyword(f.keyword, k=None)["dup_cluster_id"]
            self.con.register(name, pd.DataFrame({"c": clusters.astype(str)}))
            parts.append(f"dup_cluster_id IN (SELECT c FROM {name})")
        return " AND ".join(f"({p})" for p in parts)

    def aggregate(self, a: AggregateArgs) -> dict:
        if a.group_by == "requirement_value" and not a.dimension:
            raise ValueError("group_by=requirement_value braucht 'dimension'")
        if a.measure == "share" and a.subset is None:
            raise ValueError("measure=share braucht 'subset'")
        group = self.GROUPS[a.group_by]
        where = self._where(a.filters, "kw_base") + " AND countable"
        params = {"dim": a.dimension} if "$dim" in group else {}
        sub = f"SELECT *, {group} AS g FROM base WHERE {where}"
        n_ads = self.con.execute(f"SELECT count(*) FROM base WHERE {where}").fetchone()[0]
        out: dict = {"measure": a.measure, "group_by": a.group_by, "n_ads": int(n_ads)}
        if a.measure == "count":
            sql = f"SELECT g, count(DISTINCT ad_id) AS n FROM ({sub}) GROUP BY g"
        elif a.measure == "share":
            cond = self._where(a.subset, "kw_subset")
            sql = (f"SELECT g, count(DISTINCT ad_id) AS n_base, count(DISTINCT ad_id) FILTER (WHERE {cond}) AS n, "
                   f"round(100 * n / n_base, 2) AS share_pct FROM ({sub}) GROUP BY g")
            out["n_subset"] = int(self.con.execute(f"SELECT count(*) FROM base WHERE {where} AND {cond}").fetchone()[0])
        else:
            sql = (f"SELECT g, pay_currency AS currency, pay_standard AS standard, pay_period AS period, "
                   f"count(DISTINCT ad_id) AS n, quantile_cont(m, 0.25) AS p25, median(m) AS median, "
                   f"quantile_cont(m, 0.75) AS p75, min(m) AS min, max(m) AS max "
                   f"FROM (SELECT *, (pay_min + pay_max) / 2 AS m FROM ({sub}) WHERE pay_min IS NOT NULL) "
                   f"GROUP BY ALL")
            out["n_ads_with_pay"] = int(self.con.execute(
                f"SELECT count(*) FROM base WHERE {where} AND pay_min IS NOT NULL").fetchone()[0])
            out["note"] = ("Hauptlohn je Anzeige (Mitte einer Spanne), nominal; Beträge verschiedener Währungen, "
                           "Standards (CM, ö.W.) und Zeiträume nie vermischen.")
        order = "g" if a.group_by in self.ORDERED else ("n DESC, g" if a.measure != "share" else "n_base DESC, g")
        rows = self.con.execute(f"SELECT * FROM ({sql}) ORDER BY {order}", params).df()
        if a.group_by == "requirement_value":
            rows["g"] = rows["g"].str.split(":", n=1).str[1]
        out["groups_total"] = len(rows)
        if len(rows) > MAX_GROUPS and a.group_by not in self.ORDERED:  # a time series is never cut
            out["groups_omitted"] = len(rows) - MAX_GROUPS
            rows = rows.head(MAX_GROUPS)
        out["rows"] = [{("group" if k == "g" else k): (_num(v) if isinstance(v, float) else
                                                       (int(v) if hasattr(v, "dtype") and v.dtype.kind in "iu" else v))
                        for k, v in r.items()} for r in rows.to_dict("records")]
        if a.group_by.startswith(("position_category", "position_lemma", "position_modern", "requirement", "benefit")):
            out["note_groups"] = "Eine Anzeige kann in mehreren Gruppen stehen; Anzeigen ohne Wert fehlen."
        out["note"] = out.get("note", "") + (" " if "note" in out else "") + COUNTABLE_NOTE
        return out

    # -------------------------------------------------------------- expand_concept

    @cached_property
    def _positions(self) -> pd.DataFrame:
        d = self.con.execute("""SELECT p.lemma, p.modern, p.term, p.category, p.surface, count(DISTINCT p.ad_id) AS n
                                FROM ad_positions p JOIN base c USING (ad_id) WHERE c.countable GROUP BY ALL""").df()
        for col in ("lemma", "modern", "term"):
            d[f"f_{col}"] = d[col].fillna("").map(fold_spelling)
        return d

    @cached_property
    def _requirements(self) -> pd.DataFrame:
        d = self.con.execute("""SELECT r.dimension, r.value, count(DISTINCT r.ad_id) AS n FROM ad_requirements r
                                JOIN base c USING (ad_id) WHERE c.countable AND r.value IS NOT NULL
                                GROUP BY ALL""").df()
        d["f_value"] = d["value"].map(fold_spelling)
        return d

    def expand_concept(self, a: ExpandArgs) -> dict:
        words = re.findall(r"[^\W_]+", fold_spelling(a.term))
        if not words:
            raise ValueError("leerer Begriff")
        hit = lambda s: s.map(lambda x: all(w in x for w in words))

        p = self._positions
        m = p[hit(p["f_lemma"]) | hit(p["f_modern"]) | hit(p["f_term"])]
        lemmas = []
        if len(m):
            # ads per lemma counted once, not once per spelling; take the counts from ad level
            n_by_lemma = self.con.execute(
                """SELECT p.lemma, count(DISTINCT p.ad_id) FROM ad_positions p JOIN base c USING (ad_id)
                   WHERE c.countable AND p.lemma IN (SELECT unnest(?)) GROUP BY 1""",
                [m["lemma"].dropna().unique().tolist()]).fetchall()
            for lemma, n in sorted(n_by_lemma, key=lambda x: -x[1])[:15]:
                g = p[p["lemma"] == lemma]
                spellings = g.groupby("surface")["n"].sum().nlargest(8).index.tolist()
                lemmas.append({"lemma": lemma, "n_ads": int(n),
                               "modern": g.groupby("modern")["n"].sum().nlargest(4).index.tolist(),
                               "categories": g.groupby("category")["n"].sum().nlargest(3).index.tolist(),
                               "spellings_in_ads": spellings})
        r = self._requirements
        reqs = r[hit(r["f_value"])].nlargest(15, "n")
        cats = [c for c in CATEGORIES if any(w in fold_spelling(c) for w in words)]
        return {"term": a.term, "positions": lemmas,
                "requirements": [{"tag": f"{d}:{v}", "n_ads": int(n)} for d, v, n in
                                 zip(reqs["dimension"], reqs["value"], reqs["n"])],
                "categories": cats,
                "hint": "Filter: position_lemmas = Lemma, requirement_tags = Tag, position_categories = Berufsfeld; "
                        "für search_ads im Modus keyword die Schreibungen mit * verwenden."
                        if (lemmas or reqs.size or cats) else
                        f"Kein Eintrag für {a.term!r}; mit einem anderen oder allgemeineren Wort versuchen oder "
                        "search_ads im Modus semantic verwenden."}

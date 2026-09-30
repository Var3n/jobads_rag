"""Step 6: salary amounts and benefits.

The extraction marks amounts ("600 fl.", "1200 K", "von 400 fl.") in the `salary` column, but what an
amount pays for (Gehalt, Zulage, Quartiergeld, Kaution …), its period and the currency standard
(Conventionsmünze or österreichische Währung) are mostly written around it: "Jahresgehalt 735 fl. ö. W.",
"42 fl. Quartiergeld", "Caution von 400 fl.". Rules read a window of the ad text around each span;
the few spans they cannot read (numbers in words, currency missing) go to the LLM.

Amounts stay nominal. Gulden amounts get a standard: stated in the text, or by date (CM until
October 1858, ö. W. from November 1858).
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, Field

from hisrag.llm.client import DHClient
from hisrag.normalize.batching import run_batches
from hisrag.normalize.text import normalize_text

OEW_FROM = dt.date(1858, 11, 1)
BEFORE, AFTER = 80, 60  # characters of ad text read around a span

# ------------------------------------------------------------------ amount and currency

_NUM = r"\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d{1,2}|\s?½)?"
_DASH = r"\s*(?:—|–|-|bis)\s*"
_CURRENCY = [  # (regex on lower-case text, code); order matters: "kr." before "k"
    (r"fl\b\.?|flor\b\.?|gulden|gld\.?", "fl"),
    (r"kronen|krone|k\b\.?", "K"),
    (r"schilling|s\b\.?", "S"),
    (r"reichsmark|rm\b\.?", "RM"),
    (r"mark|mk\b\.?", "M"),
    (r"francs|franken|fr\b\.?", "Fr"),
]
_SUBUNIT = r"kr\b\.?|kreuzer|h\b\.?|heller|gr\b\.?|groschen|pf\b\.?|rpf\b\.?"
_CUR_RE = "|".join(f"(?:{rx})" for rx, _ in _CURRENCY)
_OEW = r"ö\.?\s*w|öst\w*\.?\s*w|oe\.?\s*w|[56]\.\s*w\b|österr\w*\.?\s*währ"
_CM = r"[ce]\.\s*m\b|[ce]m\b|conv\w*\.?\s*m|conventionsm"  # "EM." is OCR for "CM."
# a number followed by one of these is no amount of money: a bread ration, a share of the salary, hours
NOT_MONEY = (r"g\b|gramm|dekagramm|decagramm|dkg|kg|kilo|pfd|pfund|liter|hectol|hektol|raummeter|klafter|metzen|joch|"
             r"prozent|procent|perc|pct|p\.\s*ct|%|stunden|std\b|tage\b|wochen\b|monate\b|jahre\b|classe|klasse")


def _number(s: str) -> float:
    s = s.strip()
    if s.endswith("½"):
        return float(s[:-1] or 0) + 0.5
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):  # 1.200 = twelve hundred
        return float(s.replace(".", ""))
    return float(s.replace(",", "."))


def _currency_code(token: str) -> str | None:
    for rx, code in _CURRENCY:
        if re.fullmatch(rx, token.strip()):
            return code
    return None


class Amount(BaseModel):
    amount_min: float | None
    amount_max: float | None
    currency: str | None
    standard: str | None = None          # "CM" / "öW" for Gulden
    standard_source: str | None = None   # "stated" / "date"


def _standard(rest: str, date: dt.date | None) -> tuple[str | None, str | None]:
    if re.match(rf"\W*(?:\d+\s*(?:{_SUBUNIT})\s*)?(?:{_OEW})", rest):
        return "öW", "stated"
    if re.match(rf"\W*(?:\d+\s*(?:{_SUBUNIT})\s*)?(?:{_CM})", rest):
        return "CM", "stated"
    if date is not None:
        return ("CM" if date < OEW_FROM else "öW"), "date"
    return None, None


def is_fragment(span: str, before: str) -> bool:
    """"2 K [20 h]", "367 fl. [50 kr.]": the Kreuzer/Heller part of an amount already read from the span before."""
    return bool(re.fullmatch(rf"\W*\d+\s*(?:{_SUBUNIT})\W*", normalize_text(span).lower())
                and re.search(rf"\d\s*(?:{_CUR_RE})\W*$", normalize_text(before).lower()))


def is_not_money(text: str) -> bool:
    """"840 Gramm", "25 pCt.", "20 Stunden": a number, but no amount of money."""
    return bool(re.match(rf"\W*(?:von\s+|je\s+)?(?:{_NUM})\s*(?:{NOT_MONEY})", normalize_text(text).lower()))


def parse_amount(text: str, date: dt.date | None, before: str = "") -> Amount | None:
    """Amount at the start of `text` (the span followed by the ad text after it), or None.
    `before` is the text just before the span, for a currency written first ("Lohn Fr. [160—]")."""
    t = normalize_text(text).lower()
    num = rf"({_NUM})(?:\s*[—–-](?!\s*\d))?"  # "160—" = 160
    # currency first: "fl. 9.—", "K 1200", "Fr. 160— bis 180—"
    m = re.match(rf"\W*(?:von\s+|mit\s+)?({_CUR_RE})\s*{num}(?:{_DASH}{num})?", t)
    if m:
        cur, lo, hi, rest = _currency_code(m.group(1)), m.group(2), m.group(3), t[m.end():]
    else:
        m = re.match(rf"\W*(?:(?:von|mit|per|pr\.|zu|à|je|circa|ca\.|etwa)\s+)*{num}(?:{_DASH}{num})?", t)
        if not m:
            return None
        lo, hi, rest = m.group(1), m.group(2), t[m.end():]
        c = re.match(rf"\s*({_CUR_RE})", rest)
        if c:
            cur, rest = _currency_code(c.group(1)), rest[c.end():]
        elif k := re.match(rf"\s*({_SUBUNIT})", rest):
            # Kreuzer or Heller alone: "Taggeld 78½ kr." → 0.785 fl., "70 bis 80 kr." → 0.70–0.80 fl.
            # 100 or more would have been written in Gulden/Kronen: OCR for "K" ("400 h"), left to the LLM
            if _number(hi or lo) >= 100:
                return None
            cur = "K" if k.group(1).startswith("h") else "fl"
            standard, source = _standard(rest[k.end():], date) if cur == "fl" else (None, None)
            per = 60 if standard == "CM" else 100
            return Amount(amount_min=_number(lo) / per, amount_max=_number(hi or lo) / per, currency=cur,
                          standard=standard, standard_source=source)
        else:  # a list sharing its currency ("50, resp. 40 fl.", "500 fl., 450 fl."), or written before the span
            c = re.search(rf"\d\s*({_CUR_RE})", rest[:25])
            b = re.search(rf"(?:^|[\s(])({_CUR_RE})\s*$", normalize_text(before).lower())
            cur = _currency_code(c.group(1)) if c else _currency_code(b.group(1)) if b else None
    amount_min = _number(lo)
    amount_max = _number(hi) if hi else amount_min
    standard, source = _standard(rest, date) if cur == "fl" else (None, None)
    sub = re.match(rf"\s*,?\s*(\d{{1,2}})\s*(?:{_SUBUNIT})", rest)
    if sub and not hi and cur:
        amount_min = amount_max = amount_min + int(sub.group(1)) / (60 if standard == "CM" else 100)
    return Amount(amount_min=amount_min, amount_max=amount_max, currency=cur,
                  standard=standard, standard_source=source)


# ------------------------------------------------------------------ what it pays for, and the period

COMPONENTS = [  # (name, regex on lower-case normalized text)
    ("kaution", r"[ck]aution|erlag|sicherstellung"),
    # also OCR variants: Quatiergeld, Ouartiergeld, Quart-ergeld
    ("quartiergeld", r"[qo]uar?t[\w-]{0,2}ergeld|wohnungsgeld|quartierbeitrag|zinsbeitrag|möbelzins|mietzins"),
    ("zulage", r"zulage|adjut|triennal|quinquennal|pauschal|subvention|relutum|livreegeld|kostgeld"),
    ("taggeld", r"taggeld|diurn|diäten(?!\s*-?\s*[ck]lass)"),   # Diätenclasse is a rank
    ("pension", r"pension|ruhegenu|ruhegehalt|provision\w* für witwen"),
    ("remuneration", r"remuner|honorar|substitutionsgebühr|entlohnung|entschädigung|gage"),
    ("lohn", r"lohn|löhnung"),
    ("gehalt", r"gehalt|bestallung|bezüge|jahresbezug|gehaltsbezug|besoldung|salär|dotation|einkommen"),
]
PAY = ("gehalt", "lohn", "remuneration", "taggeld")  # Taggeld/Diurnum is the pay of day-paid clerks
PERIODS = [
    ("jahr", r"jährl|jahres|jahrl|per jahr|pro jahr|p\.\s*a\.|pro anno|per anno"),
    ("monat", r"monatl|monats|per monat|pro monat|mtl\."),
    ("woche", r"wöchentl|wochenlohn|per woche|pro woche"),
    ("tag", r"täglich|taggeld|tagelohn|taglohn|per tag|pro tag|diurn|diäten(?!\s*-?\s*[ck]lass)"),
    ("stunde", r"stündlich|per stunde|pro stunde|stundenlohn"),
]
ANNUAL_BY_DEFAULT = ("gehalt", "zulage", "quartiergeld", "pension", "remuneration")
# directly after the amount (currency, subunit, standard, "jährlich" may stand in between; no comma:
# "Gehalt [1050 fl.], Quartiergeld" lists the next item)
_AFTER_PREFIX = (rf"^[\s.]{{0,3}}(?:(?:{_CUR_RE})[\s.]*)?(?:\d+\s*(?:{_SUBUNIT})[\s.]*)?(?:(?:{_OEW}|{_CM})[\s.]*)?"
                 r"(?:(?:jährl|monatl|wöchentl|täglich)\w*[\s.]*)?")


# what may stand between the amount and a keyword further on in a list: "[500 fl.], 450 fl. und 400 fl. Gehalt"
_LIST_GAP = re.compile(rf"(?:[\s\d.,—–-]|(?:{_CUR_RE}|{_SUBUNIT}|{_OEW}|{_CM})|und\b|oder\b|resp\.|bzw\.|"
                       r"beziehungsweise\b|eventuell\b|event\.|je\b|bis\b)*")


def _nearest(patterns, before: str, after: str, closed: bool = False) -> str | None:
    """Keyword right after the amount wins ("42 fl. Quartiergeld"), unless the span ends with a comma
    (`closed`); else the nearest one before it, or after it at the end of a list of amounts
    ("500 fl., 450 fl. und 400 fl. Gehalt"), whichever is closer."""
    for name, rx in patterns if not closed else ():
        if re.match(_AFTER_PREFIX + rf"[\w-]*?(?:{rx})", after):  # also inside a compound: "Functionszulage"
            return name
    best, dist = None, float("inf")
    for name, rx in patterns:  # by word start, so the earlier pattern wins within a word ("Quartiergeldentschädigung")
        for m in re.finditer(rx, before):
            word_start = max(before.rfind(" ", 0, m.start()), before.rfind("-", 0, m.start())) + 1
            if len(before) - word_start < dist:
                best, dist = name, len(before) - word_start
    for name, rx in patterns:
        for m in re.finditer(rf"[\w-]*?(?:{rx})", after):
            gap = after[:m.start()]
            if re.search(r"\d", gap) and _LIST_GAP.fullmatch(gap) and m.start() < dist:
                best, dist = name, m.start()
            break
    return best


def classify(before: str, after: str, span: str = "") -> tuple[str | None, str | None, str | None]:
    """(component, period, period_source) from the normalized text around an amount."""
    b, a = before.lower(), after.lower()
    closed = bool(re.search(r"[,;]\s*$", span))  # "Gehalt [300 fl,] Activitätszulage"
    component = _nearest(COMPONENTS, b, a, closed)
    period, source = _nearest(PERIODS, b[-45:], a[:30], closed), "stated"
    if component == "taggeld" and period is None:
        period = "tag"
    if period is None and component in ANNUAL_BY_DEFAULT:
        period, source = "jahr", "assumed"
    if period is None or component == "kaution":
        period, source = None, None
    return component, period, source


# ------------------------------------------------------------------ spans → rows

AD_SALARY_SCHEMA = pa.schema([
    ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()),
    ("span_start", pa.int32()), ("span_end", pa.int32()), ("phrase", pa.string()),
    ("amount_min", pa.float64()), ("amount_max", pa.float64()), ("currency", pa.string()),
    ("standard", pa.string()), ("standard_source", pa.string()),
    ("component", pa.string()), ("period", pa.string()), ("period_source", pa.string()),
    ("parsed_by", pa.string()),
])


def _windows(text: str, start: int, end: int) -> tuple[str, str, str]:
    return (normalize_text(text[max(0, start - BEFORE):start]), normalize_text(text[start:end]),
            normalize_text(text[end:end + AFTER]))


def parse_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """spans: ad_id, newspaper, year, date, text, start, end, phrase → one row per span.
    Rows the rules cannot read have parsed_by = None and no amount."""
    rows = []
    for r in spans.itertuples():
        before, span, after = _windows(r.text, r.start, r.end)
        date = pd.Timestamp(r.date).date() if pd.notna(r.date) else None
        lead = re.search(r"(\d+)\s*$", before)
        if lead and re.match(rf"\W*(?:{_CUR_RE})\s*\d+\s*(?:{_SUBUNIT})", span.lower()):
            span = f"{lead.group(1)} {span}"  # "147 [fl. 50 kr.]": the number stands before the span
        fragment = is_fragment(span, before)
        not_money = not fragment and is_not_money(f"{span} {after}")
        a = None if fragment or not_money or not re.search(r"\d", span) else parse_amount(f"{span} {after}", date, before)
        ok = a is not None and a.currency is not None
        component, period, psource = classify(before, after, span)
        rows.append({
            "ad_id": r.ad_id, "newspaper": r.newspaper, "year": r.year, "span_start": r.start, "span_end": r.end,
            "phrase": r.phrase, "amount_min": a.amount_min if ok else None, "amount_max": a.amount_max if ok else None,
            "currency": a.currency if ok else None, "standard": a.standard if ok else None,
            "standard_source": a.standard_source if ok else None, "component": component,
            "period": period, "period_source": psource,
            "parsed_by": "rules" if ok else "fragment" if fragment else "not_money" if not_money else None,
            "snippet": f"…{before[-50:]} [{span}] {after[:40]}…",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ LLM for leftovers

class SalaryItem(BaseModel):
    i: int
    is_amount: bool = Field(description="false, wenn der markierte Ausdruck keinen Geldbetrag nennt")
    amount_min: float | None = None
    amount_max: float | None = None
    currency: Literal["fl", "K", "S", "RM", "M", "Tlr", "Fr"] | None = None


class SalaryBatch(BaseModel):
    items: list[SalaryItem]


SYSTEM_PROMPT = """Du liest Geldbeträge aus historischen Stellenanzeigen der Wiener Zeitung (1850–1950).
In jedem Ausschnitt ist ein Ausdruck in [eckigen Klammern] markiert; davor steht das Jahr der Anzeige. Gib für den markierten Ausdruck an:
- is_amount: nennt der markierte Ausdruck (zusammen mit dem direkt folgenden Text) einen Geldbetrag? "zwölf Stunden", "10 Wiener Klafter" → false.
- amount_min, amount_max: der Betrag als Zahl; bei einer Spanne ("von eintausend bis zweitausend Gulden") Minimum und Maximum, sonst zweimal derselbe Wert.
- currency: fl (Gulden, auch "fl.", "f."), K (Kronen), S (Schilling), RM (Reichsmark), M (Mark), Tlr (Taler), Fr (Franken/Francs).
Regeln:
- "kr." heißt Kreuzer, nie Kronen: Kreuzer werden in Gulden umgerechnet (100 kr. = 1 fl.; "70 bis 80 kr." → 0.7 bis 0.8 fl.; "zwei Gulden fünfzig Kreuzer" → 2.5 fl.). Heller ebenso in Kronen (100 h = 1 K).
- Kronen gibt es erst ab 1892. Verstümmelte Währungszeichen nach einer Zahl ("1200 b", "900 E", "2200 Kk", "3600 Kč") sind ab 1892 Kronen, vorher Gulden.
- Steht beim markierten Betrag keine Währung, nimm die Währung, in der die Beträge direkt daneben angegeben sind; gibt es keine, currency = null. Nicht raten.
- Nur der markierte Betrag zählt, nicht andere Beträge im Ausschnitt. Erfinde nichts: ist der Betrag nicht lesbar, is_amount = false.
Antworte mit einem JSON-Objekt {"items": [...]} mit genau einem Element pro Ausschnitt, "i" = Nummer des Ausschnitts."""


def leftovers(rows: pd.DataFrame) -> pd.DataFrame:
    """Distinct snippets the rules could not read; key = snippet, with the year of its first occurrence."""
    left = rows[rows["parsed_by"].isna()]
    return (left.groupby("snippet").agg(count=("ad_id", "size"), year=("year", "min")).reset_index()
                .rename(columns={"snippet": "key"}).sort_values("key").reset_index(drop=True))


def _messages(batch: pd.DataFrame) -> list[dict]:
    lines = [f"{n}. ({y}) {k}" for n, (y, k) in enumerate(zip(batch["year"], batch["key"]), 1)]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Ausschnitte:\n" + "\n".join(lines)}]


def run_llm(client: DHClient, left: pd.DataFrame, *, batch_size: int = 40, progress: bool = True):
    return run_batches(client, left, job="salary", batch_size=batch_size, progress=progress,
                       build_messages=_messages, result_model=SalaryBatch,
                       empty_item=lambda: SalaryItem(i=1, is_amount=False))


def apply_llm(rows: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Fill leftover rows from the LLM; Gulden amounts get their standard by date."""
    rows = rows.copy()
    for idx in rows.index[rows["parsed_by"].isna()]:
        item = results.get(rows.at[idx, "snippet"])
        if item is None or not item.is_amount or item.amount_min is None or item.currency is None:
            continue
        rows.loc[idx, ["amount_min", "amount_max", "currency", "parsed_by"]] = [
            item.amount_min, item.amount_max if item.amount_max is not None else item.amount_min, item.currency, "llm"]
    return rows


def assign_standard_by_date(rows: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Gulden rows without a standard (LLM rows) get it by date. dates: aligned with rows."""
    rows = rows.copy()
    need = (rows["currency"] == "fl") & rows["standard"].isna() & dates.notna()
    d = pd.to_datetime(dates[need]).dt.date
    rows.loc[need, "standard"] = ["CM" if x < OEW_FROM else "öW" for x in d]
    rows.loc[need, "standard_source"] = "date"
    return rows


# ------------------------------------------------------------------ per ad: main pay and benefits

BENEFITS = [
    ("wohnung", r"wohnung|quartier(?!geld)|unterkunft|logis|zimmer"),
    ("kost", r"kost(?!en)|verpflegung|verköstigung|brot|tisch|mittagessen|verpflegs"),
    ("kleidung", r"kleidung|montur|livree|uniform"),
    ("heizung", r"heizung|beheizung|holz|brennmaterial|kohle"),
    ("licht", r"beleuchtung|licht|kerzen"),
    ("deputat", r"deputat|naturalbez|naturalien|getreide|korn"),
    ("quartiergeld", r"[qo]uar?t[\w-]{0,2}ergeld|wohnungsgeld|zinsbeitrag"),
    ("zulagen", r"zulage|adjut|pauschal"),
]

AD_PAY_SCHEMA = pa.schema([
    ("ad_id", pa.string()), ("newspaper", pa.string()), ("year", pa.int16()),
    ("pay_min", pa.float64()), ("pay_max", pa.float64()), ("pay_currency", pa.string()),
    ("pay_standard", pa.string()), ("pay_period", pa.string()), ("pay_amounts", pa.int16()),
    *[(f"benefit_{name}", pa.bool_()) for name, _ in BENEFITS],
])


def ad_pay(rows: pd.DataFrame, benefit_spans: pd.DataFrame) -> pd.DataFrame:
    """One row per ad with an amount or a benefit. Main pay = Gehalt/Lohn/Remuneration/Taggeld amounts in the
    ad's most frequent (currency, standard, period); alternatives in a notice give min–max.
    benefit_spans: ad_id, newspaper, year, phrase (verpflegung and unspecific_salary)."""
    pay = rows[rows["component"].isin(PAY) & rows["amount_min"].notna()].copy()
    pay["unit"] = list(zip(pay["currency"], pay["standard"].fillna(""), pay["period"].fillna("")))
    out = {}
    for ad_id, g in pay.groupby("ad_id"):
        unit = g["unit"].value_counts().index[0]
        g = g[g["unit"] == unit]
        out[ad_id] = {"pay_min": g["amount_min"].min(), "pay_max": g["amount_max"].max(), "pay_currency": unit[0],
                      "pay_standard": unit[1] or None, "pay_period": unit[2] or None, "pay_amounts": len(g)}
    ads = pd.concat([rows[["ad_id", "newspaper", "year"]], benefit_spans[["ad_id", "newspaper", "year"]]])
    df = ads.drop_duplicates("ad_id").set_index("ad_id")
    df = df.join(pd.DataFrame.from_dict(out, orient="index"))
    words = benefit_spans.assign(p=benefit_spans["phrase"].map(lambda t: normalize_text(t).lower()))
    words = words.groupby("ad_id")["p"].agg(" | ".join)
    # unspecific amounts ("Quartiergeld", "Zulage") also show in the salary rows' components
    comps = rows.dropna(subset=["component"]).groupby("ad_id")["component"].agg(" | ".join)
    text = words.reindex(df.index).fillna("") + " | " + comps.reindex(df.index).fillna("")
    for name, rx in BENEFITS:
        df[f"benefit_{name}"] = text.str.contains(rx, regex=True)
    df["pay_amounts"] = df["pay_amounts"].fillna(0).astype(int)
    return df.reset_index()


def summarize(rows: pd.DataFrame, pay: pd.DataFrame) -> dict:
    n = len(rows)
    return {
        "salary_spans": n,
        "parsed_by_rules": int((rows["parsed_by"] == "rules").sum()),
        "parsed_by_llm": int((rows["parsed_by"] == "llm").sum()),
        "not_an_amount_or_unreadable": int(rows["parsed_by"].isna().sum()),
        "currency": rows["currency"].value_counts().to_dict(),
        "standard_source": rows["standard_source"].value_counts().to_dict(),
        "component": rows["component"].value_counts(dropna=False).to_dict(),
        "period": rows["period"].value_counts(dropna=False).to_dict(),
        "period_assumed": int((rows["period_source"] == "assumed").sum()),
        "ads_with_pay": int(pay["pay_min"].notna().sum()),
        "ads_with_benefits": {c: int(pay[c].sum()) for c in pay.columns if c.startswith("benefit_")},
    }

"""Step 11: the agent loop. Qwen answers a research question by calling the tools of step 10.

The model gets the system prompt, the question and the tool specs; it calls tools (several per round are
allowed) until it answers in text, at most `agent.max_steps` rounds. If it still wants tools after that, one
last request without tools asks for the answer from what it has. Every answer is checked for its citations
(ad IDs that no tool returned are flagged) and logged with the full trace to `paths.agent_log` (JSONL), the
start of the interaction log of step 13.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from hisrag.agent.feedback import append_jsonl, current_user, now
from hisrag.agent.tools import Tools
from hisrag.config import Config, load_config

SYSTEM_PROMPT = """Du bist ein Rechercheassistent für Historikerinnen und Historiker. Du beantwortest Forschungsfragen zu historischen Stellenanzeigen aus einer Datenbank, die du nur über deine Tools siehst. Du antwortest in heutigem Deutsch.

Die Datenbank:
- Stellenanzeigen der Wiener Zeitung, 1850–1950 (bisher nur diese eine Zeitung). Die Wiener Zeitung war das Amtsblatt: ein großer Teil sind Ausschreibungen öffentlicher Stellen (Lehrer, Beamte, Justiz). Aussagen über "den Arbeitsmarkt" gelten nur für diese Zeitung und müssen das sagen.
- Arten: Stellenangebote (job_offer), Stellengesuche (job_search), Dienstleistungsangebote (service_offer), Stellenvermittlungen (vermittlung).
- Aus den 1920er und 1930er Jahren gibt es fast keine Anzeigen; Aussagen über die Zwischenkriegszeit sind kaum möglich und müssen das sagen.
- Löhne sind nominal und nur in derselben Währung, demselben Standard und Zeitraum vergleichbar (Gulden Conventionsmünze bis Oktober 1858, dann österreichische Währung; Kronen ab 1892/1900). Lohnangaben finden sich fast nur 1850–1918, nach 1900 selten.
- Die Texte sind OCR mit Nachkorrektur: Lesefehler und alte Schreibung sind normal. Anzeigen mit "warning" sind unsicher.
- Berufe, Anforderungen und Löhne wurden automatisch aus den Anzeigen gelesen; sie können Fehler enthalten.

Vorgehen:
- Berufe und Anforderungen zuerst mit expand_concept nachschlagen, dann mit den gefundenen Lemmata, Tags und Berufsfeldern filtern. Auch verwandte historische Bezeichnungen nachschlagen (z. B. für Hausangestellte: Köchin, Magd, Stubenmädchen).
- expand_concept mit allen Begriffen auf einmal aufrufen (terms ist eine Liste); unabhängige Aufrufe in derselben Runde stellen.
- search_ads im Modus semantic für Themen und Fragen, im Modus keyword für Namen, Orte und feste Begriffe. Im Modus keyword nur die kennzeichnenden Wörter angeben (Rothschild, nicht Haus Rothschild); Varianten mit OR verbinden, andere Wortformen mit wort*. Ergibt eine Suche 0 Treffer, die Suche lockern (weniger Wörter, wort*, OR, weniger Filter, Modus semantic), bevor du schreibst, dass etwas nicht vorkommt.
- Zahlen, Anteile, Entwicklungen und Löhne nur mit aggregate ermitteln, nie aus Suchtreffern hochrechnen. Ein Anteil braucht eine passende Grundmenge (z. B. alle Stellenangebote desselben Jahrzehnts).
- Kleine Zahlen prüfen: Die Lemmata sind eng (die meisten Lehrerstellen haben das Lemma Lehrer, die Schulart steht nur im Text). Ergibt ein Filter wenige Anzeigen, mit einem breiteren Filter gegenprüfen (z. B. Lemma Lehrer und keyword Volksschul*), bevor du die Zahl als vollständig darstellst.
- Fragen nach Anforderungen (Sprachen, Familienstand, Religion, Alter, Bildung, Naturalleistungen …): zuerst die Anforderungs-Tags zählen, mit aggregate (group_by requirement_value mit der passenden dimension, z. B. familienstand oder religion; Naturalleistungen mit group_by benefit), auf der Grundmenge der Frage (Beruf, Art der Anzeige, Zeitraum). Die Tags fassen verschiedene Formulierungen zusammen ("unverehelicht", "ledige" → familienstand:ledig); eine keyword-Suche findet nur das genaue Wort und verfehlt die anderen. Erst danach Beispiele mit search_ads (Filter requirement_tags) und get_ad. Nie aus 0 keyword-Treffern schließen, dass eine Anforderung fehlt, ohne die Tags gezählt zu haben.
- Anforderungs-Tags genau lesen: unterrichtssprache ist die Sprache einer Schule, keine Anforderung an die Person; sprachkenntnisse sind Kenntnisse der Person. Getrennt berichten, nicht zusammenzählen.
- Die Art der Anzeige beachten: Fragt die Frage, was verlangt, erwartet, angeboten oder bezahlt wurde, auf Stellenangebote filtern (labels: job_offer), auch beim Zählen der Tags. Stellengesuche zeigen, wie sich Suchende beschreiben, nicht, was Arbeitgeber erwarten; sie nur verwenden, wenn die Frage nach den Suchenden fragt, und dann so benennen.
- get_ad für die Einzelheiten von Anzeigen, die du zitierst oder genauer prüfst.
- Zeiträume der Frage als year_from/year_to filtern.

Antwort:
- Belege jede Aussage über Anzeigen mit ihren IDs in eckigen Klammern, z. B. [wrz_18620412_017_region_0162]. Nur IDs, die ein Tool geliefert hat.
- Jede Zahl muss aus einem Tool-Ergebnis stammen. Beschreibe keine Tool-Aufrufe, die du nicht gemacht hast.
- Sage immer, worauf die Antwort beruht: bei Zahlen die Zahl der gezählten Anzeigen (n_ads aus aggregate), bei Beispielen, wie viele Anzeigen du gelesen hast. Beispiele sind Beispiele, keine repräsentative Auswahl: "häufig", "viele" oder "meist" nur mit Zahlen aus aggregate, sonst "in X der gelesenen Anzeigen".
- Bezeichnungen der Quelle bleiben stehen (z. B. Böhmisch, Ruthenisch, Commis), nur die Schreibung wird modernisiert; wörtliche Zitate in Anführungszeichen mit der Schreibung der Quelle.
- Wenn die Daten die Frage nicht oder nur teilweise beantworten, sage das deutlich.
- Knapp und gegliedert: zuerst die Antwort, dann Belege und Einschränkungen."""

# v1: pilot 2026-10-02; v2: keyword use, small counts, tag types, kinds of ad, no numbers without tools;
# v3: expand_concept takes a list of terms (one round instead of one per term);
# v4: requirement questions start by counting the tags with aggregate, not with keyword variants;
# v5: job offers only for what was demanded/offered; aggregate over all tags without a dimension
PROMPT_VERSION = "v5"
USE_TOOLS = ("Du hast noch kein Tool aufgerufen. Die Antwort muss auf den Daten beruhen: rufe zuerst die passenden "
             "Tools auf.")
FINAL_NUDGE = ("Du hast die höchste Zahl an Tool-Aufrufen erreicht. Beantworte die Frage jetzt ohne weitere Tools "
               "mit dem, was du gefunden hast, und sage, was offen bleibt.")

AD_ID = re.compile(r"\b[a-z]+_\d{8}_\d{3}_region_\d{4}\b")


@dataclass
class Answer:
    question: str
    answer: str
    id: str = field(default_factory=lambda: uuid4().hex[:12])  # ratings refer to it
    user: str = field(default_factory=current_user)
    trace: list[dict] = field(default_factory=list)  # one entry per tool call
    steps: int = 0                                     # model requests
    reasoning_chars: int = 0                           # length of Qwen's reasoning over all requests
    stopped: str = "answer"                            # answer | max_steps | error
    cited: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)  # cited but returned by no tool
    seconds: float = 0.0
    tokens: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def tools_used(self) -> str:
        return " → ".join(t["tool"] for t in self.trace)


def ids_in(obj) -> set[str]:
    return set(AD_ID.findall(json.dumps(obj, ensure_ascii=False)))


def check_citations(answer: str, trace: list[dict]) -> tuple[list[str], list[str]]:
    """(cited IDs in order of appearance, cited IDs that no tool result contained)."""
    cited = list(dict.fromkeys(AD_ID.findall(answer)))
    seen = set().union(*(ids_in(t["result"]) for t in trace)) if trace else set()
    return cited, [i for i in cited if i not in seen]


class Agent:
    def __init__(self, cfg: Config | None = None, client=None, tools: Tools | None = None):
        self.cfg = cfg or load_config()
        if client is None:
            from hisrag.llm import DHClient
            client = DHClient(self.cfg)
        self.client = client
        self.tools = tools or Tools(self.cfg, client=client)
        self.specs = self.tools.specs()

    def ask(self, question: str, *, thinking: bool | None = None, max_steps: int | None = None,
            log: bool = True, use_cache: bool = True) -> Answer:
        a_cfg = self.cfg["agent"]
        thinking = a_cfg["thinking"] if thinking is None else thinking
        max_steps = max_steps or a_cfg["max_steps"]
        settings = {"model": self.client.model, "thinking": thinking, "max_steps": max_steps,
                    "prompt_version": PROMPT_VERSION}
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        out = Answer(question=question, answer="", settings=settings)
        tokens = {"prompt": 0, "completion": 0}
        t0 = time.monotonic()

        def request(**kw):
            r = self.client.chat(messages, thinking=thinking, max_tokens=a_cfg["max_tokens"], use_cache=use_cache,
                                 timeout_s=a_cfg["timeout_s"], **kw)
            out.steps += 1
            out.reasoning_chars += len(r.reasoning or "")
            for k in tokens:
                tokens[k] += int(r.usage.get(f"{k}_tokens") or 0)
            return r

        nudged = False
        try:
            for _ in range(max_steps):
                r = request(tools=self.specs)
                messages.append(r.message)
                if not r.tool_calls:
                    if not out.trace and not nudged:  # answered from nothing: once back to the tools
                        nudged = True
                        messages.append({"role": "user", "content": USE_TOOLS})
                        continue
                    out.answer = r.content or ""
                    break
                for tc in r.tool_calls:
                    name, args = tc["function"]["name"], tc["function"]["arguments"]
                    t1 = time.monotonic()
                    result = self.tools.call(name, args)
                    out.trace.append({"step": out.steps, "tool": name, "args": _parse(args), "result": result,
                                      "ms": round(1000 * (time.monotonic() - t1))})
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
            else:
                out.stopped = "max_steps"
                messages.append({"role": "user", "content": FINAL_NUDGE})
                out.answer = request().content or ""
        except Exception as exc:  # an API failure must not lose the trace
            out.stopped = "error"
            out.answer = f"[Fehler: {type(exc).__name__}: {str(exc)[:300]}]"
        out.seconds = round(time.monotonic() - t0, 1)
        out.tokens = tokens
        out.cited, out.unknown_ids = check_citations(out.answer, out.trace)
        if log:
            self.log(out)
        return out

    def log(self, a: Answer) -> None:
        append_jsonl(self.cfg.path("agent_log"), {"time": now(), **asdict(a)})


def _parse(args: str | dict):
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return args

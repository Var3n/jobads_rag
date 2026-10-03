"""Step 12: the playground. A researcher asks a question, reads the answer with what it rests on, the tool calls
and the cited ads (text and printed clipping), and rates it. Answers and ratings go to the interaction log.

    from hisrag.agent.playground import Playground
    Playground().show()

`Playground` needs widget support in the Jupyter frontend (the extension jupyterlab_widgets on the server). Where
that is missing, `Sitzung` does the same with plain cells: s.frage("…") shows the answer, the tool calls and the
cited ads in collapsible sections (HTML <details>, no extension needed), s.bewerte("richtig", …) rates it.
"""

from __future__ import annotations

import html
import json

import ipywidgets as w
from IPython.display import HTML, Markdown, display

from hisrag.agent.feedback import rate
from hisrag.agent.loop import Agent, Answer
from hisrag.agent.render import basis_line, link_citations, tool_summary, trace_markdown
from hisrag.config import Config, load_config

VERDICT_OPTIONS = [("richtig", "richtig"), ("teilweise richtig", "teilweise"), ("falsch", "falsch")]
FIT_OPTIONS = [("passen", True), ("passen nicht", False), ("nicht geprüft", None)]


def make_agent(cfg: Config | None = None) -> Agent:
    """The agent without response cache: questions are rarely repeated, and an SQLite file on the shared NFS
    folder, written by several researchers at once, is unreliable."""
    from hisrag.llm import DHClient

    cfg = cfg or load_config()
    return Agent(cfg, DHClient(cfg, use_cache=False))


def cited_ads(agent: Agent, ids: list[str]) -> dict[str, dict]:
    """Date, kind, text and clipping link of the cited ads (the ones that exist)."""
    if not ids:
        return {}
    tools = agent.tools
    with tools._lock:
        rows = tools.con.execute("""SELECT ad_id, date, label, heading_text, text_norm, iiif_link FROM ad_clean
                                    WHERE ad_id IN (SELECT unnest(?))""", [ids]).df()
    return {r.ad_id: r._asdict() for r in rows.itertuples(index=False)}


def ad_html(ad_id: str, ad: dict | None) -> str:
    if ad is None:
        return f"<p><b>{html.escape(ad_id)}</b>: diese Anzeige gibt es nicht</p>"
    head = f"{ad['heading_text']}\n" if isinstance(ad["heading_text"], str) else ""
    link = html.escape(ad["iiif_link"])
    return (f"<p><b><a href='{link}' target='_blank'>{html.escape(ad_id)}</a></b> · {str(ad['date'])[:10]} · "
            f"{html.escape(ad['label'])}</p><div style='white-space:pre-wrap;margin:0 0 6px 0'>"
            f"{html.escape(head + ad['text_norm'])}</div><img src='{link}' style='max-width:520px;"
            f"border:1px solid #ccc;margin-bottom:14px' loading='lazy'>")


def trace_html(trace: list[dict]) -> str:
    """The tool calls as an HTML list (inside <details> Markdown is not rendered)."""
    items = "".join(f"<li>Runde {t['step']}: <code>{html.escape(t['tool'])}</code> "
                    f"{html.escape(json.dumps(t['args'], ensure_ascii=False))}<br>→ "
                    f"{html.escape(tool_summary(t['tool'], t['result']))}</li>" for t in trace)
    return f"<ul>{items}</ul>" if items else "<p>keine Tool-Aufrufe</p>"


class Playground:
    def __init__(self, cfg: Config | None = None, agent: Agent | None = None):
        self.agent = agent or make_agent(cfg)
        self.cfg = self.agent.cfg
        self.current: Answer | None = None

        self.question = w.Textarea(placeholder="Forschungsfrage in heutigem Deutsch, z. B. „Welche Sprachkenntnisse "
                                               "wurden von Gouvernanten verlangt?“",
                                   layout=w.Layout(width="100%", height="70px"))
        self.ask_button = w.Button(description="Fragen", button_style="primary", icon="search")
        self.new_button = w.Button(description="Neue Frage", icon="refresh")
        self.status = w.HTML()
        self.output = w.Output()

        self.verdict = w.ToggleButtons(options=VERDICT_OPTIONS, value=None, description="Antwort:",
                                       style={"description_width": "110px"})
        self.citations_fit = w.ToggleButtons(options=FIT_OPTIONS, value=None, description="Belege:",
                                             style={"description_width": "110px"})
        self.comment = w.Textarea(placeholder="Kommentar (optional): was fehlt, was ist falsch, was hätte geholfen?",
                                  layout=w.Layout(width="100%", height="60px"))
        self.save_button = w.Button(description="Bewertung speichern", icon="check")
        self.rating_status = w.HTML()
        self.rating = w.VBox([w.HTML("<b>Bewertung</b> (geht mit der Antwort in das Interaktionsprotokoll)"),
                              self.verdict, self.citations_fit, self.comment,
                              w.HBox([self.save_button, self.rating_status])],
                             layout=w.Layout(display="none", border="1px solid #ddd", padding="8px"))

        self.ask_button.on_click(self._ask)
        self.new_button.on_click(self._new)
        self.save_button.on_click(self._save)
        self.widget = w.VBox([self.question, w.HBox([self.ask_button, self.new_button, self.status]),
                              self.output, self.rating])

    def show(self) -> None:
        display(self.widget)

    # -------------------------------------------------------------- actions

    def _ask(self, _=None) -> None:
        q = self.question.value.strip()
        if not q:
            self.status.value = "Bitte eine Frage eingeben."
            return
        self.ask_button.disabled = True
        self.status.value = "<i>Die Antwort wird erarbeitet (meist 30–60 Sekunden) …</i>"
        self.output.clear_output()
        self.rating.layout.display = "none"
        try:
            self.current = self.agent.ask(q)
        finally:
            self.ask_button.disabled = False
        self.status.value = ""
        self._render(self.current)
        self.verdict.value, self.citations_fit.value, self.comment.value = None, None, ""
        self.rating_status.value = ""
        self.rating.layout.display = "flex"

    def _new(self, _=None) -> None:
        self.question.value = ""
        self.status.value = ""
        self.output.clear_output()
        self.rating.layout.display = "none"
        self.current = None

    def _save(self, _=None) -> None:
        if self.current is None:
            return
        if self.verdict.value is None:
            self.rating_status.value = "Bitte zuerst die Antwort bewerten (richtig / teilweise / falsch)."
            return
        rate(self.current.id, self.verdict.value, self.citations_fit.value, self.comment.value, cfg=self.cfg)
        self.rating_status.value = "✓ gespeichert"

    # -------------------------------------------------------------- display

    def cited_ads(self, ids: list[str]) -> dict[str, dict]:
        return cited_ads(self.agent, ids)

    def _render(self, a: Answer) -> None:
        ads = self.cited_ads(a.cited)
        links = {i: ad["iiif_link"] for i, ad in ads.items()}
        trace_out, ads_out = w.Output(), w.Output()
        with trace_out:
            display(Markdown(trace_markdown(a.trace) or "keine Tool-Aufrufe"))
        with ads_out:
            display(HTML("".join(ad_html(i, ads.get(i)) for i in a.cited)))
        details = w.Accordion(children=[trace_out, ads_out], selected_index=None)
        details.set_title(0, f"Tool-Aufrufe ({len(a.trace)})")
        details.set_title(1, f"Zitierte Anzeigen ({len(a.cited)})")
        with self.output:
            display(Markdown(link_citations(a.answer, links)))
            display(Markdown(f"*Grundlage: {basis_line(a)}*"))
            if a.unknown_ids:
                display(Markdown(f"**Achtung:** zitierte IDs, die kein Tool geliefert hat: {', '.join(a.unknown_ids)}"))
            display(details)


class Sitzung:
    """The playground without widgets, for frontends without widget support.

        s = Sitzung()
        s.frage("Welche Sprachkenntnisse wurden von Gouvernanten verlangt?")
        s.bewerte("teilweise", belege=True, kommentar="Englisch fehlt")
    """

    def __init__(self, cfg: Config | None = None, agent: Agent | None = None):
        self.agent = agent or make_agent(cfg)
        self.cfg = self.agent.cfg
        self.letzte: Answer | None = None

    def frage(self, question: str) -> None:
        if not question.strip():
            raise ValueError("Bitte eine Frage angeben.")
        print("Die Antwort wird erarbeitet (meist 30–60 Sekunden) …", flush=True)
        a = self.letzte = self.agent.ask(question.strip())
        ads = cited_ads(self.agent, a.cited)
        display(Markdown(link_citations(a.answer, {i: ad["iiif_link"] for i, ad in ads.items()})))
        display(Markdown(f"*Grundlage: {basis_line(a)}*"))
        if a.unknown_ids:
            display(Markdown(f"**Achtung:** zitierte IDs, die kein Tool geliefert hat: {', '.join(a.unknown_ids)}"))
        display(HTML(f"<details><summary><b>Tool-Aufrufe ({len(a.trace)})</b></summary>{trace_html(a.trace)}</details>"
                     f"<details><summary><b>Zitierte Anzeigen ({len(a.cited)})</b></summary>"
                     f"{''.join(ad_html(i, ads.get(i)) for i in a.cited)}</details>"))
        display(Markdown('Bewerten: `s.bewerte("richtig" | "teilweise" | "falsch", belege=True | False | None, '
                         'kommentar="…")`'))

    def bewerte(self, urteil: str, belege: bool | None = None, kommentar: str = "") -> None:
        """urteil: richtig, teilweise oder falsch; belege: passen die zitierten Anzeigen (None = nicht geprüft)."""
        if self.letzte is None:
            raise ValueError("Es gibt noch keine Antwort zu bewerten.")
        rate(self.letzte.id, urteil, belege, kommentar, cfg=self.cfg)
        print(f"✓ Bewertung gespeichert ({urteil}).")

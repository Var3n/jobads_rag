import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Recherche in historischen Stellenanzeigen

Stell eine Forschungsfrage in heutigem Deutsch. Der Assistent sucht in den Stellenanzeigen der **Wiener Zeitung
(1850–1950)**, zählt, liest Anzeigen und antwortet mit Belegen. Eine Antwort dauert meist 30–60 Sekunden.

* **Belege:** Die IDs in eckigen Klammern sind Anzeigen; ein Klick öffnet den Ausschnitt der gedruckten Seite. Unter
  *Zitierte Anzeigen* stehen Text und Bild aller zitierten Anzeigen, unter *Tool-Aufrufe* jeder Schritt der Recherche.
* **Grundlage:** Die Zeile unter der Antwort sagt, worauf sie beruht: gezählte Anzeigen (*n*) und gelesene Anzeigen.
* **Grenzen der Quelle:** Die Wiener Zeitung war das Amtsblatt; viele Anzeigen sind öffentliche Ausschreibungen. Aus
  den 1920er und 1930er Jahren gibt es kaum Anzeigen. Berufe, Anforderungen und Löhne wurden automatisch gelesen.
* **Bewertung:** Bitte bewerte jede Antwort, auch kurz. Fragen, Antworten und Bewertungen werden gespeichert und
  sind die Grundlage, um den Assistenten zu prüfen und zu verbessern.

Jede Frage wird für sich beantwortet; der Assistent kennt deine früheren Fragen nicht. Wenn eine Zelle meldet, dass
etwas nicht gefunden wird (z. B. nach einem Update des Codes): Kernel neu starten (Kernel → Restart Kernel)."""),
    md("Beim ersten Mal: gib deinen eigenen API-Schlüssel für DHinfra ein (er wird in `~/.hisrag.env` gespeichert, "
       "nur für dich lesbar)."),
    code("""from hisrag.config import env_file, load_config, set_api_key

if not load_config().api_key:
    set_api_key()
print("API-Schlüssel aus", env_file())"""),
    md("""## Fragen

Neue Frage: ändere den Text in `s.frage("…")` und führe die Zelle aus. *Tool-Aufrufe* und *Zitierte Anzeigen*
lassen sich unter der Antwort aufklappen."""),
    code("""from hisrag.agent.playground import Sitzung

s = Sitzung()"""),
    code("""s.frage("Welche Sprachkenntnisse wurden von Gouvernanten verlangt?")"""),
    md("""## Bewerten

Dein Urteil: `"richtig"`, `"teilweise"` oder `"falsch"`; `belege`: passen die zitierten Anzeigen zur Antwort (`True`,
`False`, oder `None` = nicht geprüft); `kommentar`: was fehlt oder falsch ist. Eine spätere Bewertung derselben
Antwort ersetzt die frühere."""),
    code("""s.bewerte("richtig", belege=True, kommentar="")"""),
    md("""## Mit Eingabefeld und Knöpfen (wenn die Oberfläche Widgets unterstützt)

Erscheint statt Feld und Knöpfen nur eine Textzeile `VBox(children=…)`, fehlt der Oberfläche die Widget-Erweiterung;
dann nimm die Zellen oben."""),
    code("""from hisrag.agent.playground import Playground

Playground().show()"""),
    md("## Deine bisherigen Fragen und Bewertungen"),
    code("""from hisrag.agent.feedback import current_user, interactions

log = interactions()
mine = log[log["user"] == current_user()] if len(log) else log
mine[["time", "question", "verdict", "citations_fit", "comment", "seconds"]].tail(20)"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/10_playground.ipynb")

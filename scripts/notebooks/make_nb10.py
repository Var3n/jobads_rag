import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Recherche in historischen Stellenanzeigen

Stellen Sie eine Forschungsfrage in heutigem Deutsch. Der Assistent sucht in den Stellenanzeigen der **Wiener Zeitung
(1850–1950)**, zählt, liest Anzeigen und antwortet mit Belegen. Eine Antwort dauert meist 30–60 Sekunden.

* **Belege:** Die IDs in eckigen Klammern sind Anzeigen; ein Klick öffnet den Ausschnitt der gedruckten Seite. Unter
  *Zitierte Anzeigen* stehen Text und Bild aller zitierten Anzeigen, unter *Tool-Aufrufe* jeder Schritt der Recherche.
* **Grundlage:** Die Zeile unter der Antwort sagt, worauf sie beruht: gezählte Anzeigen (*n*) und gelesene Anzeigen.
* **Grenzen der Quelle:** Die Wiener Zeitung war das Amtsblatt; viele Anzeigen sind öffentliche Ausschreibungen. Aus
  den 1920er und 1930er Jahren gibt es kaum Anzeigen. Berufe, Anforderungen und Löhne wurden automatisch gelesen.
* **Bewertung:** Bitte bewerten Sie jede Antwort, auch kurz. Fragen, Antworten und Bewertungen werden gespeichert und
  sind die Grundlage, um den Assistenten zu prüfen und zu verbessern.

Jede Frage wird für sich beantwortet; der Assistent kennt frühere Fragen nicht."""),
    md("Beim ersten Mal: den eigenen API-Schlüssel für DHinfra eingeben (er wird in `~/.hisrag.env` gespeichert, "
       "nur für Sie lesbar)."),
    code("""from hisrag.config import env_file, load_config, set_api_key

if not load_config().api_key:
    set_api_key()
print("API-Schlüssel aus", env_file())"""),
    code("""from hisrag.agent.playground import Playground

Playground().show()"""),
    md("## Ihre bisherigen Fragen und Bewertungen"),
    code("""from hisrag.agent.feedback import current_user, interactions

log = interactions()
mine = log[log["user"] == current_user()] if len(log) else log
mine[["time", "question", "verdict", "citations_fit", "comment", "seconds"]].tail(20)"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "hisrag", "display_name": "hisrag", "language": "python"}})
nbf.write(nb, "notebooks/10_playground.ipynb")

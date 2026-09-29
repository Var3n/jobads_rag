import pandas as pd
import pytest

from hisrag.normalize.quality import assess, death_register_entries, guess_language, max_ngram_repeat, ocr_support
from hisrag.normalize.text import normalize_text


@pytest.mark.parametrize("raw, expected", [
    ("Sprach⸗\nkenntniſſen, Muſik", "Sprachkenntnissen, Musik"),       # line-break hyphen: join
    ("Dragoneer⸗Re⸗ gimente", "Dragoneer-Regimente"),                  # correction turned break into space
    ("Ma- thematik", "Mathematik"),
    ("Zwirn⸗ und Wollhandlung", "Zwirn- und Wollhandlung"),            # elision before conjunction: keep
    ("Lehrer⸗\noder Lehrerinſtelle", "Lehrer- oder Lehrerinstelle"),
    ("Poſt⸗ und Telegraphen⸗\nDirektion", "Post- und Telegraphen-Direktion"),  # capital: keep hyphen
    ("4- bis 20000 fl.", "4- bis 20000 fl."),
    ("„Dem Fremden“ in\nder Specerei", '"Dem Fremden" in der Specerei'),
    (None, ""),
    (float("nan"), ""),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_death_register_detection():
    entry = "Kraft Vincenz, Hausdiener, 48 J., VIII., Lerchenfelderſtraße 56, Lungenſchwindſucht."
    fragment = "Pröll Anna, Stubenmädchen, 33 J., von Pinkafeld zugereiſt, Bauchfellentzündung."
    job_search = "Ein Mädchen, 20 J., ſucht Stelle als Stubenmädchen. VII., Neubaugaſſe 5."
    modern = "Textilkaufmann, 32 Jahre, mit Praxis in Industrie und Handel, sucht passenden Wirkungskreis."
    assert death_register_entries(entry) == 1
    assert death_register_entries(fragment) == 1
    assert death_register_entries(job_search) == 0
    assert death_register_entries(modern) == 0


def test_language_guess():
    assert guess_language("Per aspirare al posto di Giudice vacante presso il Giudizio") == "it"
    assert guess_language("Une famille qui vit à la campagne cherche pour les enfants") == "fr"
    assert guess_language("Ein Gärtner, verheirathet, ohne Kinder") == "de"


def test_repetition_and_support():
    loop = "Bedingung: Lehrbefähigung für Stenographie. " * 6
    assert max_ngram_repeat(loop) >= 5 and max_ngram_repeat("eine Köchin wird gesucht") == 0
    assert ocr_support("Eine Köchin wird geſucht.", "Eine Kochin wird geſucht") > 0.8
    assert ocr_support("Gute Köchin sucht Stelle zu kleiner Familie.", "öGSCαααπππτταααα u S 16645") < 0.2


def test_assess_flags():
    good = "Eine tüchtige Köchin wird für ein ſolides Haus geſucht. Näheres Bognergaſſe 5."
    ads = pd.DataFrame({
        "ad_id": ["ok", "loop", "invented", "death", "short"],
        "newspaper": "wrz", "year": 1880, "label": "job_offer", "heading_text": [None, None, None, None, "Köchinſtelle."],
        "text": [good, "Wäſcherinnen für Alles geſucht. " * 5, "Gute Köchin ſucht Stelle zu kleiner Familie oder beſſeres Haus.",
                 "Rößler Johann, Hausdiener, 66 J., IX., Marktgaſſe 16, Herzfehler.", "Köchin gesucht."],
        "text_ocr": [good, "Wäſcherinnen für Alles geſucht.", "öGSCαααπππτταααααααα u S 16645",
                     "Rößler Johann, Hausdiener, 66 J., IX., Marktgaſſe 16, Herzfehler.", "Köchin gesucht."],
    })
    q = assess(ads).set_index("ad_id")
    flagged = {ad: [c.removeprefix("flag_") for c in q.columns if c.startswith("flag_") and q.at[ad, c]] for ad in q.index}
    assert flagged == {
        "ok": [], "loop": ["pc_repetition", "pc_expanded"], "invented": ["pc_expanded", "pc_unsupported"],  # invented text is also longer than its OCR
        "death": ["death_register"], "short": ["too_short"],
    }
    assert q.at["ok", "text_norm"].startswith("Eine tüchtige Köchin wird für ein solides Haus gesucht")
    assert q.at["short", "heading_norm"] == "Köchinstelle." and pd.isna(q.at["ok", "heading_norm"])

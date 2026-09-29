import numpy as np
import pandas as pd

from hisrag.normalize.dedup import deduplicate, minhash, same_ad

NOTICE = ("An der k. k. Staatsrealschule mit deutscher Unterrichtssprache in {place} kommt mit Beginn des "
          "Schuljahres 1904/1905 eine wirkliche Lehrstelle für Mathematik und Physik als Hauptfächer zur Besetzung.")
PRIVATE = ("Eine perfecte Köchin, welche auch die Hausarbeit versteht und gute Zeugnisse besitzt, "
           "sucht einen Posten in einem soliden Hause. Näheres Bognergasse Nr. 5, im Gewölbe.")


def test_minhash_is_deterministic_and_similarity_preserving():
    a, b = minhash(PRIVATE), minhash(PRIVATE.replace("Köchin", "Kochin"))
    other = minhash(NOTICE.format(place="Budweis"))
    assert np.array_equal(a, minhash(PRIVATE))
    assert (a == b).mean() > (a == other).mean()


def test_reprint_with_ocr_variants_is_same_ad():
    # Step 3 runs on text_norm (ſ already → s), so variants are single-letter OCR confusions.
    reprint = PRIVATE.replace("perfecte", "perfekte").replace("Zeugnisse", "Zeugniße").replace("Posten", "Poften")
    assert same_ad(PRIVATE, reprint)


def test_region_boundary_differences_are_ignored():
    assert same_ad("11334 " + PRIVATE, PRIVATE + " [3085—6]")


def test_template_notice_for_other_place_is_different_ad():
    assert not same_ad(NOTICE.format(place="Budweis"), NOTICE.format(place="Karolinenthal"))
    assert not same_ad(NOTICE.format(place="Inzersdorf"), NOTICE.format(place="Atzgersdorf"))


def test_substitution_split_into_delete_and_insert_is_caught():
    a = "Behufs Besetzung der Landesthierarztesstelle bei der galizischen Statthalterei in der VIII. Rangsclasse sind Gesuche einzubringen."
    b = "Behufs Besetzung einer im galizischen Verwaltungsdienste erledigten Bezirkshauptmannsstelle in der VIII. Rangsclasse sind Gesuche einzubringen."
    assert not same_ad(a, b)


def test_fragment_is_not_matched_to_full_ad():
    assert not same_ad("Näheres Bognergasse Nr. 5, im Gewölbe.", PRIVATE)


def test_deduplicate_clusters_within_window_and_newspaper():
    rows = [
        # ad_id, newspaper, date, text, eligible, n_flags, ocr_support
        ("a1", "wrz", "1880-03-01", PRIVATE, True, 1, 0.9),
        ("a2", "wrz", "1880-03-04", PRIVATE.replace("Zeugnisse", "Zeugniße"), True, 0, 0.95),  # best → canonical
        ("a3", "wrz", "1880-03-30", PRIVATE.replace("perfecte", "perfekte"), True, 0, 0.9),
        ("far", "wrz", "1880-09-01", PRIVATE, True, 0, 1.0),        # beyond the 60-day window
        ("np", "nfp", "1880-03-02", PRIVATE, True, 0, 1.0),         # other newspaper
        ("head", "wrz", "1880-03-02", PRIVATE, False, 0, 1.0),      # ineligible (e.g. heading)
        ("b1", "wrz", "1880-03-02", NOTICE.format(place="Budweis"), True, 0, 1.0),
        ("b2", "wrz", "1880-03-05", NOTICE.format(place="Karolinenthal"), True, 0, 1.0),
    ]
    regions = pd.DataFrame(rows, columns=["ad_id", "newspaper", "date", "text_norm", "eligible", "n_flags", "ocr_support"])
    regions["year"] = 1880
    out, report = deduplicate(regions)
    out = out.set_index("ad_id")

    assert set(out.loc[["a1", "a2", "a3"], "dup_cluster_id"]) == {"a2"}
    assert out.loc["a2", "is_canonical"] and not out.loc["a1", "is_canonical"]
    assert (out.loc["a1", "dup_cluster_size"], out.loc["a1", "run_days"]) == (3, 29)
    assert str(out.loc["a3", "run_first_date"]) == "1880-03-01"
    for single in ["far", "np", "head", "b1", "b2"]:
        assert out.loc[single, "dup_cluster_id"] == single and out.loc[single, "dup_cluster_size"] == 1
    assert report["distinct_ads"] == 6

# Project status (proof of concept)

Last updated: 2026-10-02, step 8 done, step 9 next. Read this first when picking the project up; `README.md` has setup and commands.

## Goal and constraints

RAG system for Digital Humanities researchers over historical newspaper job ads, used from notebooks, answers in
modern German. Proof of concept on one newspaper (Wiener Zeitung, `wrz`, 52,823 regions, 1850–1950); final scale is
18 million ads from 29 newspapers with the same schema. No evaluation set yet: quality is judged from impressions,
pilot CSVs and review notebooks, and interactions will be logged to become one.

Consequences of the scale (120 requests/min, ~10 tokens/s per request):

* The LLM never processes ads one by one. It builds **dictionaries of distinct forms** (positions, requirement
  phrases) that are joined back to the ads, and it answers questions at query time.
* Every step runs per partition (`newspaper=…/year=…`) so it can later be run newspaper by newspaper.
* Every LLM step has a cheap **pilot** first (CSV for manual review), then the full run.
* Reasoning (`enable_thinking`) is off for bulk work: in the step 4 pilot it cost ~1,450 output tokens per form
  (full run ≈ 11 h instead of 45 min) without clearly better answers.

## Working setup

* Code is written and tested locally with an offline fake API (`hisrag.llm.FakeOpenAI`); `pytest` runs without a key.
* Everything that calls the DHinfra API runs on the cluster (JupyterHub, mamba env `hisrag`); the user pastes the
  printed JSON report and copies pilot CSVs into the local repo folder for review. Local CSVs are gitignored.
* Repo: github.com/Var3n/jobads_rag. Pull on the cluster with `git pull`; `nbstripout --install` keeps notebook
  outputs out of git.
* All LLM responses are cached on the cluster (`data/cache/llm_cache.sqlite`, keyed by the full request), so re-running a
  step after a code change only sends requests whose prompt changed. Cut-off answers and JSON repairs are cached as
  well (since 2026-09-30; before, 3–4 mapping requests were re-sent on every run and gave slightly different tags), so a
  re-run reproduces the tables exactly.
* Models (DHinfra slugs): `qwen3.5-397b` (chat, tools, vision, 256k context, JSON-schema output works), embeddings
  `bge-m3`, `qwen3-embedding-8b`, `jina-embeddings-v3-query/passage`, `jina-embeddings-v4-text-retrieval`,
  `embeddinggemma-300m`. **No rerank model** (Qwen can rerank if needed).

## Pipeline: the 15-step plan and where it stands

| # | Step | Status | Output / key numbers |
|---|---|---|---|
| 0 | API check | done | all endpoints work; first request of a session can take ~2 min (cold start) |
| 1 | Ingest CSV → Parquet | done | `data/ads`: 52,823 regions; headings linked to the ad below (1,069 of 1,822) |
| 2 | Text normalization + quality flags | done | `ad_text`: `text_norm`, language (de/it/fr), flags below |
| 3 | Repeated printings | done | `ad_dups`: 44,291 distinct ads; `is_canonical` marks one printing per ad |
| 4 | Position dictionary (LLM) | done | `position_dict` (6,090 forms, 4,402 positions), `ad_positions` (26,347 mentions, 21,173 ads) |
| 5 | Requirement tags (LLM) | done | `requirement_dict` (22,494 phrases), `ad_requirements` (66,632 tag mentions, 23,037 ads) |
| 6 | Salary parsing | done | `ad_salary` (22,639 spans: 21,667 by rules, 289 by LLM), `ad_pay` (9,722 ads with main pay, benefits per ad) |
| 7 | Final clean table + sanity plots | done | `ad_clean`: 41,024 countable ads; `notebooks/05_clean_table.ipynb` |
| 8 | Retrieval comparison | done | 299 LLM research questions, pooled LLM judgments; winner qwen3-embedding-8b on enriched text, 1,024 dims (nDCG@10 0.654) |
| 9 | Index | **next** | LanceDB with vectors, keyword index and filter columns |
| 10 | Agent tools | open | `search_ads`, `get_ad`, `aggregate` (SQL templates), `expand_concept` |
| 11 | Agent loop | open | Qwen tool calling (tested in step 0), cites ad IDs, always reports how many ads an answer rests on |
| 12 | Playground notebook | open | answer, tool trace, cited clippings (IIIF), rating widget → interaction log |
| 13 | Exploration by researchers | open | the log becomes the first evaluation set |
| 14 | Extensions | open | concept graph and `sample_ads` only where the log shows weaknesses |
| 15 | Scale test | open | run steps 1–9 on 2–3 more newspapers, extrapolate to 18M |

All tables are DuckDB views: `from hisrag.data import query; query("SELECT … FROM ads JOIN ad_text USING (ad_id) …")`.
Counts of distinct ads use `ad_dups.is_canonical` and exclude `ad_text.flag_death_register`.

## Decisions and findings worth knowing

**Data quality (step 2).** Flags, calibrated on samples:
`pc_repetition` (post-correction looped, 278), `pc_expanded` (>1.3× the OCR, 446), `pc_unsupported` (<60 % of the
corrected text's character trigrams occur in the OCR: reconstructed or **invented** text, 268), `too_short` (186),
`death_register` (1,184 entries of the Vienna death register misclassified as ads, 1860s–1890s; up to 9 % of the
1880s). Planned handling in the RAG: death register excluded from search and counts; `pc_unsupported` searchable with a
warning but not counted; loops/expansions kept.

**Repeated printings (step 3).** Template notices for different places can be 95 % identical, so similarity alone
fails. Pairs are verified word by word: same ad only if no content was *substituted* (a place, date, number or long
word on both sides without an OCR-variant counterpart). Biased towards merging less, because a wrong merge would hide an
ad in search. Remaining errors ~1–2 % (numbered sub-items of one notice, near-identical official titles).

**Positions (step 4, prompt `positions-v2`).** Per form: `term` (historical title, OCR errors corrected), `lemma`
(masculine base form for grouping), `modern`, `gender_form` (of the wording: m/f/m/f/n), `category` (16 fixed).
The form is judged by itself; context only disambiguates ("Lehrstelle" = teaching post in Austrian notices,
"Lehrkanzel" = professorship). Categories of a lemma are unified only when one has ≥ 75 % of its mentions (generic
titles like Adjunct keep per-form categories). `hisco_code` is reserved: HISCO matching is being built separately.
The Wiener Zeitung is dominated by official vacancies (teaching 9,425 mentions, administration 3,715): answers about
"the labour market" must say so.

**Requirements (step 5, vocabulary `vocab/requirements.yaml`, version 3).** 28 dimensions in six groups (Person,
Qualifikation, Eigenschaften, Stelle, Bewerbung, Sonstiges); closed lists for demographics and languages, free short
values otherwise. Application formalities and a school's language of instruction are kept apart from requirements of
the person. **Language names stay as the source names them** (Böhmisch, Walachisch, Ruthenisch), only spelling is
unified; this was the user's decision. Skill values stay free for now; the project's economist will give feedback on
the vocabulary. Editing the YAML changes the prompt version, so the next run re-maps all phrases (~1 h 50 min).
Post-model checks: a `detail` is kept only if its words occur in the phrase, allowing historical spellings and cue
words (möglichst → bevorzugt); 2,455 dropped, mostly rightly ("Hauptfach"/"Nebenfach" 1,814 mentions come from the ad,
not the phrase; they would need a per-ad rule). `detail_raw` keeps the model's version. Values outside closed lists are
marked `in_vocab = false` (~0.2 %). **Context leaks:** the mapping sometimes tags what one example ad says, above all for
short frequent phrases with several contexts ("verheirathet" → kinderlos, "absolvirter" → Bergschule/Techniker/Wundarzt,
"wissenschaftlich gebildeter" → a whole ad). Tags whose value does not occur in the phrase (9,551) are checked again by
the model with only the phrase (`verified`); 3,203 were rejected (7,447 mentions, 11 % of all tag mentions). They stay in
`requirement_dict` but give no rows in `ad_requirements`. Known wrong rejections: "Reife- und Lehrbefähigungszeugnisse"
→ Matura/belegtes Gesuch (135 mentions); fixing it needs a vocabulary example, i.e. a full re-map. Open question for later:
`sprachkenntnisse` has the highest rejection rate (66 %), check whether "Landessprachen" should become a value.

**Salary (step 6).** The `salary` spans (22,646 in 10,940 ads, 95 % in job offers, almost all 1850s–1910s; only 64
after 1920) hold bare amounts ("600 fl.", "1200 K"); what an amount pays for, its period and the Gulden standard are
in the text around it, so rules read a window of ±60–80 characters. Full run: rules read 21,667 spans (96 %), ~490 are
numbers that are no money (bread rations in Gramm, allowances in "pCt.", teaching hours) or fragments, 458 distinct
leftovers went to the LLM (289 amounts read, 197 spans unreadable or no amount). LLM prompt: the year is given with
each excerpt, "kr." is always Kreuzer, no guessed currency. Decisions: amounts stay **nominal** (no CM → ö.W.
conversion, user's decision); Gulden get CM/ö.W. as stated, else by date (CM until October 1858; of 1,989 amounts that
state their standard, 1,987 match the date rule, so the rule is safe for the ~13,300 others); Kreuzer count 1/100 fl. in
ö.W. and 1/60 in CM; Heller 1/100 K. `component` from the nearest keyword (the word right after the amount wins,
"42 fl. Quartiergeld"; lists of amounts take the keyword at their end); period stated or **assumed yearly** for
Gehalt, Zulage, Quartiergeld, Remuneration, Pension (flag `period_source`). Main pay per ad (`ad_pay`) =
Gehalt/Lohn/Remuneration/Taggeld, never Kaution or Pension; alternatives give min–max. Benefit flags per ad from the
`verpflegung` and `unspecific_salary` spans; `salary_importance` is left out (almost only stray "fl." fragments).
Known limits: keywords farther than the window (Quartiergeld categories, long position lists) leave `component` empty
(~7 %); OCR-split numbers ("9 45 fl.") are read wrongly.

**Clean table (step 7).** `ad_clean`: 52,823 regions, 49,817 searchable, 41,024 countable (30,670 job offers, 5,896
job searches, 3,205 service offers, 1,253 agency ads). Of the countable ads 40.7 % have a position, 42.6 % requirement
tags, 21.2 % a main pay. The 1920s (271 countable) and 1930s (17) are nearly empty, so the corpus is effectively
1850–1918 plus the 1940s; answers about the interwar years must say so. The 1850s are half job searches, from the 1870s
on 83–89 % job offers; pay is stated in 67 % of job offers in the 1860s, ~15 % by 1900, ~0 after 1920; the death
register sits in the 1870s–1880s. `notebooks/05_clean_table.ipynb` draws points resting on < 200 ads hollow.

**Retrieval comparison (step 8, `hisrag/eval/`, `notebooks/06_retrieval_comparison.ipynb`).**
*Design.* The first design (known-item: find one ad again from a paraphrase) was dropped after its pilot: the questions
read like headlines of the ad and copied 42 % of its long words, and researchers ask topical questions with many
relevant ads (`get_ad` is a lookup by ID, not a search). Final design (user's choice): 300 seed ads (countable, German,
with a position, ≥ 120 characters, decades weighted by √size); the LLM writes a research question plus a relevance
criterion as general as the question (no years, names, places below crown land); every method returns its top 10 over
all 49,817 searchable ads (one printing per ad); every pooled ad is judged 0/1/2 by the LLM, blind to the method, one
question per request, with a ≤ 12-word reason before the grade (the reason is stored). Measures per question: nDCG@10
(gains 0/1/3), p@10 strict (grade 2) and lenient (≥ 1), recall against the relevant printing clusters in the pool,
`seed_found`. 22 methods: BM25 (spelling-folded, `fold_spelling` in `normalize/text.py`), 5 embedding models,
BM25+model hybrids (RRF), each on `raw` text and `enriched` text (ad + "Stelle/Anforderungen/Lohn" from steps 4–6).
*Judge reliability.* Three judge prompt versions on 20 pilot questions agreed on 75–89 % of grades, but ranked the 22
methods almost identically (Spearman 0.91–0.98), so the ranking is trustworthy, absolute numbers less so. The final
judge is somewhat over-literal on 2 vs 1 ("Realgymnasium ist kein reines Gymnasium"). Caveat: question writer, judge
and the winning embedding model are all Qwen.
*Results (299 questions, 21,670 judged pairs).* nDCG@10 / strict p@10 after the extension: qwen3-embedding-8b/enriched
0.657 / 0.438, embeddinggemma-300m/enriched 0.633 / 0.434, jina v3/enriched 0.592, bge-m3/enriched 0.519, best hybrid
0.511, jina v4/enriched 0.425, BM25/enriched 0.348 / 0.219. Paired: qwen vs embeddinggemma overall not distinguishable
(+0.025, interval −0.002…+0.052), but equal on job offers and clearly better on **job searches** (0.668 vs 0.557,
n = 45) and in the 1850s–60s; all other methods reliably worse. Enriched beats raw for every model. Hybrids lose to the
pure models (BM25 is weak on paraphrased questions). Shortened qwen vectors (Matryoshka, `score --extend`): no
measurable loss down to 768 dims (2048: 0.662, 1024: 0.654, 768: 0.648), measurable at 512 (0.629) and 256 (0.607);
the job-search lead holds at every length ≥ 768.
*Cost.* Embedding 50k ads: qwen3-8b 21 min, embeddinggemma 1.7, bge-m3 2.2, jina v3 2.9, jina v4 13. For 18M ads:
qwen ~127 h (shortening does not reduce this), embeddinggemma ~10 h; storage float32 qwen@1024 74 GB, @4096 295 GB.
**Decision (user, 2026-10-02): qwen3-embedding-8b on the enriched text, stored at 1,024 dimensions.** BM25 is not
fused; it becomes a separate exact-word search mode (the test excluded names and places by design, where exact
matching matters). Whether qwen's embedding time is acceptable at 18M is checked at the scale test (step 15);
embeddinggemma-300m is the fallback (same storage at 768 dims, weaker on job searches).
*Files on the cluster.* `data/eval/` (queries, runs, judgments, scores parquet), `data/embeddings/<variant>/<model>/`
(full 4096-dim qwen vectors for raw and enriched; all five models), `data/embeddings/questions/`.

## Open items (in order)

1. **Step 9: index.** Plan agreed in outline, not started:
   * LanceDB table (new dependency `lancedb`) built from `ad_clean` WHERE `searchable`, one row per ad: `ad_id`,
     `vector` = qwen3-embedding-8b on the `enriched` text cut to 1,024 dims and renormalized (reuse
     `hisrag.eval.retrieval.documents()`, `load_vectors()`, `truncate()`; the stored 4096-dim vectors in
     `data/embeddings/enriched/qwen3-embedding-8b/` can be reused, no new embedding for the Wiener Zeitung).
   * Filter columns from `ad_clean`: newspaper, year, decade, date, label, `countable`, `is_canonical`,
     `dup_cluster_id`, lang, position categories/lemmas, `position_gender`, requirement dimensions, pay fields, benefit
     flags, quality warning; plus the display text and `iiif_link`.
   * Keyword search as a separate mode: LanceDB full-text index or the existing `BM25` with `fold_spelling`
     (historical spellings must match, e.g. Wirthschafterin/Wirtschafterin); decide by testing exact names/places.
   * Query side: embed the question with the qwen query prefix (config) and cut to 1,024 dims; one result per printing
     cluster (`dup_cluster_id`), as in the evaluation.
   * Per partition (newspaper) so more newspapers can be added; check index build time and size for the scale test.
   * Tests locally with FakeOpenAI vectors; on the cluster a small notebook to try queries (e.g. the step-8 questions)
     before the agent tools of step 10.
2. Steps 10 onward as in the table.

Smaller known issues: one hallucinated company name from context in step 5; single-occurrence OCR garbles in step 4
("Applent", "Praschneiderin") are mapped with guesses; the generic "Lehrling" category depends on its example.

## Practical notes for development

* Windows machine, Git Bash and PowerShell. Run Python as `.venv/Scripts/python` (a bare `python` can hang on the
  Windows Store alias). Long Python edits with quotes or backslashes break inside bash heredocs (backslashes are
  swallowed): use the Edit tool, or write the script to the scratchpad with the Write tool and run it.
* Notebooks 04–06 are generated by `scripts/notebooks/make_nb0X.py` (run from the repo root; they overwrite the
  notebook). Before pushing, notebooks are executed locally cell by cell with stand-in tables where the real ones only
  exist on the cluster: a local `ad_clean` built from the real steps 1–3 and 6 (salary with a fake LLM) plus random
  stand-in positions/requirements, and fake eval files from `FakeOpenAI` (random vectors and grades). Only the
  mechanics are checked that way, never the numbers.
* The user runs cluster jobs that take long with `nohup … > data/logs/<name>.log 2>&1 &`; `embed` and `score` resume.
* DuckDB quirks met so far: views cannot take prepared parameters (queries against views can); `QUALIFY` does not
  combine with `GROUP BY ALL`; `USING SAMPLE` is applied before `WHERE` (use `ORDER BY random() LIMIT n`);
  `.arrow()` returns a `RecordBatchReader` in newer versions (`.read_all()`).
* JSON reports: numpy scalars as dict keys break `json.dumps` (convert with `str(k)`/`int(v)`).

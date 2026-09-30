# Project status (proof of concept)

Last updated: 2026-09-30, step 5 audit. Read this first when picking the project up; `README.md` has setup and commands.

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
  step after a code change only sends requests whose prompt changed.
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
| 5 | Requirement tags (LLM) | done, audit open | `requirement_dict` (22,494 phrases), `ad_requirements` (66,632 tag mentions, 23,037 ads) |
| 6 | Salary parsing | **next** | amount, currency (fl. CM/ö.W., Kronen, Schilling, RM), period; rules first, LLM for leftovers |
| 7 | Final clean table + sanity plots | open | one joined view, plots per decade |
| 8 | Embedding comparison | open | synthetic known-item queries (Qwen writes a query for a known ad), recall@k, keyword vs. vector vs. hybrid |
| 9 | Index | open | LanceDB with vectors, keyword index and filter columns |
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
Post-model checks: a `detail` is kept only if its words occur in the phrase (`detail_raw` keeps the model's version);
values outside closed lists are marked `in_vocab = false` (~0.2 %, left as they are).

## Open items (in order)

1. **Value leaks from the context (step 5).** The mapping sometimes tags what an example ad says, not the phrase:
   `"verheirathet"` → kinderlos (340 mentions), `"gebildetes"` → weiblich, `"absolvirter"` → Bergschule/Techniker/Wundarzt,
   `"guter"` → Schießen/Musik. A value-in-phrase flag alone cannot separate these from fair paraphrases (29 % of tag
   mentions are flagged, mostly like "gehörig instruierten" → vorschriftsmäßiges Gesuch). So the flagged tags (~9,000
   distinct phrase–tag pairs, ~150 requests) are checked again by the model with **only the phrase** (`TagVerifier`,
   `verified` in the tag struct; rejected tags give no ad rows). Next on the cluster:
   `python -m hisrag.normalize requirements --verify-pilot` → review `requirements_verify_pilot.csv` (120 tags, top 60 by
   mentions + 60 random); if good, the full run, then the section "Tags checked without context" in notebook 03.
   Detail check (done): spelling folding and cue words cut the drops from 2,797 to 2,459; the remaining drops are right
   ("Hauptfach"/"Nebenfach" 1,814 mentions, "in Wort und Schrift", "bevorzugt" come from the ad, not the phrase). If
   Hauptfach/Nebenfach is needed, it has to come from a per-ad rule on the text, not the phrase dictionary.
2. **Step 6: salary.** Columns `salary`, `salary_period`, `unspecific_salary`, `verpflegung`, `salary_importance`
   (spans like `"von 10 fl CM"`, `"Monats⸗lohn"`, `"Quartiergeld"`, `"Wohnung, ganzer Verköstigung"`). Currency depends
   on date (Gulden CM until 1858, ö.W. after, Kronen from 1892/1900, Schilling 1925, Reichsmark 1938–45, Schilling
   again). Nominal amounts only in the PoC. Can be built and tested locally; LLM only for what the rules miss.
3. Steps 7 onward as in the table.

Smaller known issues: one hallucinated company name from context in step 5; single-occurrence OCR garbles in step 4
("Applent", "Praschneiderin") are mapped with guesses; the generic "Lehrling" category depends on its example.

## Practical notes for development

* Windows machine, Git Bash and PowerShell. Long Python edits with quotes or backslashes break inside bash heredocs:
  write the edit script to the scratchpad with the Write tool and run it.
* Notebooks are generated with `nbformat` from a small script and executed locally (with fake tables where the real
  ones only exist on the cluster) before pushing.
* DuckDB quirks met so far: views cannot take prepared parameters; `QUALIFY` does not combine with `GROUP BY ALL`;
  `USING SAMPLE` is applied before `WHERE`.

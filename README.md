# hisrag: RAG over historical job advertisements

Current state, decisions and next steps: [docs/STATUS.md](docs/STATUS.md).

Proof of concept for question answering over historical newspaper job ads (Wiener Zeitung subset, 1850–1950),
using the extracted metadata (positions, requirements, gender, salary …) alongside the text.
LLM and embeddings run on the DHinfra cluster API (`qwen3.5-397b` plus several embedding models).

## Setup (JupyterHub)

```bash
git clone https://github.com/Var3n/jobads_rag.git
cd jobads_rag
mamba env create -f environment.yml
mamba activate hisrag
python -m ipykernel install --user --name hisrag --display-name "hisrag"
nbstripout --install        # git ignores notebook outputs, so running a notebook never blocks `git pull`
```

API key: the first notebook cell asks for it if it is missing and stores it in `.env` (owner-only
permissions). `.env` is hidden in the JupyterHub file browser; that's expected. To set it by hand instead:
`python -c "import hisrag; hisrag.set_api_key()"`.

Data is not part of the repo. Put the CSV at `data/raw/wrz_extractions.csv`, or point to it in a
`config.local.yaml` (gitignored), which overrides any key of `config.yaml`:

```yaml
paths:
  raw_csv: /path/on/cluster/wrz_extractions.csv
```

Updating after new commits: `git pull`. If git says a notebook would be overwritten (a clone without
`nbstripout --install`), discard its outputs with `git checkout -- notebooks/<name>.ipynb` first.
After a pull, restart the kernel; the package is installed in editable mode, so nothing needs reinstalling.

## Pipeline

**Step 1: import.** Reads the extraction CSV and writes one row per region to `data/ads/newspaper=…/year=…/`
(Parquet). It parses all span columns, attaches gender to each position, drops exact duplicate regions, and
links each heading to the ad directly below it (`heading_text`, often the job title). It prints a report.

```bash
python -m hisrag.ingest                      # uses paths.raw_csv
python -m hisrag.ingest other_paper.csv      # one CSV per newspaper; re-running replaces its partitions
```

**Step 2: text normalization and quality flags.** Writes `data/derived/ad_text/`: `text_norm` / `heading_norm`
(for search only: `ſ`→s, line-break hyphens rejoined, quotes unified; historical spellings are kept),
a language guess (`de`/`it`/`fr`), raw quality metrics, and these flags:

| Flag | Meaning | Wiener Zeitung |
|---|---|---|
| `flag_pc_repetition` | post-correction looped or duplicated text | 278 |
| `flag_pc_expanded` | post-correction >1.3× as long as the OCR | 446 |
| `flag_pc_unsupported` | <60 % of the corrected text is supported by the OCR: reconstructed from noise, possibly invented | 268 |
| `flag_too_short` | under 30 characters | 186 |
| `flag_death_register` | entry of the Vienna death register, not an ad (1860s–1890s) | 1,184 |

```bash
python -m hisrag.normalize text
```

**Step 3: repeated printings.** Ads ran for several days or weeks (official notices usually three times).
Writes `data/derived/ad_dups/`: `dup_cluster_id` (the canonical region of each ad), `dup_cluster_size`,
`is_canonical`, and the run's first/last date. Candidates come from MinHash on character 5-grams (same
newspaper, within 60 days); each pair is then checked word by word, because template notices for different
places can be 95 % identical. Headings, death register entries, too-short and `pc_unsupported` regions stay
single. On the Wiener Zeitung sample: 52,823 regions → 44,291 distinct ads; the longest run is 64 printings
over two years.

```bash
python -m hisrag.normalize dedup
```

Count distinct ads with `WHERE is_canonical`, printings without it.

`notebooks/01_quality_review.ipynb` shows flagged regions and clusters next to their scanned clippings for checking.

**Step 4: position dictionary (uses the LLM).** Collects every distinct position form (extracted spans plus
headings; ~6,100 on the Wiener Zeitung sample) and has Qwen normalize each form once, 25 per request with one
context snippet each. Per form: zero or more entries with `term` (historical title, e.g. Unterlehrerin,
Commis), `lemma` (gender-neutral base for grouping), `modern` (today's equivalent, Commis → Handlungsgehilfe),
`gender_form` (m / f / m/f / n) and `category` (16 fixed categories). `hisco_code` is reserved for the HISCO
matching. Writes `data/derived/position_dict/` (one row per form) and `data/derived/ad_positions/` (one row per
position mention in an ad, from spans or the linked heading).

```bash
python -m hisrag.normalize positions --pilot   # ~80 forms → data/derived/pilot/positions_pilot.csv (a few minutes)
python -m hisrag.normalize positions           # full run, ~45 min at 16 parallel requests
```

Reasoning (`--thinking`) was tested in the pilot and is off by default: ~1,450 output tokens per form (≈11 h for
the full run instead of ~45 min) for answers that differed mainly in category choices, not clearly for the
better. Because each form is judged from one context, a lemma's categories are unified afterwards when one
category has at least 75 % of its mentions (Amtsdiener, Assistent); generic titles that genuinely occur in several
fields (Adjunct, Aufseher) keep per-form categories. A batch whose answer cannot be parsed is split in half and
retried. `notebooks/02_positions_review.ipynb` shows low-confidence forms, the forms behind a lemma, and first
counts per decade.

Responses are cached, so re-running after an interruption only sends what is missing.

**Step 5: requirement tags (uses the LLM).** Maps every distinct phrase of the columns `job_specific`,
`background`, `language`, `activity`, `attitude_towards_work` and `interpersonal` (~22,500 on the Wiener Zeitung
sample, death register excluded) to zero or more tags `dimension` / `value` / `detail`. The 28 dimensions in six
groups (Person, Qualifikation, Eigenschaften, Stelle, Bewerbung, Sonstiges) are defined in
[`vocab/requirements.yaml`](vocab/requirements.yaml): demographic dimensions and languages have closed value
lists, skills and duties free short values. Application formalities ("gehörig instruierte Gesuche") and a school's
language of instruction are kept apart from requirements of the person. Editing the YAML changes the prompt
version, so the next run re-maps everything. Writes `data/derived/requirement_dict/` and
`data/derived/ad_requirements/` (one row per phrase mention and tag).

```bash
python -m hisrag.normalize requirements --pilot   # 20 phrases per column → data/derived/pilot/requirements_pilot.csv
python -m hisrag.normalize requirements --verify-pilot  # 120 tags checked without context → requirements_verify_pilot.csv
python -m hisrag.normalize requirements           # full run, ~560 + ~150 requests (~1 h 50 min + ~20 min)
```

Three checks run after the model: a `detail` is kept only if its words occur in the phrase, allowing historical
spellings (the model's original is kept as `detail_raw` for auditing); values of closed-list dimensions are marked
`in_vocab`; and tags whose value does not occur in the phrase are checked again by the model **without context**
(`verified`), because the mapping sometimes tags what an example ad says ("verheirathet" → kinderlos). Rejected tags
stay in `requirement_dict` with `verified = false` but give no rows in `ad_requirements`. Re-running the step after a
code change is fast, since unchanged requests come from the response cache; editing the YAML re-maps everything.
`notebooks/03_requirements_review.ipynb` shows top values per dimension, dropped details, values outside the lists, and a
first comparison of personal requirements in ads for female vs. male position titles.

**Step 6: salary (rules, LLM only for leftovers).** Amounts in the `salary` spans ("600 fl.", "1200 K", "367 fl. 50 kr.", "1400—1600 K") are read by rules from a
window of ad text around each span: the currency standard of Gulden (C. M. / ö. W. when stated, otherwise by date: CM
until October 1858), what the amount pays for (`component`: gehalt, lohn, remuneration, taggeld, zulage,
quartiergeld, kaution, pension) and its period (stated, or assumed yearly for salaries and allowances). Amounts stay
nominal. Only the few spans the rules cannot read (numbers in words, OCR garbles, ~500 forms) go to the LLM. Writes
`data/derived/ad_salary/` (one row per amount) and `data/derived/ad_pay/` (per ad: main pay as min–max of its
Gehalt/Lohn/Remuneration/Taggeld amounts, without Kaution or Pension, plus benefit flags such as `benefit_wohnung`,
`benefit_kost`, `benefit_kleidung` from the `verpflegung` and `unspecific_salary` spans).

```bash
python -m hisrag.normalize salary --pilot   # 200 rule-read amounts + 40 LLM leftovers → data/derived/pilot/salary_pilot.csv
python -m hisrag.normalize salary           # full run, rules in seconds, ~13 LLM requests
```

`notebooks/04_salary_review.ipynb` shows coverage, samples per component, the currency standard check, median pay per
decade and position category, and benefits per decade.

**Step 7: clean table (SQL, no LLM).** Joins steps 1–6 into `data/derived/ad_clean/`, one row per ad: text, language,
quality flags and warning, repeated-printing cluster, position terms/lemmas/categories and gender of the wording,
requirement tags, main pay, all amounts and benefit flags. `searchable` excludes death-register entries and bare
headings; `countable` additionally keeps only the canonical printing and drops text invented by the post-correction.
Counts of ads use `countable`.

```bash
python -m hisrag.normalize clean
```

`notebooks/05_clean_table.ipynb` has the sanity plots per decade (ads, kind of ad, coverage of positions, tags and pay,
quality flags, language, position categories) with the table under each plot.

**Step 8: retrieval comparison (LLM for questions and judgments, embeddings).** From ~300 seed ads (countable, German,
with a position, spread over decades) the LLM writes research questions in modern German, each with a relevance
criterion; the seed ad is one of several relevant ads. Every method returns its top 10 (one printing per ad); the LLM
judges every pooled ad against question and criterion without knowing the method (2 relevant, 1 partly, 0 not).
Compared: BM25 on spelling-folded words, each configured embedding model, and BM25 + model hybrids (reciprocal rank
fusion), on the ad text (`raw`) and on the ad plus its normalized fields (`enriched`). Measures: precision@10,
nDCG@10, recall against all relevant ads in the pool. Embeddings are stored in `data/embeddings/<variant>/<model>/`
in chunks of 1,024 ads, so an interrupted run resumes.

```bash
python -m hisrag.eval queries --pilot           # 30 questions → data/eval/queries_pilot.csv for review
python -m hisrag.eval embed --pilot             # first chunk per model, estimates the full time
python -m hisrag.eval queries                   # 300 questions → data/eval/queries.parquet
python -m hisrag.eval embed --variant all       # all models, both texts (or --models a,b / --variant raw)
python -m hisrag.eval score --pilot             # judge 20 questions → data/eval/judgments_pilot.csv for review
python -m hisrag.eval score                     # all questions: runs, judgments, scores per method
```

Query the result from Python or a notebook:

```python
from hisrag.data import query
query("SELECT decade, label, count(*) AS n FROM ads GROUP BY ALL ORDER BY ALL")
query("SELECT a.text, t.lang FROM ads a JOIN ad_text t USING (ad_id) WHERE t.n_flags = 0 LIMIT 5")
```

## Layout

| Path | Contents |
|---|---|
| `hisrag/llm/` | API client: rate limiting, response cache, usage log, structured output, embeddings; `FakeOpenAI` for offline tests |
| `hisrag/ingest/` | Step 1: CSV → Parquet |
| `hisrag/normalize/` | Steps 2–6: text normalization, dedup, position/requirement dictionaries, salary |
| `hisrag/index/` | Steps 8–9: embeddings, hybrid search index |
| `hisrag/agent/` | Steps 10–12: tools, agent loop, playground |
| `hisrag/graph/` | Step 14: concept graph |
| `hisrag/eval/` | Synthetic retrieval checks, citation checker, interaction log |
| `hisrag/data.py` | Parquet storage; DuckDB views `ads` and one per derived table (`ad_text`, …) |
| `notebooks/` | Notebooks to run on the cluster; they only call package code |

## Tests

```bash
pytest
```

The tests run offline against `FakeOpenAI` and do not need an API key.

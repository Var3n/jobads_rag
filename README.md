# hisrag: RAG over historical job advertisements

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

`notebooks/01_quality_review.ipynb` shows flagged regions next to their scanned clipping for checking.

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

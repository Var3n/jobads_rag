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
cp .env.example .env        # then put your DHINFRA_API_KEY in .env
```

Data is not part of the repo. Put the CSV at `data/raw/wrz_extractions.csv`, or point to it in a
`config.local.yaml` (gitignored), which overrides any key of `config.yaml`:

```yaml
paths:
  raw_csv: /path/on/cluster/wrz_extractions.csv
```

Updating after new commits: `git pull` (the package is installed in editable mode, so restart the kernel and you're done).

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
| `notebooks/` | Notebooks to run on the cluster; they only call package code |

## Tests

```bash
pytest
```

The tests run offline against `FakeOpenAI` and do not need an API key.

# How To: Querying the OLMo Index and Running Your Own Experiments with InfiniSearch

This guide explains, based on the code in `python_scripts/`, how to

1. **run your own queries against the OLMo 2 pre-training corpus** (the 4.6T-token infini-gram index used in the qualitative analysis, §5 of the paper), and
2. **run your own retrieval experiments** with the full pipeline on a local infini-gram index (TREC RAG 2024 / MS MARCO v2.1, BEIR, or your own corpus), including how to set the hyperparameters.

---

## Contents

0. [Overview of the pipeline and the code](#0-overview-of-the-pipeline-and-the-code)
1. [Setup](#1-setup)
2. [Step 1 for every use case: keyword expansions](#2-step-1-for-every-use-case-keyword-expansions)
3. [Part A: Querying the OLMo 2 index through the infini-gram API](#3-part-a-querying-the-olmo-2-index-through-the-infini-gram-api)
4. [Part B: Running experiments on a local index](#4-part-b-running-experiments-on-a-local-index)
5. [Hyperparameters](#5-hyperparameters)
6. [Research recipes](#6-research-recipes)
7. [Troubleshooting](#7-troubleshooting)

---

## 0. Overview of the pipeline and the code

```
query ──► LLM keyword expansion ──► CNF pieces ──► peek (count) & grab ──► find_cnf ──► documents
                                                                                         │
                         cross-encoder top-100  ◄──  bi-encoder top-1000  ◄──────────────┘
```

| Step | What happens | Code |
|---|---|---|
| Keyword expansion | An LLM returns `KEY_ENTITIES` (each with 5 expansions) and 10 `ASSOCIATED_TERMS` (paper §3.1, prompt in App. A.1) | `batch_keywords.py` (Anthropic Batch API), `openrouter_keywords.py`, `ollama_keywords.py` |
| CNF pieces | Each term becomes a CNF *piece*: one AND-clause per content word, each clause an OR over case variants (`vicarious OR Vicarious`) | `progressive_queries._make_term_piece`, `pipeline._build_pieces_from_expansion` |
| Query construction | Count every piece, keep the specific ones, build 2-term AND combinations, cut to the document budget (paper §3.2) | `peek_grab_v2.peek_and_grab_v2` |
| First-stage retrieval | Execute the CNF queries with `find_cnf` | `adaptive_queries.run_adaptive` |
| Document resolution | Map pointers to documents, read metadata and tokens, decode text | `resolve_documents.resolve_all_queries` (local index) |
| OLMo / public API | Same expansion → peek & grab → retrieval flow against the hosted infini-gram index | `api_experiments.py` |
| Semantic filtering | Bi-encoder `all-MiniLM-L6-v2`, keeps top 1000 | `biencoder_rank.rerank_biencoder` |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-12-v2`, keeps top 100 | `crossencoder_rank.rerank_crossencoder` |
| Evaluation | pytrec_eval nDCG, Recall, P, MRR, MAP per stage | `ir_metrics.compute_metrics`, `pipeline.evaluate`, `recalc_metrics.py`, `trec_output.py` |
| Orchestration | YAML config → full run with output files | `run_experiment.run_from_config` |
| Qualitative ranking | Cross-encoder ranking of retrieved documents per claim, written to CSV | `qual_analysis.py` |
| BEIR | Download, convert, index a BEIR dataset and write a config | `beir_setup.py` |

Data shipped with this material:

| Path | Content |
|---|---|
| `topics_rag24_test.txt` | 301 TREC RAG 2024 test topics (`qid<TAB>query`) |
| `qrels.rag24.test-umbrela-all.txt` | UMBRELA qrels for those topics (TREC format) |
| `extension_files/*.jsonl` | Pre-computed keyword expansions for all 301 topics, one file per LLM (Opus 4.6, Sonnet 4.6, Llama 3.3 70B, Gemma 4 31B, DeepSeek V4 Pro, Qwen 3.6 27B) |
| `extension_files/api_expansions.jsonl` | Expansions for the 11 claims used in the OLMo case study (§5) |
| `qualitative_analysis/*.csv` | Full cross-encoder rankings of everything retrieved from OLMo for each claim |
| `example_config.yaml` | Template experiment configuration |
| `token_example.ini` | Template for API tokens |

---

## 1. Setup

### 1.1 Install

```bash
pip install -r requirements.txt
# optional, only for BEIR downloads via the beir package (beir_setup.py falls back to urllib otherwise)
pip install beir
```

The pipeline runs on CPU. A GPU only speeds up the bi- and cross-encoder.

`spacy` is required because `llm_keyword_filter.py` and `phrase_extraction.py` import it. The pipeline itself doesn't load a spaCy model, so `python -m spacy download en_core_web_sm` is only needed if you call those modules' own functions.

### 1.2 Python path and working directory

The scripts import each other by module name (`from pipeline import ...`), and all default paths (`./topics_rag24_test.txt`, `token_file.ini`) are relative to the **repository root**. Run everything from the root with `python_scripts/` on the path:

```bash
cd supplementary_material_infinisearch
export PYTHONPATH=$PWD/python_scripts
```

or, in a notebook started from the root:

```python
import sys; sys.path.insert(0, "python_scripts")
```

### 1.3 Tokens

Copy `token_example.ini` to `token_file.ini` in the repository root and fill in what you need (`utils.get_token` reads `KEY='value'` lines):

```ini
HF_TOKEN='hf_...'                 # for the gated Llama-2 tokenizer
ANTHROPIC_API_KEY='sk-ant-...'    # batch_keywords.py, api_experiments.py
OPENROUTER_API_KEY='sk-or-...'    # openrouter_keywords.py, api_experiments.py (non-Claude models)
OPENAI_API_KEY='sk-...'           # optional, api_experiments.py fallback for OpenAI models
```

> **Note:** `batch_keywords.py` sets `ANTHROPIC_API_KEY` from `token_file.ini` when it is imported, and `openrouter_keywords.py` / `ollama_keywords.py` import the prompt from it. Keep an `ANTHROPIC_API_KEY` line in `token_file.ini` (a placeholder is fine if you only use OpenRouter or Ollama).

### 1.4 Tokenizer

Both the MS MARCO index and the OLMo index are built on **Llama-2 tokens**. The code loads `meta-llama/Llama-2-7b-hf`, which is gated on Hugging Face: accept the license on the model page, then either run `huggingface-cli login` or `export HF_TOKEN=...`.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False
)
```

---

## 2. Step 1 for every use case: keyword expansions

The retriever never sees the raw query. It only sees the LLM's keyword expansion, so this step comes first.

### 2.1 Input format (topics)

A TSV file with one query per line (`utils.load_topics` also accepts JSONL with `id`/`title`):

```
my-001	What is vicarious trauma and how can it be coped with?
my-002	Therapists can experience vicarious trauma
```

### 2.2 Output format (expansions)

One JSON object per line, keyed by `qid`:

```json
{"qid": "my-001",
 "KEY_ENTITIES": {
    "vicarious trauma": ["secondary traumatic stress", "compassion fatigue", "vicarious traumatization", "indirect trauma", "empathic distress"],
    "coping":           ["self-care strategies", "resilience building", "stress management", "prevention strategies", "protective factors"]},
 "ASSOCIATED_TERMS": ["burnout prevention", "trauma stewardship", "helping professionals", "..."]}
```

How the pipeline reads this (`pipeline._build_pieces_from_expansion`):

- Each **key entity name is itself used as a search term**, in addition to its expansions.
- The **order of `KEY_ENTITIES` matters**: the *first* entity is treated as the most discriminative. Only it is combined with associated terms (see §5).
- `ASSOCIATED_TERMS` (or `ASSOCIATED`, as named in the prompt) is a flat list.
- `VERBS`, present in some older files, is ignored.
- Stopwords are removed from each term. A multi-word term that is reduced to a single content word is dropped. A term containing a word that never occurs in the index is dropped.

### 2.3 Generating expansions

All three generators use the same `SYSTEM_PROMPT` (defined in `batch_keywords.py`, identical to Figure 4 in the paper) with temperature 0.

**Anthropic Batch API** (what we used for Claude Opus 4.6 / Sonnet 4.6):

```bash
python python_scripts/batch_keywords.py submit  --topics my_topics.txt --model claude-opus-4-6
python python_scripts/batch_keywords.py wait    --batch-id msgbatch_XXXX
python python_scripts/batch_keywords.py collect --batch-id msgbatch_XXXX --output my_expansions.json
# writes my_expansions.json and my_expansions.jsonl (+ my_expansions_errors.json)
```

Note that the CLI default model is `claude-sonnet-4-6`. Pass `--model claude-opus-4-6` to reproduce the paper's main setting.

**OpenRouter** (open-weight models, as for Llama 3.3 70B and Gemma 4):

```bash
python python_scripts/openrouter_keywords.py run --topics my_topics.txt \
    --model meta-llama/llama-3.3-70b-instruct --workers 4
# -> keyword_files/meta-llama_llama-3.3-70b-instruct.jsonl  (resumable; re-run to fill gaps)

python python_scripts/openrouter_keywords.py compare --topics my_topics.txt \
    --models google/gemma-4-31b-it,meta-llama/llama-3.3-70b-instruct --output-dir keyword_files/
```

**Ollama** (fully local):

```bash
ollama serve &   # and: ollama pull <model>
python python_scripts/ollama_keywords.py run --topics my_topics.txt --model llama3.3:70b
# -> keyword_expansions_llama3.3_70b.jsonl
```

Entries that fail to parse are stored as `{"qid": ..., "raw": "..."}` and have no `KEY_ENTITIES`, so the pipeline effectively skips them. Check for them before running retrieval.

You can also write or edit expansions by hand. Any file in the format above works.

---

## 3. Part A: Querying the OLMo 2 index through the infini-gram API

The OLMo 2 training data (~3.2B documents, 4.6T tokens) is far too large to download. It is served by the public infini-gram API at `https://api.infini-gram.io/`. Relevant index names:

| Index | Corpus |
|---|---|
| `v4_olmo-2-0325-32b-instruct_llama` | Full OLMo-2-0325-32B-Instruct training data (pre-, mid-, post-training). Used in the paper. |
| `v4_olmo-2-1124-13b-instruct_llama` | Full OLMo-2-1124-13B-Instruct training data |
| `v4_olmo-mix-1124_llama` | OLMo 2 pre-training mix only |

All of these use the Llama-2 tokenizer, so CNF pieces built with `tokenizer` above can be sent directly as token IDs.

### 3.1 Quick start: raw API calls

You don't need the pipeline to try a single CNF query. In the string syntax, **`OR` binds tighter than `AND`**:

```python
import requests
API = "https://api.infini-gram.io/"
IDX = "v4_olmo-2-0325-32b-instruct_llama"

# 1) How many matches?
r = requests.post(API, json={"index": IDX, "query_type": "count",
                             "query": "vicarious trauma AND hypervigilance",
                             "max_diff_tokens": 100}).json()
print(r["count"], r["approx"])

# 2) Get pointers to matching documents
r = requests.post(API, json={"index": IDX, "query_type": "find_cnf",
                             "query": "vicarious trauma AND hypervigilance",
                             "max_diff_tokens": 100}).json()
ptrs_by_shard = r["ptrs_by_shard"]          # one list of pointers per shard

# 3) Fetch one document
s = next(i for i, p in enumerate(ptrs_by_shard) if p)
doc = requests.post(API, json={"index": IDX, "query_type": "get_doc_by_ptr",
                               "s": s, "ptr": ptrs_by_shard[s][0],
                               "max_disp_len": 1000}).json()
print("".join(span[0] for span in doc["spans"]))
```

API limits to keep in mind (from the infini-gram documentation):

- `max_clause_freq` can be at most **500,000** (the API default is 50,000). Clauses with more matches are **subsampled**, so counts are approximate (`approx: true`), and `find_cnf` only returns pointers from the sample.
- `max_diff_tokens` can be at most **1,000** (default 100).
- `get_doc_by_ptr` returns one document per request, with `max_disp_len` ≤ 10,000 tokens. Fetching documents is therefore the slow part, which is why the paper caps retrieval at **300 documents per query** for the OLMo case study.
- No rate limits are documented, and availability isn't guaranteed. Wrap calls in retries.

### 3.2 Full InfiniSearch retrieval against the API: `api_experiments.py`

`api_experiments.py` runs the whole flow against the public API in one call: keyword expansion → peek & grab (the same phases as the local pipeline, §5.1) → document retrieval. This is the code behind the OLMo case study in §5 of the paper.

```python
import sys; sys.path.insert(0, "python_scripts")
from transformers import AutoTokenizer
from api_experiments import run_experiment

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          add_bos_token=False, add_eos_token=False)

result = run_experiment(
    query="Therapists can experience vicarious trauma",
    model="claude-opus-4-6",                         # expansion LLM (default: claude-sonnet-4-20250514)
    tokenizer=tokenizer,
    index="v4_olmo-2-0325-32b-instruct_llama",       # default
    cache_path="./my_expansions.jsonl",              # expansions cache; results go to ./my_expansions_results.pkl
    max_standalone_key=1500,
    max_budget=20000,
    max_queries=10,                                  # retrieve from 10 CNF queries ...
    max_docs_per_query=30,                           # ... with up to 30 documents each (= 300 max)
    max_disp_len=500,                                # tokens returned per document
    api_delay=0.5,                                   # seconds between API calls
)
print(result["n_documents"], "documents")
for d in result["documents"][:5]:
    print(d["doc_id"], d["from_queries"], d["text"][:120])
```

**What happens inside**

1. **Expansion** (`get_expansions`). The `SYSTEM_PROMPT` from `batch_keywords.py` is sent to the given model: Anthropic for `claude-*` names, OpenRouter for everything else (OpenAI as a fallback for `gpt-*` / `o1` / `o3` models when no OpenRouter key is set). The result is cached in `cache_path` as one JSON line `{"text": <query>, "model": <model>, "KEY_ENTITIES": ..., "ASSOCIATED_TERMS": ...}`. A query/model pair already in the cache isn't sent to the LLM again, so you can also write your own expansions into this file by hand. `extension_files/api_expansions.jsonl` is the cache for the 11 claims in the paper.
2. **Pieces** (`_build_pieces`). Each term becomes a *contiguous n-gram* with three case variants (as written, lowercase, title case) OR'ed together. Unlike the local pipeline, multi-word terms are not split into proximity-AND clauses.
3. **Peek & grab** (`peek_and_grab`). All pieces and 2-term combinations are counted with the API `count` query, and pieces and combinations are selected with the standalone thresholds, budget cut and fill described in §5.1. Combinations use the API's default `max_diff_tokens` (100).
4. **Retrieval** (`retrieve_documents`). The selected queries are sorted by estimated count (largest first) and cut to `max_queries`. For each query, `find_cnf` (with `max_clause_freq` = 500,000) returns pointers. Up to `max_docs_per_query` of them are sampled evenly across the pointer list and fetched with `get_doc_by_ptr`. Documents are deduplicated by `doc_id`, which is `<source file>:<line>` from the document metadata (e.g. `dclm-0243.json.zst:692182`), and `from_queries` records every CNF query that found each document.
5. **Saving** (`save_results`). The result is added to `<cache_path without .jsonl>_results.pkl`, replacing any earlier result for the same query and model. If that query/model pair already has documents in the pickle, `run_experiment` returns the stored result without calling the API, so interrupted runs can be resumed.

The returned dict (and each entry of the pickle) contains `query`, `model`, `expansion`, `documents` (each with `doc_id`, `doc_ix`, `doc_len`, `text`, `from_queries`), `n_documents`, `all_queries` (the selected CNF queries with `description`, `estimated_count`, `level`), `all_key_pieces` / `all_assoc_pieces` (the count of every term), `grabbed`, `combo_queries`, `filled`, the budget totals, and `time`.

**Settings used in the paper's OLMo case study.** These can be read back from the stored results: index `v4_olmo-2-0325-32b-instruct_llama`, expansion model `claude-opus-4-6`, `max_standalone_key` = 1500, `max_standalone_assoc` = 750 (the default in `peek_and_grab`), `max_budget` = 20,000, `max_queries` = 10, `max_docs_per_query` = 30. That gives at most 300 documents per claim, and 206–298 unique documents in practice.

`run_experiment` exposes `max_standalone_key` and `max_budget`. To change `max_standalone_assoc` or `max_assoc_combo`, call the steps yourself:

```python
from api_experiments import get_expansions, peek_and_grab, retrieve_documents, save_results

exp = get_expansions(query, model="claude-opus-4-6", cache_path="./my_expansions.jsonl")
res = peek_and_grab(exp, tokenizer, max_standalone_key=1500, max_standalone_assoc=750,
                    max_budget=20000, max_assoc_combo=50000)
res["documents"] = retrieve_documents(res["all_queries"], tokenizer,
                                      max_docs_per_query=30, max_queries=10)
res.update(query=query, model="claude-opus-4-6", expansion=exp, n_documents=len(res["documents"]))
save_results(res, path="./my_expansions_results.pkl")
```

### 3.3 Ranking the retrieved documents

`qual_analysis.py` scores every retrieved document against its query with the cross-encoder (`ms-marco-MiniLM-L-12-v2`), deduplicates by text, and writes one CSV per query:

```python
from qual_analysis import rank_all_results, rank_and_save

# All queries in a results pickle (this is how qualitative_analysis/*.csv was produced)
rank_all_results("./my_expansions_results.pkl", output_dir="./my_qual_analysis/", top_k=50)

# Or a single result
rank_and_save(result["query"], result["documents"], output_dir="./my_qual_analysis/", top_k=50)
```

Each CSV has the same columns as the files in `qualitative_analysis/`:

| Column | Meaning |
|---|---|
| `rank`, `score` | Cross-encoder rank and score |
| `query` | Your query/claim |
| `doc_id` | `<source file>:<line>` in the OLMo training data |
| `from_queries` | Which CNF queries retrieved the document, e.g. `(counselors) AND (vicarious trauma)`. Useful to see whether a hit came from an expansion rather than the original wording. |
| `text` | First `max_text_len` characters of the document (2,500 in `rank_all_results`) |

**Tips for OLMo queries**

- **A batch of claims**: loop over them with `run_experiment` and the same `cache_path`, then call `rank_all_results` once. In the paper, a model output was split into 11 atomic claims (by Claude Opus 4.6, manually verified), and each claim was a separate query.
- **Comparing against other sources** (as we did with OlmoTrace in §5.1): pass their texts to the ranker so they are scored alongside your documents, e.g. `rank_all_results(..., extra_texts=olmotrace_snippets)`. Extra texts appear in the CSV with an empty `doc_id`, so you can read off where they would have ranked.
- **Latency** is dominated by the number of API calls (a `count` per term and per combination, then one `get_doc_by_ptr` per document) plus `api_delay` between calls. Expect a few minutes per query. Fewer key entities or associated terms, or a lower `max_queries` / `max_docs_per_query`, makes it faster.
- **Longer document texts**: raise `max_disp_len` (API maximum 10,000 tokens).
- **Text spacing**: the API returns documents as text spans split at the matches. `fix_doc_spacing(documents)` repairs missing spaces around matched keywords if needed.
- **Other indices**: set `index=` to any name from the table above, e.g. `v4_olmo-mix-1124_llama` for the pre-training mix only. Indices built with a different tokenizer (such as `v4_dolmasample_olmo`) need the matching tokenizer.

---

## 4. Part B: Running experiments on a local index

For benchmark experiments you need a local infini-gram index of the corpus, because evaluation needs every document in the pool and their IDs.

### 4.1 Build an index

**Any corpus.** Write a JSONL file in which every line has a `text` field and a `docid` field whose values match your qrels:

```json
{"text": "Title. Body text ...", "docid": "doc-123"}
```

Then index it with Llama-2 tokens and metadata (`--add_metadata` is required: `resolve_documents.py` reads `metadata["docid"]` from the index's metadata files):

```bash
python -m infini_gram.indexing --data_dir ./my_corpus/ --save_dir ./my_index/ \
    --tokenizer llama --cpus 16 --mem 64 --shards 1 --add_metadata --ulimit 1048576
```

See the [infini-gram indexing docs](https://infini-gram.readthedocs.io/en/latest/indexing.html) for sharding and memory requirements.

**TREC RAG 2024 (MS MARCO v2.1).** Index the *segmented* MS MARCO v2.1 corpus ([download](https://trec-rag.github.io/annoucements/2025-rag25-corpus/)) the same way, keeping the segment ID (e.g. `msmarco_v2.1_doc_13_1647729865#1_3617399591`) as the `docid`. This repository already contains the topics, qrels and expansions.

**BEIR.** `beir_setup.py` downloads a dataset, converts the corpus (`title. text` + `docid`), builds the index, writes `topics.txt` / `qrels.txt`, and creates a `config.yaml`:

```python
from beir_setup import setup_full
setup_full("scifact", "./beir/scifact/", cpus=16, mem=64)
```

The steps can also be run one at a time: `download_beir`, `convert_corpus`, `convert_queries_qrels`, `build_index`, `create_config`. Afterwards, generate expansions for `./beir/scifact/topics.txt` (§2.3), save them as `./beir/scifact/kiss.jsonl` (the path written to the config), and run the config. For the paper we used the BEIR test split and the same prompt and hyperparameters as for TREC RAG.

### 4.2 Write a config

Copy `example_config.yaml` into a new directory per experiment and edit it:

```yaml
# Paths
topics_path: ./topics_rag24_test.txt
qrels_path: ./qrels.rag24.test-umbrela-all.txt
expansions_path: ./extension_files/opus-4-6.jsonl
out_path: ./runs/opus_baseline/          # omit to write next to the config file

# Index
index_dir: /path/to/msmarco_segmented_index/
tokenizer_name: meta-llama/Llama-2-7b-hf

stages: [retrieval, biencoder, crossencoder]   # drop stages to skip them

biencoder_model: sentence-transformers/all-MiniLM-L6-v2
crossencoder_model: cross-encoder/ms-marco-MiniLM-L-12-v2

max_topics: null          # e.g. 10 for a quick test
top_k_list: [10, 100, 1000]

# Pipeline hyperparameters: see §5
max_standalone_key: 5000
max_standalone_assoc: 1000
prox_peek: 10
prox_cross: 200
max_clause_freq: 8000000
max_budget: 40000
max_doc_len: 20000
biencoder_top_k: 1000
crossencoder_top_k: 100
```

### 4.3 Run

```python
import sys; sys.path.insert(0, "python_scripts")
from run_experiment import run_from_config

results, eval_results = run_from_config("./runs/opus_baseline/config.yaml")
```

Loading a large index takes time. To run several configs against the same index, load the engine and tokenizer once and pass them in:

```python
from transformers import AutoTokenizer
from infini_gram.engine import InfiniGramEngine

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
eng = InfiniGramEngine(index_dir="/path/to/msmarco_segmented_index/", eos_token_id=tok.eos_token_id)

for cfg in ["./runs/budget_20k/config.yaml", "./runs/budget_40k/config.yaml"]:
    run_from_config(cfg, engine=eng, tokenizer=tok)
```

### 4.4 Outputs (in `out_path`)

| File | Content |
|---|---|
| `config_resolved.yaml` | The config after merging with all defaults. This is exactly what was run. |
| `results.pkl` | Everything per topic (see below) |
| `trec_run_pool.txt`, `trec_run_biencoder.txt`, `trec_run_crossencoder.txt` | Standard TREC run files per stage, usable with `trec_eval` / `ranx` |
| `per_query.csv` | Per stage and query: recall, nDCG, precision and found@k, MRR, pool hits |
| `metrics.csv` | Aggregated pytrec_eval metrics |
| `eval_summary.txt` | Printed evaluation tables |

The `pool` stage measures the first-stage retriever alone. Its recall is the recall ceiling for the rerankers (`R@Pool` in Table 6).

`results.pkl` is a list of dicts, one per topic, containing `pool_docs`, `executed_queries` (each with `description`, `estimated_count`, `cnt`, `level`), `biencoder_ranked`, `crossencoder_ranked`, `timings`, the `config`, and the full `peek_result`. `peek_result` holds the match count of every piece (`all_key_pieces`, `all_assoc_pieces`), what was grabbed, combined, or filled, and the budget totals. It's the main place to inspect *why* a topic did well or badly.

### 4.5 Single query in Python

```python
from pipeline import run_pipeline, DEFAULT_CONFIG
from utils import load_all_expansions
from sentence_transformers import SentenceTransformer, CrossEncoder

resources = {"engine": eng, "tokenizer": tok,
             "biencoder_model": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
             "crossencoder_model": CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")}
config = dict(DEFAULT_CONFIG, index_dir="/path/to/msmarco_segmented_index/",
              max_standalone_key=5000, max_standalone_assoc=1000, prox_cross=200,
              max_clause_freq=8_000_000, max_budget=40_000, max_doc_len=20_000)

exp = load_all_expansions("./extension_files/opus-4-6.jsonl")
r = run_pipeline("2024-145979", "what is vicarious trauma and how can it be coped with?",
                 exp["2024-145979"], config, resources, verbose=True)
for d in r["crossencoder_ranked"][:10]:
    print(round(float(d["score"]), 3), d["doc_id"], d["text"][:120])
```

Set `biencoder_model` / `crossencoder_model` to `None` to stop after first-stage retrieval.

---

## 5. Hyperparameters

### 5.1 How the query construction uses them

`peek_and_grab_v2` implements the three stages of §3.2, plus a fill step:

1. **Phase 1, standalone peek (Stage 1).** Every piece is counted (multi-word pieces with `max_diff_tokens = prox_peek`). Key pieces with `0 < count ≤ max_standalone_key` and associated pieces with `0 < count ≤ max_standalone_assoc` are *grabbed*: retrieved directly and excluded from combination. Pieces with count 0 are dropped.
2. **Phase 2, 2-term ANDs (Stage 2).** From the remaining pieces:
   - **key × key**: every pair of pieces from *different* key entities (needs ≥ 2 entities with remaining pieces);
   - **key × assoc**: pieces of the *first* key entity × remaining associated terms with count ≤ `max_assoc_combo`. A key × assoc combination is kept only if its count ≤ `max_standalone_key`.

   All combinations are counted with `max_diff_tokens = prox_cross`. Combinations with zero matches are dropped.
3. **Phase 3, budget cut (Stage 3).** If grabbed + combinations exceed `max_budget` (paper: *B*), combinations are added in order (key × key first, then key × assoc, each from tightest to broadest) until the budget is reached. Very cheap key × assoc combinations (count ≤ `max_standalone_assoc / 2`) are always kept, even if they exceed the budget.
4. **Phase 4, fill.** If `budget_fill` is on and the total is still under budget, the remaining unused standalone pieces are added from cheapest to broadest while they fit.

Counts are match counts, not unique documents, so the real pool is smaller than the budget (with *B* = 40,000 we retrieved 25,662 documents per query on average).

### 5.2 Reference table

Only keys in `pipeline.DEFAULT_CONFIG` are passed to the pipeline. Precedence is **`DEFAULT_CONFIG` < your YAML**, and any other key in the YAML that isn't an experiment-level setting is silently ignored. Always check `config_resolved.yaml`.

| Key | Paper | What it controls | `pipeline.py` default | Paper setting |
|---|---|---|---|---|
| `max_standalone_key` | *max_sa* | Count threshold for retrieving key-entity pieces on their own, and the upper bound for key × assoc combinations | 1,500 | 5,000 |
| `max_standalone_assoc` | | Count threshold for retrieving associated terms on their own. Half of it is the "always keep" threshold for cheap key × assoc combinations. | 750 | 1,000 |
| `prox_peek` | | Max token distance between the words of one multi-word term (`max_diff_tokens` for standalone pieces) | 10 | 10 |
| `prox_cross` | | Max token distance between the parts of a combined query (`max_diff_tokens` for 2-term ANDs). In combinations this applies to *all* clauses, including the words within each term. | 100 | 200 |
| `max_clause_freq` | | Clauses with more matches are subsampled by infini-gram (approximate counts and retrieval). Higher means exact but slower. **API maximum: 500,000.** | 2,000,000 | 8,000,000 |
| `max_budget` | *B* | Total match budget over all retrieval queries | 20,000 | 40,000 |
| `max_assoc_combo` | | Associated terms with more matches than this are not combined with the first key entity | 50,000 | 50,000 |
| `budget_fill` | | Whether Phase 4 fills unused budget with remaining standalone pieces | `True` | `True` |
| `max_doc_len` | | Tokens read per document from the local index (from the beginning) | 2,000 | 20,000 |
| `index_dir` | | Local index (set automatically from the experiment config) | | |
| `biencoder_top_k` | | Documents kept after the bi-encoder | 1,000 | 1,000 |
| `biencoder_batch_size` | | Encoding batch size | 256 | 256 |
| `biencoder_max_doc_length` | | Text is cut to `4 ×` this many **characters** before encoding (the model's own max sequence length also applies) | 512 | 512 |
| `crossencoder_top_k` | | Documents kept after the cross-encoder | 100 | 100 |
| `crossencoder_batch_size` | | Scoring batch size | 32 | 32 |
| `crossencoder_max_length` | | Text is cut to `4 ×` this many characters | 512 | 512 |

Experiment-level keys (read by `run_experiment.py`): `topics_path`, `qrels_path`, `expansions_path`, `out_path`, `index_dir`, `tokenizer_name`, `stages`, `biencoder_model`, `crossencoder_model`, `max_topics`, `top_k_list`.

Not exposed as hyperparameters, but easy to change in code:

- **Number of expansions** (5 per key entity, 10 associated terms): set in `SYSTEM_PROMPT` in `batch_keywords.py`.
- **Which entity counts as "first"**: the order in `KEY_ENTITIES`, i.e. the LLM's output order.
- **Stopword list** used when building pieces (`STOPWORDS` in `llm_keyword_filter.py`).
- **Case variants** per word: the word as given, lowercased, and capitalized (`progressive_queries._case_variants`).

### 5.3 Tuning advice

- **Recall too low?** Look at `peek_result["all_key_pieces"]`. If most pieces are "remain" with huge counts and few combinations fit, raise `max_budget` or `max_standalone_key`. If many pieces have count 0, the terms don't occur verbatim: check spelling and tokenization (word-initial tokens differ, see Limitations in the paper) or improve the expansions.
- **Pool too noisy / reranking too slow?** Lower `max_standalone_assoc` (associated terms produced most of the irrelevant standalone hits in our error analysis, §5.2), lower `max_budget`, or tighten `prox_cross`.
- **First-stage latency** grows with the number of CNF queries and with `max_clause_freq`. A lower `max_clause_freq` speeds up counting on broad terms at the cost of approximate counts.
- **Larger corpora** have higher counts for the same terms. When moving from MS MARCO to a pre-training-scale index, the standalone thresholds and the budget need to be set relative to the number of documents you can afford to process downstream (we used 300 documents for OLMo).
- Start with `max_topics: 10` and `stages: [retrieval]` to iterate on query construction quickly. Pool recall alone tells you whether the first stage works.

---

## 6. Research recipes

**Compare LLMs for keyword expansion (Table 1).** Create one config per file in `extension_files/` (or your own), with `stages: [retrieval]`, and compare pool recall in `per_query.csv` (mean, median, and the mean over the worst 10% of queries for CVaR).

**Ablations: key entities only / no associated terms (Opus_kws, Opus_eo).** Derive modified expansion files:

```python
import json
with open("extension_files/opus-4-6.jsonl") as fin, \
     open("opus_kws.jsonl", "w") as f_kws, open("opus_eo.jsonl", "w") as f_eo:
    for line in fin:
        e = json.loads(line)
        eo = dict(e, ASSOCIATED_TERMS=[])                        # expansions only
        kws = dict(eo, KEY_ENTITIES={k: [] for k in e["KEY_ENTITIES"]})  # entity names only
        f_eo.write(json.dumps(eo) + "\n"); f_kws.write(json.dumps(kws) + "\n")
```

Because entity names are always used as terms, empty expansion lists leave exactly the key entities.

**Budget / threshold sweeps.** Generate configs in a loop and reuse the loaded engine (§4.3):

```python
import yaml, os
base = yaml.safe_load(open("example_config.yaml"))
for B in [10_000, 20_000, 40_000, 80_000]:
    d = f"./runs/budget_{B}"; os.makedirs(d, exist_ok=True)
    yaml.dump(dict(base, max_budget=B, out_path=d), open(f"{d}/config.yaml", "w"))
    run_from_config(f"{d}/config.yaml", engine=eng, tokenizer=tok)
```

**Stronger rerankers.** Any sentence-transformers bi-encoder or cross-encoder name can go into `biencoder_model` / `crossencoder_model`. The paper's numbers with the small MiniLM models should be read as a lower bound.

**Own prompts.** Edit `SYSTEM_PROMPT` in `batch_keywords.py` (all three generators import it). Keep the output schema (`KEY_ENTITIES` as a dict of lists, `ASSOCIATED_TERMS` or `ASSOCIATED` as a list) so the pipeline can read it.

**BM25 baselines.** `pyserini` is included in the requirements for building Lucene baselines. We used the parameters of Pradeep et al. (2024) with the same cross-encoder. The `trec_run_*.txt` files can be evaluated with the same tools as Pyserini runs.

**Research on pre-training data (OLMo).** Use Part A with your own claims: for data attribution, fact-checking model outputs against the training data, or checking whether benchmark items appear in paraphrased form. `from_queries` in the output shows which expansion produced each hit, which helps separate verbatim matches from semantic ones.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'pipeline'` (or another script name) | `python_scripts/` is not on the path: see §1.2 |
| `TypeError` when importing `batch_keywords` / `openrouter_keywords` / `ollama_keywords` | `ANTHROPIC_API_KEY` missing from `token_file.ini` (set at import time): add it, a placeholder works |
| `OSError` / 401 when loading `meta-llama/Llama-2-7b-hf` | Gated model: accept the license and log in to Hugging Face (§1.4) |
| `[qid] No expansion, skipping` | The `qid` in the topics file has no line in the expansions file, or the LLM output was unparsable (`"raw"` entry) |
| `WARNING: 0 docs resolved` | Every piece had count 0 or exceeded all thresholds with no fitting combination. Inspect `peek_result`; raise `max_standalone_key` / `max_budget` or fix the expansions. |
| All `doc_id`s are `None`, evaluation is 0 | Index built without `--add_metadata`, or the metadata field isn't named `docid` |
| API returns an error about `max_clause_freq` or `max_diff_tokens` | Values above 500,000 / 1,000 aren't allowed |
| API calls time out, fail sporadically, or a claim returns few documents | Public service without uptime guarantees: failed calls count as 0 matches in `api_experiments.py`. Re-run the claim (delete its entry from the results pickle first) or raise `api_delay`. |
| Very slow on a local index | Index on a fast SSD, lower `max_clause_freq`, reduce `max_budget`; use `max_topics` while developing |

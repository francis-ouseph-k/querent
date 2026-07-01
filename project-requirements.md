---

# Requirements Document
## Digital Evaluation System — NL→SQL
**Version:** 2.0
**Date:** 2026-06-06
**Status:** Phase 1 Complete · Phase 2 In Progress

---

## 1. Overview

The system converts natural-language business questions into validated, read-only
PostgreSQL SQL queries. It is purpose-built for a university digital evaluation pipeline
with a 69-object schema containing views, indexes, foreign keys, partitioned tables, JSONB
fields, workflow-driven entities, and audit structures.

**Target users:** Faculty administrators querying examination and evaluation data without
SQL expertise.

**Example:** *"Show all scripts pending third evaluation in board 5 where deviation
exceeds threshold"* → validated PostgreSQL SELECT executed against read-only replica in
under 2 seconds.

**Deployment target:** Local 8 GB GPU environment. Fully local inference and retrieval.
No external LLM APIs.

---

## 2. Scope and Constraints

### 2.1 In Scope

| Item | Phase |
|---|---|
| Semantic DDL parsing and chunking | 1 |
| FK graph construction and BFS traversal | 1 |
| Dense + BM25 + graph hybrid retrieval | 1 |
| RRF fusion and context budget management | 1 |
| 7-section prompt assembly | 1 |
| JSON-constrained SQL generation | 1 |
| 12-step SQL validation pipeline | 1 |
| Retry and repair loop | 1 |
| CLI interface with dry-run default | 1 |
| Failure corpus logging (`:correct` mechanism) | 1 |
| Schema drift detection | 1 |
| Failure corpus preparation pipeline | 2 |
| Synthetic pair bootstrapping | 2 |
| QLoRA fine-tuning | 2 |
| Regression-guarded evaluation | 2 |
| GGUF export pipeline | 2 |

### 2.2 Explicitly Out of Scope

The following are deferred to Phase 3+ and must not be included in Phase 1 or Phase 2
deliverables:

- FastAPI / REST API layer
- Web-based user interface
- Conversational memory and multi-turn SQL refinement
- Intent decomposition (breaking one question into multiple sub-queries)
- Adaptive retrieval (dynamic strategy selection per query type)
- Schema summarisation and compression
- Domain-finetuned BGE embeddings
- 7B model evaluation
- Execution-plan-aware prompt generation

### 2.3 Hardware Constraints

- **8 GB VRAM ceiling** — drives model selection (3B), quantisation (Q4_K_M),
  context window management, and QLoRA training configuration.
- **Single GPU** — llama-server and Phase 2 training cannot share the GPU.
  llama-server must be stopped before Phase 2 training begins.
- **Local execution only** — no external LLM APIs at any phase.

### 2.4 Database Constraints

- **Read-only replica only** — the system must never connect to the primary database.
- **Never generate or execute DML/DDL** — enforced at generation (JSON schema), validation
  (AST + regex), and connection (PostgreSQL `default_transaction_read_only=on`) levels.
- **Row limit enforcement** — inject `LIMIT` via AST manipulation, not string append.
- **EXPLAIN cost gate** — reject queries above threshold before execution.
- **Statement timeout** — 30 seconds per connection.

---

## 3. Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11+ | Core implementation |
| LLM inference | Qwen2.5-Coder 3B Instruct Q4_K_M GGUF | — | SQL generation |
| LLM runtime | llama-server (llama.cpp) | pre-compiled | Local inference, JSON constrained output |
| Embedding model | BAAI/bge-small-en-v1.5 | 384-dim | Dense embeddings for Qdrant |
| Vector store | Qdrant | 1.7+ | Dense semantic retrieval |
| Keyword store | OpenSearch | 2.11+ | BM25 sparse retrieval |
| Graph engine | NetworkX | 3.2+ | FK graph and BFS traversal |
| SQL parser | sqlglot | 20.0+ | AST-based validation, manipulation |
| SQL formatter | sqlparse | 0.4.4+ | Lightweight formatting and syntax checks |
| DB driver | psycopg2 | 2.9.9+ | PostgreSQL ThreadedConnectionPool |
| Settings | pydantic-settings | 2.1+ | Type-validated env var configuration |
| Logging | structlog | 24.1+ | JSONL structured logging with request_id |
| CLI | Rich + prompt_toolkit | — | Syntax-highlighted terminal interface |
| Fine-tuning | transformers + peft + trl | pinned | QLoRA training (Phase 2 only) |
| Quantisation (training) | bitsandbytes | pinned | 4-bit NF4 for QLoRA (Phase 2 only) |
| GGUF conversion | convert_hf_to_gguf.py | llama.cpp source | HF → GGUF (Phase 2 export) |
| GGUF quantisation | llama-quantize.exe | pre-compiled | Q4_K_M compression (Phase 2 export) |
| API layer | FastAPI | — | **Deferred — Phase 3+** |

---

## 4. Phase 1 — Production NL→SQL System

**Status:** Complete.

### 4.1 Schema Ingestion Pipeline

#### 4.1.1 DDL Parsing

- Parse `digital_evaluation_schema_v10_4_1.sql` using sqlglot (PostgreSQL 16 dialect) in
  5 passes: tables → foreign keys → indexes → column comments → views and triggers.
- Filter noise (`django_migrations`, `auth_user`).
- Extract primary keys and foreign key constraints explicitly.
- Preserve structural and relational context: 69 DDL objects, ~150 FK edges,
  composite keys, and JSONB schema shapes.

> **⚠ Unresolved discrepancy (found during this correction pass, not fixed here):**
> `config/settings.py` currently hardcodes `ddl_path = "data/docs/digital_evaluation_schema_v10_4_1.sql"`
> — so this requirements doc's reference to v10.4.1 matches what the *code* actually
> ingests today. But `data/docs/` also contains `digital_evaluation_schema_v10_5.sql`,
> with v10.4.1 sitting in `data/docs/archive/` alongside other superseded versions —
> the naming convention strongly suggests v10.5 is the current authoritative schema and
> v10.4.1 is stale. If that's correct, `settings.py`'s `ddl_path` needs updating (and a
> re-ingestion run), not just this document. If v10.4.1 is intentionally still the Phase 1
> ingestion target for some other reason, that reasoning isn't captured anywhere and should
> be. Either way, the object counts in §8 (69 DDL objects, ~150 FK edges, ~212 chunks) are
> only accurate for whichever version ends up configured — re-verify them once this is
> resolved, don't carry them forward on trust.

#### 4.1.2 Semantic Chunk Generation

Generate 11 chunk types from `TableInventory`. Use exact `ChunkType` enum values:

| ChunkType | Source | Contents | Indexed in |
|---|---|---|---|
| `TABLE` | CREATE TABLE | Column definitions, types, PKs, FKs, comments | Qdrant + OpenSearch |
| `VIEW` | CREATE VIEW | View definition, business purpose, referenced tables | Qdrant + OpenSearch |
| `FK_MAP` | FK constraints | FK relationships with from_col / to_col metadata | Qdrant + OpenSearch |
| `STATUS` | CHECK constraints | Status code lists (FIRST, SECOND, THIRD, FROZEN, etc.) | Qdrant + OpenSearch |
| `WORKFLOW` | Column comments + triggers | Lifecycle state descriptions and business rules | Qdrant + OpenSearch |
| `INDEX` | CREATE INDEX | Non-trivial index definitions and indexed columns | Qdrant + OpenSearch |
| `AUDIT` | Audit table patterns | Audit and history table descriptions | Qdrant + OpenSearch |
| `PARTITION` | Partition definitions | Partition keys, strategy, retention policy | Qdrant + OpenSearch |
| `GLOSSARY` | data/glossary.json | Institution-specific term definitions | Qdrant + OpenSearch |
| `BUSINESS_RULE` | Guardrail definitions | Cross-table business rules and domain mappings | Qdrant + OpenSearch |
| `FEW_SHOT` | data/few_shot_examples.json | NL→SQL example pairs | **Qdrant only** |

Each chunk carries metadata: `chunk_id`, `chunk_type`, `text`, `domain_tags`,
`referenced_tables`, `module`, `workflow`.

**`FEW_SHOT` chunks are indexed into Qdrant only.** Semantic similarity is more
appropriate than keyword matching for finding relevant examples.

#### 4.1.3 FK Graph Structure

- Build a NetworkX `DiGraph` with 69 nodes and ~150 directed FK edges.
- Each edge represents `(child_table) → (parent_table)`.
- Node metadata: primary key column, all column names.
- Edge metadata: `from_col`, `to_col`.
- Serialise to `data/fk_graph.json` (JSON format — pickle is forbidden for security).
- Graph cycle detection capped at 20 cycles maximum (avoid full materialisation).

#### 4.1.4 Embedding and Indexing

- Embed all chunks using BAAI/bge-small-en-v1.5 (384-dimensional, CPU by default).
- Upsert into Qdrant (dense vector store).
- Index into OpenSearch with custom domain tokeniser that preserves `DEK`, `URN`,
  `board_id`, and other domain tokens as single units.
- Ingestion must be idempotent and support incremental updates (changed-table detection).

#### 4.1.5 Schema Drift Detection

- After each successful ingestion, store a DDL hash in `data/.schema_hash`.
- On `python main.py` startup, compare current DDL hash against stored hash.
- If hashes differ, log `schema_drift_detected` and print a loud operator warning.
- The system continues but retrieved chunks may be stale until re-ingestion.

### 4.2 Runtime Query Pipeline

#### 4.2.1 Query Understanding (No LLM Call)

- Expand abbreviations and normalise the user query using keyword rules.
- Classify intent: `workflow_state`, `aggregation`, `lookup`, `join`, `reporting`.
- Extract entity tables using domain keyword mapping.
- Output: normalised query + intent + entity table list.
- **No LLM call at this stage** — pure rule-based keyword and pattern matching.

#### 4.2.2 Hybrid Retrieval

```
User query (normalised)
        │
        ├── Qdrant dense top-20 (cosine similarity, BGE-small-en-v1.5)
        ├── OpenSearch BM25 top-20 (custom domain tokeniser)
        └── NetworkX BFS (bidirectional, max 2 hops from entity tables)
                │
                ▼
        Candidate deduplication (by chunk_id)
                │
                ▼
        RRF Fusion (k=60): score = 1/(60+rank_dense) + 1/(60+rank_bm25)
                │
                ▼
        Optional cross-encoder reranking (disabled by default)
                │
                ▼
        Context Budget Manager (7,000 tokens)
          · Pin mandatory entity chunks first
          · Fill remaining slots by RRF rank
          · Stop at budget
```

**Bidirectional BFS requirement:** The FK graph is directed (child → parent). BFS must
traverse edges in both directions to find all valid join paths. Maximum 2 hops.

**Mandatory entity chunk pinning:** Chunks for tables identified by QueryUnderstanding
must be included regardless of retrieval score. This prevents high-scoring glossary chunks
from displacing critical table definitions.

#### 4.2.3 Prompt Construction

Assemble a 7-section structured prompt in fixed order:

```
[SYSTEM]    Hardcoded instructions: SELECT-only, JSON output contract, safety rules
[SCHEMA]    TABLE + VIEW chunks (highest priority, placed early)
[WORKFLOW]  WORKFLOW + STATUS chunks (lifecycle state descriptions)
[GLOSSARY]  GLOSSARY chunks (domain vocabulary definitions)
[JOINS]     FK graph join path text from BFS traversal
[EXAMPLES]  FEW_SHOT chunks (top-3 by semantic similarity to current query)
[QUERY]     User's normalised question (placed last — recency bias in attention)
```

Section ordering is not optional. It reflects transformer attention recency bias:
system instruction first (always attended), user question last (highest attention weight).

**Token budget verification:** After assembly, count total tokens with tiktoken. Warn if
total exceeds 90% of `LLM_CONTEXT_SIZE` (8,192). Log at WARNING level.

**Chunk deduplication:** Before section distribution, deduplicate by `chunk_id`. A chunk
that appears in multiple retrieval results must appear in the prompt only once.

#### 4.2.4 SQL Generation

- **Model:** Qwen2.5-Coder 3B Instruct Q4_K_M via llama-server.
- **Format:** Output is enforced as JSON containing the SQL.
- **Grammar:** `config/sql_select.gbnf` — currently disabled (commented out). Validation relies on downstream JSON extraction and AST parsing.
  DML tokens are masked before sampling — they cannot be generated regardless of prompt.
- **Temperature:** 0.2 (near-deterministic).
- **Output contract:**
  ```json
  {
    "sql":         "<valid PostgreSQL SELECT>",
    "tables_used": ["table1", "table2"],
    "confidence":  0.0–1.0,
    "explanation": "one sentence"
  }
  ```
- **Three-layer JSON parsing:** direct parse → JSON extraction → regex SQL extraction.
  If regex fallback is used, `confidence` must be set to `0.0`, not `0.3`.

#### 4.2.5 SQL Validation Pipeline

12 sequential steps (`validation/core/sql_validator.py::build_default_pipeline`). First
failing step returns error context for retry:

| # | Step | Check | Method |
|---|---|---|---|
| 1 | Syntax | PostgreSQL grammar | sqlglot parse (dialect="postgres") |
| 2 | Placeholder | No parameter placeholders (`:qp_id`, `$1`) | AST scan — LLM must use literal values, not parameterised queries |
| 3 | Alias | No hallucinated table aliases | AST — catches aliases the LLM invents that don't map to a declared table |
| 4 | Schema grounding | No hallucinated tables or columns | AST walk; CTE aliases excluded; column-level check via `TableInventory` |
| 5 | Join | No Cartesian joins | AST inspection (not regex — `FROM\s+\w+\s*,\s*\w+` false-positives on `generate_series(1, 10)`) |
| 6 | Safety | No DML/DDL statements | AST DML/DDL node check; blocked keyword regex as secondary defence |
| 7 | Security | Tenant filter present or injected | AST injection; CTE-aware; tenant table set derived dynamically from schema map |
| 8 | Group-by alignment | Non-aggregated SELECT columns appear in GROUP BY | AST — rejects PostgreSQL-invalid aggregate/group mismatches |
| 9 | Cost | EXPLAIN cost below threshold | PostgreSQL EXPLAIN; default threshold via `VALIDATION_EXPLAIN_COST_THRESHOLD`; deterministic autofix attempted on PG planner hints before rejecting |
| 10 | Semantic | Lightweight heuristic logic checks (business-rule / phrasing alignment) | Rule-based heuristics over NL + SQL |
| 11 | Hardcoded literal | Suspicious hardcoded integer ID literals | AST — flags IDs with no basis in the NL question |
| 12 | Aggregation | Nested aggregates / missing GROUP BY | AST |

**Critical implementation requirements:**
- Tenant-scoped table set must be derived dynamically from the schema map at startup
  (any table with `board_id` or `course_id` column). Hard-coded table lists are forbidden.
- Tenant filter injection must be CTE-aware — if the scoped table is inside a CTE body,
  inject the predicate there, not on the outer SELECT.
- Cartesian join detection must use AST inspection, not regex. The regex
  `FROM\s+\w+\s*,\s*\w+` false-positives on `generate_series(1, 10)`.
- CTE alias names must be excluded from the schema hallucination check. CTEs appear as
  `exp.Table` nodes in sqlglot AST but are not schema tables.

#### 4.2.6 Retry and Repair Loop

- On validation failure, construct a correction prompt including the original question,
  failed SQL, and specific error message from the failing validation step.
- Maximum retries is tunable via `VALIDATION_MAX_RETRIES` (`config/settings.py` default: 2;
  current `.env` override: **4** — every failed query pays for up to 5 full pipeline passes
  at the tuned setting, not 3. Keep this doc's number in sync with `.env` if it's retuned.
- After retry exhaustion, log failure to `failures/` and return an error to the user.
- Track retry success rate as a production reliability metric.

#### 4.2.7 Execution

- Use `ThreadedConnectionPool` (min 2, max 20, tunable via `PG_POOL_MIN` / `PG_POOL_MAX`).
- Pool initialisation must use double-checked locking (thread-safe).
- Before returning a connection to the pool on error, always call `conn.rollback()`.
- Enforce `default_transaction_read_only=on` at the connection level.
- Enforce `statement_timeout=30000` (30 seconds) per connection — not a pipeline
  validation step; applied at connection setup, same layer as read-only enforcement.
- Dry-run mode (default): validate SQL, display result, do not execute.
- Execute mode: validate then execute; row limit enforced via AST LIMIT injection.

#### 4.2.8 Failure Logging

- Write failed queries atomically to `failures/` using `.tmp` + `os.replace()`.
  Non-atomic writes risk partial JSON files corrupting the Phase 2 training corpus.
- Each failure record: `timestamp`, `nl_query`, `failed_sql`, `error`, `retries`,
  `corrected_sql` (populated via `:correct` CLI command).
- Filename must include `request_id` for uniqueness under concurrent load.

### 4.3 CLI Interface

- Interactive terminal UI using Rich + prompt_toolkit.
- Syntax-highlighted SQL output. Per-stage timing display.
- Query history via `↑ ↓` navigation.
- **Default mode: dry-run.** User must explicitly switch to execute mode.
- Commands: `:dry`, `:exec`, `:debug`, `:correct`, `:clear`, `:help`, `:quit` / `:q`.
- Non-interactive mode: `python main.py --query "..."` with `--dry-run` / `--exec` flags.

### 4.4 Phase 1 Deliverables

| Deliverable | Module |
|---|---|
| DDL parser (5-pass, sqlglot AST) | `ingestion/ddl_parser.py` |
| Semantic chunk generator (10 types) | `ingestion/chunk_generator.py` |
| FK graph builder and BFS traversal | `ingestion/graph_builder.py` |
| Qdrant indexer | `indexing/qdrant_indexer.py` |
| OpenSearch indexer (custom tokeniser) | `indexing/opensearch_indexer.py` |
| Hybrid retrieval orchestrator | `retrieval/orchestrator.py` |
| Optional cross-encoder reranker | `retrieval/reranker.py` |
| Query understanding (rule-based) | `generation/query_understanding.py` |
| Execution orchestrator | `pipeline/runner.py` |
| JSON-constrained SQL generator | `generation/sql_generator.py` |
| 12-step validation pipeline + RetryValidator | `validation/core/sql_validator.py` |
| Semantic logic heuristics | `validation/semantic/semantic_checks.py` |
| CLI interface | `cli/interface.py` |
| Schema drift detection | `utils/schema_versioning.py` |
| Blocklist constants | `validation/utils/blocklist.py` |
| System prompts and templates | `generation/prompt_builder.py` |
| JSON schema definition file | `config/sql_select.json` |
| Ingestion entry point | `ingest.py` |
| Application entry point | `main.py` |
| Phase 1 requirements file | `requirements.txt` |
| Environment configuration | `.env` |

**Note:** `validation/` has grown into a multi-module package rather than the flat files
implied above — the 12 steps in §4.2.5 are split across `validation/ast/` (syntax,
placeholder, alias, join, safety, aggregation), `validation/schema/`, `validation/security/`,
`validation/execution/` (cost), and `validation/semantic/` (semantic, hardcoded-literal),
orchestrated by `validation/core/sql_validator.py`. The table above lists the primary
entry points, not an exhaustive file list.

### 4.5 Phase 1 Non-Functional Requirements

| Requirement | Target |
|---|---|
| End-to-end latency (p50) | < 2 seconds |
| End-to-end latency (p95) | < 5 seconds |
| Concurrent evaluator support | 5,000 peak during burst windows |
| Syntax pass rate | > 95% of generated SQL |
| Hallucination rate | < 5% (table + column level) |
| Execution success rate | > 85% of validated SQL |
| Retry success rate | > 60% of failed queries repaired |
| Schema drift detection | At every startup |
| Failure log integrity | Atomic writes; no partial JSON files |

---

## 5. Phase 2 — QLoRA Fine-Tuning Pipeline

**Status:** Pipeline implemented. Awaiting corpus readiness.

### 5.1 Entry Criteria (Must All Be Met Before Starting)

1. Failure corpus contains at least **200–300 corrected NL→SQL pairs** in `failures/`
   (populated via `:correct` command during Phase 1 production use).
2. Phase 1 retrieval quality metrics confirm retrieval is working — SQL generation
   quality, not retrieval quality, is the remaining bottleneck.
3. A stable held-out evaluation set exists (auto-created by `data_pipeline.py`).

**Warning:** Fine-tuning on fewer than 50 pairs or low-quality pairs can make the model
worse on queries it currently handles correctly. Do not start Phase 2 early.

### 5.2 Training Data Preparation

**Module:** `fine_tuning/data_pipeline.py`

- Load training pairs from three sources in priority order:
  1. `failures/` corrected pairs (highest quality — real user queries)
  2. `data/few_shot_examples.json` curated pairs
  3. `data/phase2_synthetic.jsonl` (optional, use only when corpus < 50 pairs)
- Apply quality filters: minimum NL length, SQL length bounds, SELECT-only enforcement.
- Validate SQL correctness with sqlglot before including any pair.
- Deduplicate by NL query text.
- **Enrich each pair with Phase 1 schema context** by calling the live
  `RetrievalOrchestrator` for each NL question. This is the critical distribution
  match constraint — training prompts must use the same 7-section structure as
  Phase 1 inference prompts.
- Split into train (85%) and held-out eval (15%) sets with stratified sampling.
- Write `data/phase2_train.jsonl` and `data/phase2_eval.jsonl` atomically.

**`--skip-retrieval` flag:** Bypasses schema context enrichment. Causes training/inference
distribution mismatch. Use only when Qdrant and OpenSearch are unavailable. Output
must include a clear warning when this flag is used.

**Synthetic data bootstrapping:** `fine_tuning/generate_synthetic.py` generates typed-literal
SQL pairs (not parameterised placeholders) from the FK graph and schema. Use as a
bootstrap only when real corpus < 50 pairs. Stop using synthetic data once 200+ real
corrected pairs are available.

### 5.3 QLoRA Fine-Tuning

**Module:** `fine_tuning/trainer.py`

**Hardware requirement:** 8 GB VRAM. Stop llama-server before starting.

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen/Qwen2.5-Coder-3B-Instruct (HuggingFace) | ~6 GB; full-precision weights required for training |
| Quantisation (training) | 4-bit NF4 via bitsandbytes | Fits base model in ~2 GB GPU memory |
| LoRA rank | 16 | Adequate capacity for domain adaptation |
| LoRA alpha | 32 | Standard 2× rank scaling |
| LoRA dropout | 0.05 | Regularisation |
| Target modules | q_proj, k_proj, v_proj, o_proj | All attention projections |
| Batch size | 2 per device | VRAM constraint |
| Gradient accumulation | 8 steps | Effective batch = 16 |
| Gradient checkpointing | Enabled | Required for 8 GB VRAM |
| Epochs | 3 | Default |
| Learning rate | 2e-4 | Standard QLoRA |
| LR scheduler | Cosine | Prevents end-of-training overshoot |
| Warmup ratio | 0.03 | Prevents unstable early updates |
| Max sequence length | 1,024 tokens | |

**Requirements:**
- Verify LoRA target modules exist in the model before training begins.
- Support `--resume-from-checkpoint` to recover from interrupted training.
- Save LoRA adapter to `models/adapters/fine_tuning-v{N}/` (~50 MB).
- Base model weights at `models/hf/` must never be modified.

**Prompt format constraint (critical):** Apply Qwen2.5 ChatML template explicitly:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```
Stop token is `<|im_end|>` (token ID 151645). Any file encoding corruption that replaces
this token with a visually similar character will silently destroy training.

### 5.4 Evaluation and Regression Guard

**Module:** `fine_tuning/evaluator.py`

Run against held-out eval set before any export. Block export if any metric regresses.

| Metric | Target | Method |
|---|---|---|
| Syntax pass rate | > 95% | sqlglot parse |
| No-hallucination rate | > 95% | AST walk: table-level + column-level |
| Execution valid rate | > 85% | PostgreSQL EXPLAIN |
| Exact match rate | Improve vs baseline | Case-insensitive string equality |
| p50 generation latency | < 2s | Wall-clock after GPU warmup (2 throwaway runs) |

- Compare all metrics against `data/eval_baseline.json`.
- If any metric regresses by more than 3 percentage points: exit code 1, block export.
- On first run (no baseline): write current metrics as the baseline.
- `exact_match_rate` is case-insensitive string equality — not result-set comparison.
  True semantic evaluation (execute both queries, compare rows) is a Phase 3+ enhancement.

### 5.5 Export Pipeline

**Module:** `fine_tuning/export.py`

Three sequential steps. Each step verifies output before proceeding:

**Step 1 — Merge**
- Load base model in full FP16 precision on CPU (~12 GB RAM required, not GPU).
- Call `model.merge_and_unload()` to fold adapter into base weights.
- Save merged model to `models/merged/fine_tuning-v{N}/`.
- Merged model is deleted after successful quantisation (save ~12 GB disk).
- `--keep-merged` flag retains it for additional quantisation passes.

**Step 2 — Convert to GGUF**
- Run `D:\llama.cpp\convert_hf_to_gguf.py` via subprocess.
- 2-hour timeout. Fail with clear error if timeout exceeded.
- Verify output file size > 1 GB before proceeding. A smaller file indicates failed conversion.
- Output: `models/qwen/qwen2.5-coder-3b-finetuned-v{N}-f16.gguf`.

**Step 3 — Quantise**
- Run `llama-quantize.exe` with `Q4_K_M` quantisation type.
- 1-hour timeout.
- Verify output file size > 1 GB before deleting F16 GGUF.
- Output: `models/qwen/qwen2.5-coder-3b-finetuned-v{N}-q4_k_m.gguf` (~2.4 GB).
- F16 GGUF deleted by default after successful quantisation.
- `--keep-f16` retains it.

**Tool paths:** All tool paths (`LLAMA_CPP_SOURCE`, `LLAMA_PRECOMPILED`, `HF_MODEL_DIR`)
must be overridable via environment variables without code changes.

**Deployment:** Point llama-server at the new GGUF. No Phase 1 code changes required.
The old GGUF is not deleted automatically — retain until fine-tuned model is verified.

### 5.6 Phase 2 Disk Space Requirements

| Stage | Space | Permanent? |
|---|---|---|
| HuggingFace base model | ~6.2 GB | Yes — training base |
| LoRA adapter | ~50 MB | Yes — reuse across cycles |
| Merged model | ~12 GB | No — deleted after quantisation |
| F16 GGUF | ~6 GB | No — deleted after quantisation |
| Final Q4_K_M GGUF | ~2.4 GB | Yes — replaces inference GGUF |
| **Peak during export** | **~26 GB** | |
| **If retaining old GGUF** | **~29 GB** | |

Minimum 30 GB free disk space before starting export.

### 5.7 Phase 2 Deliverables

| Deliverable | Module |
|---|---|
| Training data preparation pipeline | `fine_tuning/data_pipeline.py` |
| Synthetic pair bootstrapper | `fine_tuning/generate_synthetic.py` |
| QLoRA fine-tuning trainer | `fine_tuning/trainer.py` |
| Regression-guarded evaluator | `fine_tuning/evaluator.py` |
| 3-step export pipeline | `fine_tuning/export.py` |
| Phase 2 requirements file (pinned versions) | `requirements_phase2.txt` |

---

## 6. Cross-Cutting Requirements

### 6.1 Code Quality

- Production-grade, modular Python 3.11+.
- Type annotations throughout. `from __future__ import annotations` for 3.9 compatibility.
- Pydantic-settings for all configuration — type-validated at startup, all via env vars.
- structlog for structured JSONL logging — `request_id` bound end-to-end per request.
- Atomic file writes for all persistent outputs (`.tmp` + `os.replace()`).
- Surgical code changes following `FIX #N` documentation discipline.
- Original inline comments preserved across patches.

### 6.2 Execution Defences

- Prompt injection resistance — AST validation as defence-in-depth.
- Cost thresholding — `EXPLAIN` query cost ceiling (`VALIDATION_EXPLAIN_COST_THRESHOLD`).
- Read-only PostgreSQL role at the database level — application-level controls alone
  are insufficient. PostgreSQL functions can have side effects even inside SELECT.
- Tenant filter injection via AST manipulation, never string concatenation.
- All model artifacts in safetensors format (never pickle).
- FK graph serialised as JSON (never pickle).
- OpenSearch password excluded from settings serialisation (`exclude=True`).
- Deployment paths containing sensitive information must not be hardcoded — use
  environment variable overrides.

### 6.3 Observability

- All pipeline events logged to `logs/nl_sql.jsonl` as JSONL with `request_id`.
- Key metrics tracked per request: retrieval hit counts, token usage, generation
  confidence, validation step results, retry count, execution row count, total latency.
- Schema drift logged as `schema_drift_detected` event.
- Tenant filter injection/absence logged with specific event types.

### 6.4 Configuration

- All settings read from `.env` via Pydantic-settings.
- No hardcoded values except established defaults with documented override mechanism.
- Separate `requirements.txt` (Phase 1) and `requirements_phase2.txt` (Phase 2).
- Phase 2 dependency versions must be pinned with upper bounds — the transformers/peft/trl
  stack has unstable APIs across minor versions.

---

## 7. Future Roadmap (Phase 3+)

The following items are confirmed out of scope for Phase 1 and Phase 2. They are
candidates for a future Phase 3:

| Item | Effort | Impact |
|---|---|---|
| Join Templates (`JOIN_TEMPLATE` chunk type) | Low | High |
| FastAPI REST / web UI layer | Medium | High |
| Table Statistics Chunks (`TABLE_STATS`) | Low | Medium |
| Domain-finetuned BGE embeddings | High | Medium |
| Conversational memory and multi-turn refinement | Medium | Medium |
| Semantic result-set evaluation (execute + compare) | Medium | Medium |
| Adaptive retrieval strategy selection | Medium | Medium |
| Intent decomposition for complex queries | High | Medium |

---

## 8. Schema Scale Reference

| Data Domain | Value |
|---|---|
| DDL Objects | 69 |
| Foreign Key Edges | ~150 |
| Schema Chunk Limit | 9000 tokens |
| Triggers | 15 |
| Design decisions (DDL-documented) | 22 |
| Semantic chunks (post-ingestion) | ~212 |
| Peak concurrent evaluators | 5,000 |
| High-volume core tables (5-year steady state) | ~20 million rows |
| Supporting transactional tables (5-year steady state) | ~5 million rows |

---
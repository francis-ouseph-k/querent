---

# Requirements Document
## Digital Evaluation System — NL→SQL
**Version:** 2.1
**Date:** 2026-07-06
**Status:** Phase 1 Complete · Phase 2 In Progress
**Revision note:** Reconciled against live code — schema v10.5, Steiner-tree
retrieval, bf16 Blackwell-safe LoRA, 4096 max-seq, 9000-token budget, 32768
context, no train/eval split (fixed 191-question benchmark), GBNF disabled.

---

## 1. Overview

The system converts natural-language business questions into validated, read-only
PostgreSQL SQL queries. It is purpose-built for a university digital evaluation pipeline
with the v10.5 schema (61 table definitions including 7 partition children, plus 2 views)
containing indexes, foreign keys, partitioned tables, JSONB fields, workflow-driven
entities, and audit structures.

**Target users:** Faculty administrators querying examination and evaluation data without
SQL expertise.

**Example:** *"Show all scripts pending third evaluation in board 5 where deviation
exceeds threshold"* → validated PostgreSQL SELECT executed against read-only replica in
under 2 seconds.

**Deployment target:** Local 8 GB GPU environment (RTX 5060 Ti, Blackwell / sm_120).
Fully local inference and retrieval. No external LLM APIs.

---

## 2. Scope and Constraints

### 2.1 In Scope

| Item | Phase |
|---|---|
| Semantic DDL parsing and chunking | 1 |
| FK graph construction and Steiner-tree traversal | 1 |
| Dense + BM25 + graph hybrid retrieval | 1 |
| RRF fusion and context budget management | 1 |
| 7-section prompt assembly | 1 |
| JSON-constrained SQL generation | 1 |
| 12-step SQL validation pipeline | 1 |
| Retry and repair loop | 1 |
| CLI interface with dry-run default | 1 |
| Failure corpus logging (`:correct` mechanism) | 1 |
| Schema drift detection | 1 |
| Curated-corpus de-leak + enrichment pipeline | 2 |
| Synthetic pair bootstrapping | 2 |
| LoRA fine-tuning (bf16, Blackwell-safe) | 2 |
| Benchmark A/B regression comparison (RESULT-1 vs RESULT-2) | 2 |
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
- Result-set semantic evaluation (execute both queries, compare rows)

### 2.3 Hardware Constraints

- **8 GB VRAM ceiling** — drives model selection (3B), quantisation (Q4_K_M),
  context window management, and the bf16 LoRA training configuration.
- **Single GPU** — llama-server and Phase 2 training cannot share the GPU.
  llama-server must be stopped before Phase 2 training begins.
- **Blackwell / sm_120** — the training path must be bf16 standard LoRA. bitsandbytes
  NF4 quantisation produces all-NaN logits on sm_120 and is disabled (legacy only).
  PyTorch must be a CUDA 12.8 (cu128) build (torch ≥ 2.7) — cu121 wheels have no sm_120
  kernels.
- **Local execution only** — no external LLM APIs at any phase.

### 2.4 Database Constraints

- **Read-only replica only** — the system must never connect to the primary database.
- **Never generate or execute DML/DDL** — enforced at generation (JSON contract), validation
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
| LLM runtime | llama-server (llama.cpp) | pre-compiled | Local inference, JSON output contract |
| Embedding model | BAAI/bge-small-en-v1.5 | 384-dim | Dense embeddings for Qdrant |
| Vector store | Qdrant | 1.7+ | Dense semantic retrieval |
| Keyword store | OpenSearch | 2.11+ | BM25 sparse retrieval |
| Graph engine | NetworkX | 3.2+ | FK graph and Steiner-tree traversal |
| SQL parser | sqlglot | 20.0+ | AST-based validation, manipulation |
| SQL formatter | sqlparse | 0.4.4+ | Lightweight formatting and syntax checks |
| DB driver | psycopg2 | 2.9.9+ | PostgreSQL ThreadedConnectionPool |
| Settings | pydantic-settings | 2.1+ | Type-validated env var configuration |
| Logging | structlog | 24.1+ | JSONL structured logging with request_id |
| CLI | Rich + prompt_toolkit | — | Syntax-highlighted terminal interface |
| Fine-tuning | transformers + peft + trl | pinned (see requirements_phase2.txt) | bf16 LoRA training (Phase 2 only) |
| Training precision | bf16 standard LoRA (no bitsandbytes) | — | Blackwell sm_120 active path; NF4/bitsandbytes legacy only |
| GGUF conversion | convert_hf_to_gguf.py | llama.cpp source | HF → GGUF (Phase 2 export) |
| GGUF quantisation | llama-quantize.exe | pre-compiled | Q4_K_M compression (Phase 2 export) |
| API layer | FastAPI | — | **Deferred — Phase 3+** |

---

## 4. Phase 1 — Production NL→SQL System

**Status:** Complete.

### 4.1 Schema Ingestion Pipeline

#### 4.1.1 DDL Parsing

- Parse `digital_evaluation_schema_v10_5.sql` using sqlglot (PostgreSQL 16 dialect) in
  passes: tables → foreign keys → indexes → column comments → views and triggers.
- Filter noise (`django_migrations`, `auth_user`).
- Extract primary keys and foreign key constraints explicitly.
- Preserve structural and relational context: 61 table definitions (incl. 7 partition
  children) + 2 views, ~149 FK constraints, 96 indexes, 22 triggers, composite keys,
  and JSONB schema shapes.

> **Schema version (resolved):** `.env` sets `DDL_PATH=data/docs/digital_evaluation_schema_v10_5.sql`,
> which is authoritative and ingested at runtime. `config/settings.py` still declares an
> older default (`ddl_path = "…v10_4_1.sql"`); the `.env` value overrides it, but the
> stale default should be updated to v10.5 for clarity. v10.4.1 now lives under
> `data/docs/archive/`. The object counts above are for v10.5 — re-verify the post-ingestion
> chunk count with `python ingest.py --dry-run` whenever the DDL changes.

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

- Build a NetworkX `DiGraph` with ~63 nodes (61 tables + 2 views) and ~149 directed FK edges.
- Each edge represents `(child_table) → (parent_table)`.
- Node metadata: primary key column, all column names.
- Edge metadata: `from_col`, `to_col`.
- Serialise to `data/fk_graph.json` (JSON format — pickle is forbidden for security).
- Trivial self-referential FK loops (`parent_id → id`) are filtered at ingestion time;
  join-path search runs on the undirected projection (see §4.2.2).

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
        └── NetworkX Steiner Tree over the undirected FK graph
                (minimal connecting subtree across the entity tables)
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
        Context Budget Manager (9,000 tokens)
          · Pin mandatory entity chunks first
          · Fill remaining slots by RRF rank
          · Single global running total — never exceeds the budget
          · Stop at budget
```

**Steiner-tree requirement:** The FK graph is directed (child → parent). Join-path search
runs a global Steiner-tree over the **undirected** projection of the graph, producing the
minimal subtree that connects the entity tables. This naturally navigates cyclic FK
structures without visited-set tracking and without a fixed hop limit.

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
[JOINS]     FK graph join path text from Steiner-tree traversal
[EXAMPLES]  FEW_SHOT chunks (top-3 by semantic similarity to current query)
[QUERY]     User's normalised question (placed last — recency bias in attention)
```

Section ordering is not optional. It reflects transformer attention recency bias:
system instruction first (always attended), user question last (highest attention weight).

**Token budget verification:** After assembly, count total tokens with the aligned
tokenizer. Warn if total approaches `LLM_CONTEXT_SIZE` (32,768). Log at WARNING level.

**Chunk deduplication:** Before section distribution, deduplicate by `chunk_id`. A chunk
that appears in multiple retrieval results must appear in the prompt only once.

#### 4.2.4 SQL Generation

- **Model:** Qwen2.5-Coder 3B Instruct Q4_K_M via llama-server.
- **Format:** Output is enforced as JSON containing the SQL.
- **Grammar:** `config/sql_select.gbnf` — **currently disabled (commented out)** at this
  stage. DML prevention therefore relies on the downstream JSON contract plus the
  validation pipeline (AST safety check + blocked-keyword regex), not on grammar-level
  token masking. Re-enabling GBNF is optional and additive.
- **Temperature:** 0.2 (near-deterministic) for interactive use; 0 for benchmark A/B runs.
- **Output contract:**
  ```json
  {
    "sql":         "<valid PostgreSQL SELECT>",
    "tables_used": ["table1", "table2"],
    "confidence":  0.0–1.0,
    "explanation": "one sentence"
  }
  ```
- **Multi-layer JSON parsing:** direct parse → JSON extraction → regex SQL extraction.
  If a regex fallback is used, `confidence` must be set to `0.0`, not `0.3`.

#### 4.2.5 SQL Validation Pipeline

12 sequential steps (`validation/core/sql_validator.py::build_default_pipeline`). First
failing step returns error context for retry:

| # | Step | Check | Method |
|---|---|---|---|
| 1 | Syntax | PostgreSQL grammar | sqlglot parse (dialect="postgres") |
| 2 | Placeholder | No parameter placeholders (`:qp_id`, `$1`) | AST scan — LLM must use literal values, not parameterised queries |
| 3 | Alias | No undeclared table aliases | AST — catches aliases used before/without declaration in FROM/JOIN |
| 4 | Schema grounding | No hallucinated tables or columns | Per-SELECT-scoped AST walk; CTE aliases excluded; CHECK-enum membership; column check via `TableInventory` |
| 5 | Join | No Cartesian joins | AST inspection (not regex — `FROM\s+\w+\s*,\s*\w+` false-positives on `generate_series(1, 10)`) |
| 6 | Safety | No DML/DDL statements | AST DML/DDL node check; blocked keyword regex as secondary defence |
| 7 | Security | Tenant filter present or injected | AST injection; CTE-aware; tenant table set derived dynamically from schema map |
| 8 | Group-by alignment | Non-aggregated SELECT columns appear in GROUP BY | AST — rejects PostgreSQL-invalid aggregate/group mismatches |
| 9 | Cost | EXPLAIN cost below threshold | PostgreSQL EXPLAIN; default threshold via `VALIDATION_EXPLAIN_COST_THRESHOLD`; skipped when no DB connection |
| 10 | Semantic | Lightweight heuristic logic checks (business-rule / phrasing alignment) | Rule-based heuristics over NL + SQL |
| 11 | Hardcoded literal | Literal filter inside a LEFT JOIN ON clause (NL-aware) | AST — only fails when the literal appears in the NL question |
| 12 | Aggregation | Nested aggregates / invalid aggregate structure | AST |

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
  the failed SQL, the exact failing-step message, and the offending table's authoritative
  live column list ("use ONLY these columns").
- Maximum retries is tunable via `VALIDATION_MAX_RETRIES` (`config/settings.py` default: 2;
  the current `.env` override may differ — keep this doc in sync with `.env` if it is retuned).
- Stall early-abort: if a correction pass reproduces the identical `(step, message)` error,
  the loop stops immediately rather than burning the remaining attempts.
- On failure, the runner logs the SQL that *actually* failed (the last corrected attempt),
  not the first-pass generation.
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
| DDL parser (multi-pass, sqlglot AST) | `ingestion/ddl_parser.py` |
| Semantic chunk generator | `ingestion/chunk_generator.py` |
| FK graph builder and Steiner-tree traversal | `ingestion/graph_builder.py` |
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
| GBNF grammar (currently disabled) | `config/sql_select.gbnf` |
| Ingestion entry point | `ingest.py` |
| Application entry point | `main.py` |
| Phase 1 requirements file | `requirements.txt` |
| Environment configuration | `.env` |

**Note:** `validation/` is a multi-module package — the 12 steps in §4.2.5 are split across
`validation/ast/` (syntax, placeholder, alias, join, safety, aggregation),
`validation/schema/`, `validation/security/`, `validation/execution/` (cost), and
`validation/semantic/` (semantic, hardcoded-literal), orchestrated by
`validation/core/sql_validator.py`. The table lists primary entry points, not an
exhaustive file list.

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

## 5. Phase 2 — LoRA Fine-Tuning Pipeline (bf16, Blackwell-safe)

**Status:** Pipeline implemented. Awaiting corpus readiness.

The measurement design is a controlled A/B: benchmark the base model (RESULT-1), fine-tune,
benchmark the fine-tuned model (RESULT-2), and compare. RESULT-1 and RESULT-2 use the same
fixed 191-question hold-out benchmark and the same pipeline — only the model weights change.

### 5.1 Entry Criteria (Must All Be Met Before Starting)

1. Failure/curated corpus contains enough corrected NL→SQL pairs to fine-tune (guideline:
   200–300; populated via `:correct` during Phase 1 use and manual curation).
2. Phase 1 retrieval quality metrics confirm retrieval is working — SQL generation
   quality, not retrieval quality, is the remaining bottleneck.
3. A stable, fixed held-out benchmark exists — `data/inputs/benchmark-test-set.jsonl`
   (191 questions), **separate** from the training corpus. It is NOT an auto-split; the
   training pipeline performs no train/eval split. De-leak the corpus against this
   benchmark first (`python -m fine_tuning.deleak_train`) so no benchmark question leaks
   into training.

**Warning:** Fine-tuning on fewer than 50 pairs or low-quality pairs can make the model
worse on queries it currently handles correctly. Do not start Phase 2 early.

### 5.2 Training Data Preparation

**Active path (no split):** `fine_tuning/deleak_train.py` → `fine_tuning/build_train_from_curated.py`

- `deleak_train.py` removes benchmark-overlapping rows (exact + token-Jaccard near-dup)
  from the curated corpus, writing a leak-free `*.clean.jsonl` plus an auditable
  `deleak_report.json`.
- `build_train_from_curated.py` enriches each clean row with Phase-1 schema context by
  calling the live `RetrievalOrchestrator` (Qdrant + OpenSearch must be running), reusing
  `data_pipeline._format_pair`. It writes `data/fine_tuning_train.jsonl` (= `FT_TRAIN_DATA`).
  **No train/eval split, and it does not read `failures/`.** The `--skip-retrieval` flag
  produces empty schema context (train/inference mismatch) and must not be used for a real run.
- This is the **critical distribution-match constraint** — training prompts use the same
  structure and live-retriever schema context as Phase 1 inference. The trainer internally
  holds out a 10% dev slice for `eval_loss` / early stopping only; that dev slice is not the
  benchmark and never touches it.

**Legacy/alternate path:** `fine_tuning/data_pipeline.py` reads `failures/`, filters, and
performs an 85/15 train/eval split into `data/fine_tuning_train.jsonl` /
`data/fine_tuning_eval.jsonl`. It is superseded by `build_train_from_curated.py` for the
current A/B workflow and is retained for the failure-corpus-driven flow.

**Synthetic data bootstrapping:** `fine_tuning/generate_synthetic.py` generates typed-literal
SQL pairs (not parameterised placeholders) from the FK graph and schema. Use as a
bootstrap only when the real corpus is small. Stop using synthetic data once enough real
corrected pairs are available.

### 5.3 LoRA Fine-Tuning (bf16, Blackwell-safe)

**Module:** `fine_tuning/trainer.py` · run as `python -m fine_tuning.trainer`

**Hardware requirement:** 8 GB VRAM. Stop llama-server before starting.

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen/Qwen2.5-Coder-3B-Instruct (HuggingFace) | ~6 GB; full-precision weights required for training |
| Training precision | **bf16 standard LoRA (no bitsandbytes)** | Blackwell sm_120 safe; NF4 produces NaN logits on sm_120 |
| Model load | `device_map={"": "cuda:0"}`, `attn_implementation="eager"` | Validated Blackwell-safe load path |
| SFTConfig precision | `bf16=True, fp16=False` | Blackwell Tensor Cores favour bf16 |
| LoRA rank | 16 | Adequate capacity for domain adaptation |
| LoRA alpha | 32 | Standard 2× rank scaling |
| LoRA dropout | 0.05 | Regularisation |
| Target modules | q_proj, k_proj, v_proj, o_proj | All attention projections |
| Batch size | 2 per device | VRAM constraint |
| Gradient accumulation | 8 steps | Effective batch = 16 |
| Gradient checkpointing | Enabled | Required for 8 GB VRAM |
| Epochs | 3 | Default |
| Learning rate | 2e-4 | Standard LoRA |
| LR scheduler | Cosine | Prevents end-of-training overshoot |
| Warmup ratio | 0.03 | Prevents unstable early updates |
| Max sequence length | **4,096 tokens** | Fits the full schema-context prompt; gradient checkpointing offsets the VRAM cost |

**Legacy (Ampere sm_80 / Turing sm_75, commented out):** 4-bit NF4 QLoRA via bitsandbytes,
`device_map="auto"`, `fp16=True, bf16=False`. Restore only on non-Blackwell GPUs — see the
`[LEGACY]` / `[BLACKWELL-SAFE]` blocks in `trainer.py`.

**Requirements:**
- Verify LoRA target modules exist in the model before training begins.
- Support `--resume-from-checkpoint` to recover from interrupted training.
- Save LoRA adapter to `FT_ADAPTER_DIR/fine_tuning-v{N}/` (~100 MB, bf16 path).
- Base model weights (`FT_HF_MODEL_DIR`) must never be modified.
- The VRAM pre-check reports **free** memory (`torch.cuda.mem_get_info`) so a running
  llama-server cannot hide behind total capacity.

**Prompt format constraint (critical):** Apply the Qwen2.5 ChatML template explicitly:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```
Stop token is `<|im_end|>` (token ID 151645). Any file-encoding corruption that replaces
this token with a visually similar character will silently destroy training.

### 5.4 Evaluation and Regression Guard

**Primary (active) method — benchmark A/B.** Run the fixed 191-question benchmark through
`batch_run.py --dry-run` twice: once on the base GGUF (RESULT-1) and once on the fine-tuned
GGUF (RESULT-2), everything else identical. Then diff with `compare_results.py`:

- Reports overall + per-difficulty validation-pass delta, regressions (pass→fail), gains
  (fail→pass), and error-taxonomy shift.
- Exit code 1 on net regression (pass-rate drop, or a benchmark question missing from
  RESULT-2). CI-friendly gate.
- Scope limitation: `--dry-run` compares **validation-pass** outcomes only, not executed
  result sets.

**Optional (legacy) method — `fine_tuning/evaluator.py`.** Scores an eval split against a
stored baseline (`FT_BASELINE_PATH`). Not part of the batch_run A/B and requires the
`data_pipeline.py` split to exist.

| Metric | Target | Method |
|---|---|---|
| Syntax pass rate | > 95% | sqlglot parse |
| No-hallucination rate | > 95% | AST walk: table-level + column-level |
| Execution valid rate | > 85% | PostgreSQL EXPLAIN |
| Exact match rate | Improve vs baseline | Case-insensitive string equality |
| p50 generation latency | < 2s | Wall-clock after GPU warmup |

- `exact_match_rate` is case-insensitive string equality — **not** result-set comparison.
  True semantic evaluation (execute both queries, compare rows) is a Phase 3+ enhancement
  and is not implemented in `evaluator.py`.
- Where a baseline exists, block export if a metric regresses beyond tolerance.

### 5.5 Export Pipeline

**Module:** `fine_tuning/export.py` · run as `python -m fine_tuning.export --version v{N}`

Three sequential steps. Each step verifies output before proceeding:

**Step 1 — Merge**
- Load base model in bf16 and fold the adapter via `model.merge_and_unload()`.
- Save merged model to `FT_MERGED_DIR/fine_tuning-v{N}/`.
- Merged model is deleted after successful quantisation unless `--keep-merged`.

**Step 2 — Convert to GGUF**
- Run `convert_hf_to_gguf.py` (path via `LLAMA_CPP_SOURCE`) as a subprocess.
- Verify output file size before proceeding. A tiny file indicates failed conversion.
- Output: `FT_GGUF_OUTPUT_DIR/qwen2.5-coder-3b-finetuned-v{N}-f16.gguf`.

**Step 3 — Quantise**
- Run `llama-quantize.exe` (path via `LLAMA_PRECOMPILED`) with `Q4_K_M`.
- Verify output size before deleting the F16 GGUF (unless `--keep-f16`).
- Output: `FT_GGUF_OUTPUT_DIR/qwen2.5-coder-3b-finetuned-v{N}-q4_k_m.gguf` (~2.4 GB).

**Tool/paths:** All FT paths resolve from `.env` via `config/settings.py` (`FT_`-prefixed);
tool paths (`LLAMA_CPP_SOURCE`, `LLAMA_PRECOMPILED`, `HF_MODEL_DIR`) are env-overridable.
Keep the GGUF output dir and `LLM_MODEL_PATH` on the same layout so llama-server finds the
new model.

**Deployment:** Point llama-server at the new GGUF. No Phase 1 code changes required.
The old GGUF is not deleted automatically — retain until the fine-tuned model is verified.

### 5.6 Phase 2 Disk Space Requirements

| Stage | Space | Permanent? |
|---|---|---|
| HuggingFace base model | ~6.2 GB | Yes — training base |
| LoRA adapter | ~100 MB (bf16) | Yes — reuse across cycles |
| Merged model | ~12 GB | No — deleted after quantisation |
| F16 GGUF | ~6 GB | No — deleted after quantisation |
| Final Q4_K_M GGUF | ~2.4 GB | Yes — replaces inference GGUF |
| **Peak during export** | **~26 GB** | |
| **If retaining old GGUF** | **~29 GB** | |

Minimum 30 GB free disk space before starting export.

### 5.7 Phase 2 Deliverables

| Deliverable | Module |
|---|---|
| Corpus de-leak utility | `fine_tuning/deleak_train.py` |
| Curated-corpus enrichment (no split) | `fine_tuning/build_train_from_curated.py` |
| Training data preparation (legacy, failures-based split) | `fine_tuning/data_pipeline.py` |
| Synthetic pair bootstrapper | `fine_tuning/generate_synthetic.py` |
| LoRA fine-tuning trainer (bf16) | `fine_tuning/trainer.py` |
| Regression-guarded evaluator (optional) | `fine_tuning/evaluator.py` |
| 3-step export pipeline | `fine_tuning/export.py` |
| Benchmark A/B comparator | `compare_results.py` (project root) |
| Phase 2 requirements file (pinned; torch cu128 for sm_120) | `requirements_phase2.txt` |

---

## 6. Cross-Cutting Requirements

### 6.1 Code Quality

- Production-grade, modular Python 3.11+.
- Type annotations throughout. `from __future__ import annotations`.
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
- No hardcoded values except established defaults with a documented override mechanism.
- Separate `requirements.txt` (Phase 1) and `requirements_phase2.txt` (Phase 2).
- Phase 2 dependency versions must be pinned with upper bounds — the transformers/peft/trl
  stack has unstable APIs across minor versions. PyTorch must be a cu128 build (≥ 2.7) on
  Blackwell.

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

## 8. Schema Scale Reference (v10.5)

| Data Domain | Value |
|---|---|
| Table definitions (incl. 7 partition children) | 61 |
| Views | 2 |
| Foreign Key constraints | ~149 |
| Indexes | 96 |
| Triggers | 22 |
| Design decisions (DDL-documented, D-1…D-24) | 24 |
| Retrieval context budget | 9,000 tokens |
| LLM context window | 32,768 tokens |
| Semantic chunks (post-ingestion) | re-verify via `python ingest.py --dry-run` |
| Peak concurrent evaluators | 5,000 |
| High-volume core tables (5-year steady state) | ~20 million rows |
| Supporting transactional tables (5-year steady state) | ~5 million rows |

---
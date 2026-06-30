# Validation Pipeline Architecture

The `validation/` directory contains a modular, domain-driven pipeline designed to validate, sanitize, and automatically repair LLM-generated SQL queries before execution. 

This architecture prevents SQL injections, ensures schema compliance, verifies semantic intent, and handles tenant-level isolation.

## Core Concepts

### 1. Pluggable Pipeline (`core/sql_validator.py`)
Validation is orchestrated through a 10-step sequential pipeline. Every candidate query is parsed **once** into an Abstract Syntax Tree (AST) using `sqlglot`. This AST, along with mutable validation state, is passed through each validation step via a shared `ValidationContext`.

### 2. Severity Mechanism (`core/base.py`)
Validation steps yield a `ValidationResult` which includes a `severity` flag:
- **`BLOCK`**: Indicates an unambiguous bug (e.g., hallucinated columns, SQL syntax errors, cartesian joins). The pipeline short-circuits immediately, and the query is sent to the retry loop.
- **`ADVISORY`**: Indicates a heuristic disagreement (e.g., potential over-filtering). The pipeline continues executing, but the advisory is logged and applies a confidence penalty downstream. This prevents mutually-exclusive heuristics from forcing infinite retry loops.

### 3. Self-Correction Loop (`utils/autofix.py` / `RetryValidator`)
If the pipeline yields a `BLOCK` severity, the `RetryValidator` wraps the error message, the schema context, and the failed SQL into a correction prompt. The LLM is queried again to fix its own mistake. This repeats up to `MAX_RETRIES`.

---

## Directory Structure & Component Interactions

### `core/` — Orchestration and State
- **`sql_validator.py`**: The entry point. Constructs the pipeline array (Syntax -> Alias -> Schema -> Joins -> Safety -> Security -> Aggregation -> Semantic, etc.) and iterates through it.
- **`base.py`**: Defines `BaseValidationStep` interface and `ValidationResult`. All validators inherit from this.
- **`context.py`**: Defines `ValidationContext`, a mutable dataclass carrying the AST, schema inventory, alias maps, and accumulated advisories.

### `ast/` — Structural Integrity
Analyzes the structure of the SQL query directly via the `sqlglot` AST.
- **`syntax.py`**: Checks for fundamental parsing failures and parameter placeholder formatting.
- **`joins.py`**: Verifies that `JOIN` conditions are present, avoiding catastrophic Cartesian products. Also validates foreign key paths.
- **`aggregation.py`**: Ensures `GROUP BY` clauses correctly align with non-aggregated `SELECT` projections.
- **`safety.py`**: Strictly rejects any DML/DDL (e.g., `INSERT`, `DROP`, `UPDATE`) and blocks system function calls.

### `schema/` — Database Ground Truth
Ensures the query matches reality.
- **`schema_validator.py`**: Orchestrator for schema checks.
- **`tables.py`**: Verifies all referenced tables actually exist in the database.
- **`columns.py`**: Scans for hallucinated columns against the known schema inventory.
- **`types.py`**: Validates that literal string/integer comparisons match the target column's data type.

### `semantic/` — Intent and Logical Correctness
Ensures the query answers the specific natural language question asked.
- **`semantic_checks.py`**: AST-based heuristics checking for over-filtering, inert ON-clause filters, or ambiguous column references. Many of these yield `ADVISORY` severities.
- **`logical_audit.py`**: NLP-to-SQL alignment. Checks that the nouns/terms used in the NL prompt actually translated into the correct tables and columns in the SQL.
- **`nl_requirements.py`**: Defines hard constraints for specific natural language keywords.

### `security/` — Row Level Security (RLS)
- **`validation.py` & `tenant_injector.py`**: Identifies tables requiring tenant isolation (e.g., multi-tenant tables). Idempotently injects `WHERE tenant_id = X` clauses into the AST, ensuring data silos are respected even if the LLM forgot to add them.

### `execution/` — Database-Dependent Checks
- **`cost.py`**: Runs an `EXPLAIN` on the generated SQL against a read-only replica. Rejects queries that exceed the `VALIDATION_EXPLAIN_COST_THRESHOLD` to prevent runaway analytical queries from locking up the database.

### `utils/` — Shared Helpers
- **`autofix.py`**: Houses the retry and remediation loop logic.
- **`blocklist.py`**: Maintains matrices of known hallucinated "phantom" columns and provides targeted correction hints to the LLM (e.g., "Use student_erp_id instead of id").

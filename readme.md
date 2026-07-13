```markdown
# Querent

Natural Language → SQL platform with hybrid RAG, schema reasoning, 
and constrained generation for complex relational databases.

## What it does

Takes natural language questions about a relational database and generates 
validated, executable SQL. Built for complex enterprise schemas (62+ tables, 
150+ foreign keys).

## Architecture

```
Natural Language Query
        │
        ▼
Query Understanding (Intent + Entity + Table Mapping)
        │
        ▼
Hybrid Retrieval Layer (Vector + BM25 + FK Graph → RRF → Cross-Encoder Rerank)
        │
        ▼
Context Assembly (Schema + Joins + Glossary + Examples)
        │
        ▼
Local LLM Inference (SQL Generation)
        │
        ▼
Validation Pipeline (Syntax → Schema → Safety → Semantic)
        │
        ├── Pass → Execution → Results
        │
        └── Fail → Repair Loop → Re-validation
```

## Design Principles

- **Schema-first reasoning** — retrieval grounded in actual relational structure
- **Deterministic validation** — no execution without AST-level safety checks
- **Local-only inference** — no external LLM or API dependencies in this implementation
- **Graph-aware retrieval** — join paths derived from FK relationships
- **Failure-driven improvement** — production errors structured for training reuse

## Why This System is Different

| Capability | Description |
|---|---|
| Graph-aware retrieval | Uses FK graph traversal to identify valid join paths |
| Multi-source context fusion | Combines vector search, keyword search, and schema graph signals via RRF, then cross-encoder reranking |
| AST-based validation | Enforces SQL correctness beyond regex or heuristic checks |
| Controlled execution gate | Prevents unsafe or invalid SQL from reaching the database |
| Repair loop | Iteratively corrects recoverable SQL failures |
| Local specialisation | Optional LoRA fine-tuning adapts a local model to the target schema without modifying base model weights |

## Technology Stack

- Python 3.11
- PostgreSQL-compatible databases
- Local LLM inference (llama.cpp / GGUF)
- Vector search (Qdrant)
- Keyword search (OpenSearch)
- Cross-encoder reranking (sentence-transformers)
- SQL AST parsing (sqlglot)
- Parameter-efficient fine-tuning (PEFT / LoRA, TRL)
- Structured logging (structlog)

## Repository Structure

```
pipeline/      Orchestration and execution flow
retrieval/     Hybrid retrieval (vector + keyword + graph), RRF fusion, reranking
generation/    Prompting, query understanding, LLM inference
validation/    AST-based SQL validation and repair
               ├── ast/        Syntax-level checks
               ├── schema/     Table / column / type validation
               ├── semantic/   Logical and semantic audits
               └── security/   Safety and execution-gate checks
indexing/      Schema indexing pipelines
ingestion/     Schema and metadata ingestion
fine_tuning/   Optional local LoRA fine-tuning (data prep, trainer, export)
config/        Runtime configuration
tests/         System validation tests
```

## Setup

```bash
git clone https://github.com/francis-ouseph-k/querent.git
cd querent
pip install -r requirements.txt
pip install -r requirements_fine_tuning.txt

# Configure your database connection in config/
python -m pipeline.main
```

## Example

**Input:** "What is the total marks scored by students in Physics in the 2024 odd semester?"

**Output:**
```sql
SELECT SUM(marks_obtained) 
FROM student_marks sm 
JOIN subjects s ON sm.subject_id = s.id 
WHERE s.name = 'Physics' 
  AND sm.academic_year = 2024 
  AND sm.semester_type = 'odd';
```

**Validation:** PASSED (12/12 checks)

## Evaluation

Tested against 62-table PostgreSQL schema with 150+ foreign keys. 
80% SQL validation pass rate on complex multi-join queries.

## Status

Complete end-to-end NL → SQL pipeline with retrieval, generation, validation, 
and execution layers. 

Current focus: improving semantic accuracy for complex multi-join queries and 
expanding the failure-driven training dataset.

Developed over 30+ commits across schema ingestion, retrieval pipelines, 
reasoning engines, and validation layers. Architecture iterated through 
multiple RAG and fine-tuning experiments before converging on the current 
hybrid approach.

## Background

Built as an independent architecture exercise to validate hybrid RAG and 
constrained generation approaches for enterprise NL→SQL, following production 
experience architecting similar systems in higher education.

## Contact

- LinkedIn: [linkedin.com/in/francis-ouseph-k](https://linkedin.com/in/francis-ouseph-k)
- Email: francis.ouseph.k [at] gmail [dot] com

## License

MIT License. See `LICENSE` for details.
```
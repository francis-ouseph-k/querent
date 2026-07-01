# Natural Language → SQL Query Engine

Turn plain-English business questions into validated, production-ready SQL for large enterprise relational databases using hybrid retrieval, local LLM inference, and multi-layer validation—without requiring cloud-based LLM services.

---

## Overview

Enterprise relational databases often contain hundreds of interconnected tables, complex relationships, and domain-specific terminology, making accurate SQL generation significantly more challenging than conventional text-to-SQL benchmarks.

This project implements an enterprise-oriented Natural Language → SQL pipeline that combines Retrieval-Augmented Generation (RAG), deterministic validation, and iterative self-correction to generate SQL that is not only syntactically correct but also semantically aligned with the user's intent.

The system is designed to operate entirely on local infrastructure, making it suitable for environments where privacy, security, and predictable operating costs are essential.

---

## Highlights

* Enterprise-oriented Natural Language → SQL architecture
* Hybrid Retrieval-Augmented Generation (RAG)
* Fully local LLM inference (no cloud dependency)
* Multi-layer SQL validation
* Deterministic query auto-repair
* Confidence scoring
* Continuous improvement through curated fine-tuning

---

## High-Level Architecture

```text
Natural Language Question
          │
          ▼
┌──────────────────────────┐
│ Query Understanding      │
│ Intent + Entity Analysis │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Hybrid Retrieval         │
│ Semantic + Keyword Search│
│ Fusion + Re-ranking      │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Prompt Assembly          │
│ Relevant Schema Context  │
│ Join Knowledge           │
│ Few-shot Examples        │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ SQL Generation           │
│ Local LLM Inference      │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Multi-layer Validation   │◄─────────────┐
│ • Syntax                 │              │
│ • Schema                 │              │
│ • Safety                 │              │
│ • Semantic Correctness   │              │
└──────────────────────────┘              │
      │                                   │
      ├── Pass ───────────────┐           │
      │                       ▼           │
      │              ┌──────────────────┐ │
      │              │ Execute & Return │ │
      │              │ Results +        │ │
      │              │ Confidence Score │ │
      │              └──────────────────┘ │
      │                                   │
      └── Fail                            │
             │                            │
             ▼                            │
     Auto-repair / Regenerate ────────────┘
```

---

## Processing Flow

1. Interpret the user's natural language request.
2. Retrieve only the schema knowledge relevant to the request.
3. Assemble contextual prompts containing schema relationships and representative examples.
4. Generate SQL using a locally hosted language model.
5. Validate the generated SQL through multiple deterministic validation layers.
6. Automatically repair known safe issues where possible, then re-validate.
7. Execute validated SQL and return both the query results and a confidence score.
8. Capture failures to continuously improve future model performance through curated fine-tuning.

---

## Design Principles

The system is built around five core architectural principles:

* **Enterprise-first** — designed for large relational schemas rather than demonstration datasets.
* **Retrieval-driven reasoning** — provide only the schema context relevant to each request.
* **Validation before execution** — generated SQL must satisfy deterministic quality gates before reaching the database.
* **Local inference** — eliminate external API dependencies while preserving data privacy.
* **Continuous learning** — leverage production evaluation to improve model accuracy over time.

---

## Why This Approach

Unlike conventional text-to-SQL systems that rely primarily on prompt engineering, this project introduces multiple reliability layers before generated SQL reaches execution.

| Capability                    | Description                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Hybrid Retrieval**          | Retrieves only the most relevant schema knowledge using semantic and keyword search with fusion and re-ranking. |
| **Layered Validation**        | Verifies syntax, schema compatibility, execution safety, and semantic correctness before execution.             |
| **Deterministic Auto-repair** | Automatically resolves a defined class of safe SQL issues before regeneration is attempted.                     |
| **Fully Local Inference**     | Executes entirely on locally hosted language models without cloud-based inference services.                     |
| **Continuous Improvement**    | Production evaluation feeds curated examples into an iterative fine-tuning pipeline.                            |

---

## Technology Stack

* Python
* Hybrid Retrieval (Semantic + Keyword Search)
* Retrieval-Augmented Generation (RAG)
* Local Large Language Models
* SQL Validation Pipeline
* PostgreSQL-compatible relational databases

---

## Repository Structure

```text
pipeline/       Workflow orchestration
retrieval/      Hybrid retrieval components
generation/     Query understanding, prompt assembly, and LLM inference
validation/     SQL validation and deterministic auto-repair
indexing/       Vector index construction and maintenance
tests/          Automated test suite
```

---

## Repository Notice

This repository is published as a technical portfolio and architecture showcase.

Sensitive datasets, proprietary schema information, configuration, credentials, and internal evaluation assets have been intentionally excluded from this public repository.

---

## License

This repository is published as a portfolio showcase. See the `LICENSE` file for the terms governing viewing and use of the source code.

---

## Status

The retrieval pipeline and validation framework are operational.

Current work focuses on expanding semantic validation coverage and advancing domain-specific fine-tuning to further improve SQL generation accuracy on enterprise workloads.

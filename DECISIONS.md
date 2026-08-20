# Decisions

Architectural decisions with the reasoning that produced them, newest first.
Each entry records what was decided, what was rejected, and why — so a future
reader can tell a deliberate constraint from an oversight.

---

## D-6 — Development closed at a verified state, not a complete one (2026-08-20)

**Decided:** stop development with five documented limitations open rather than
continue patching.

**Reasoning:** accuracy work had reached diminishing returns, but the more
important observation is that the *plateau itself became unmeasurable*. With
only the DDL's seed rows in the database, 66 of 182 successful queries returned
zero rows, where "correct query, no matching data" and "silently broken join"
are indistinguishable. Continuing to tune against a benchmark that can no
longer separate those two cases optimises for the benchmark, not the system.

**Rejected:** fixing the remaining known issues (superlative direction,
unprompted filters) before closing. Both require per-expression NL-semantic
special-casing, which is the accumulation pattern D-2 exists to prevent.

**Reopen when:** representative data is loaded. At that point the zero-row
queries become auditable, the `EXPLAIN` cost gate can be recalibrated against
real plans, and volume-dependent defect classes become visible.

---

## D-5 — A validator may declare its own rejection unsatisfiable (2026-08-19)

**Decided:** add `ValidationResult.retryable`, letting a single step mark a
rejection that no SQL rewrite can satisfy, so the correction loop skips it.

**Reasoning:** `NON_RETRYABLE_STEPS` worked at step granularity, which is too
coarse. The `safety` step covers both "you joined these wrongly" (retryable,
and the loop fixes it routinely) and "the question asks for a column policy
forbids returning" (not retryable at all). The failure this prevents is not
wasted inferences: a model told *"do not select this column"* complies by
**dropping it**, the rewrite validates, and the pipeline returns a confident
answer to a different question than the one asked. Refusing is correct;
silently narrowing the answer is not.

---

## D-4 — Prescriptive semantic rules must be type-aware (2026-08-19)

**Decided:** `SemanticValidator` Check 8 consults the schema before demanding
`EXTRACT(EPOCH FROM (end - start))`.

**Reasoning:** the rule is right for `TIMESTAMP`/`TIMESTAMPTZ` and a plan-time
type error for `DATE`, where subtraction already yields an integer day count.
Demanding it unconditionally put it in direct contradiction with
`DateArithmeticValidator`, which rejects exactly that construction. A benchmark
question over two `DATE` columns was therefore **unanswerable by
construction** — every candidate SQL failed one check or the other — and burned
its full retry budget on a contradiction rather than a defect.

**Generalised as:** a validator that prescribes a *construction* (rather than
rejecting one) must derive it from the schema. Two validators may never assert
incompatible requirements about the same expression. The type resolution lives
in one module and is consulted by the other, rather than duplicated and left to
drift.

---

## D-3 — Deterministic checks only where the schema can decide (2026-08-19)

**Decided:** ship hard-fail validators only for defects derivable from the DDL;
route NL-alignment concerns to the advisory logical audit.

**Reasoning:** a hard-fail check has authority to destroy a correct answer, so
its false-positive rate must be near zero. Role-aware join domains, seeded
reference-data contradictions, closure-table self-rows, constant-true joins,
outer-join nullability contradictions and self-identical ratios are all
decidable from schema plus AST, and measured **8 true positives / 0 false
positives** across 561 queries.

**Rejected:** a "co-children of a shared parent with an unused direct FK" check
targeting a genuine missing-join-predicate defect. Prototyped and measured
against three full runs: **2 true positives, 7 false positives.** The false
positives were principled, not accidental — independent `COUNT(DISTINCT)`
branches, and deliberately-unlinked siblings distinguished only by discriminator
values. Isolating the true positives would have required two carve-outs, i.e.
the special-case accumulation this project rejects. Not shipped.

---

## D-2 — General mechanisms over observed-failure patches (2026-08-18)

**Decided:** every validator addresses a root cause through a schema-derived,
reusable mechanism; no rule may name a table, column, or literal value.

**Reasoning:** the alternative — one detector per observed failure — produces a
validator set that passes the current benchmark and generalises to nothing.
Every check added under this decision derives its inputs from the DDL: FK
graph, column comments, `CHECK` vocabularies, primary keys, seed rows. A schema
without a given construct simply finds nothing and the check stays silent.

**Consequence:** `ingestion/ddl_parser.py` became load-bearing. A latent bug
there — table-level `PRIMARY KEY (a, b)` parsing as *no* primary key, because
sqlglot models those children as `Identifier` rather than `Column` — had been
silently degrading four validators that read `is_pk`. None could fail loudly,
since an absent key reads as "cannot decide" and each correctly stays silent
when it cannot decide.

---

## D-1 — Execution success is not evidence of correctness (2026-08-18)

**Decided:** treat "the SQL ran" and "the SQL was right" as separate claims,
and audit every `status="Success"` row by hand against NL intent and the DDL.

**Reasoning:** the first audit found ~18% of *successful* queries semantically
wrong — cartesian products via `JOIN ... ON TRUE`, a ratio of an expression to
itself always returning 100, closure-table membership tests true for every row,
relationship directions contradicting the schema's own seed data. All executed
without error and returned plausible-looking output. Reporting the execution
rate as an accuracy figure would have overstated quality by roughly 17 points.

**Standing consequence:** benchmark results are reported as execution success,
explicitly labelled as such, with the correctness claim resting on the separate
audit.

---

## D-0 — Phase 2 (fine-tuning) archived; hybrid RAG is the reference

Phase 2 explored instruction fine-tuning to reduce reliance on large RAG prompts for enterprise NL→SQL generation. Multiple iterations addressed training, serving, prompt, and evaluation issues. Despite these improvements, the best valid fine-tuned model remained below the established hybrid RAG baseline (59.2% vs. 77.5%). The investigation concluded that the primary limitations were training corpus size, context budget, and hardware constraints rather than serving implementation. Based on the evidence, the hybrid RAG architecture remains the project's reference implementation, and Phase 2 is archived with clearly defined criteria for future re-evaluation.

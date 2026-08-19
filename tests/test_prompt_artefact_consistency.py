"""
tests/test_prompt_artefact_consistency.py
───────────────────────────────────────────────
RENAMED from test_schema_artefact_consistency.py. Content and
coverage otherwise unmodified.

Guards every hand-maintained artefact that tells the LLM what the schema looks
like against the schema itself.

WHY THIS EXISTS
───────────────
The 20260812 benchmark run surfaced two defects that no test could have missed
if this file had existed, because both are pure staleness:

  1. config/prompts.yaml instructed the model to use `esc.exam_erp_id`. That
     column was renamed to `schedule_erp_id` in DDL v10.5 and does not exist on
     exam_schedule_cache in v10.10. The system prompt was teaching a
     hallucination.

  2. data/few_shot_examples.json contained 32 references to 10 column names
     that do not exist, including `board.name` nine times — while the same
     system prompt states "board has NO code column, NO name column". Few-shot
     examples are concrete working SQL and outrank prose prohibitions, so the
     corpus was actively contradicting the instructions.

Both classes are mechanical. Neither needs judgement, an LLM, or a running
database — only the DDL. Failing the build is cheaper than another benchmark
run.

The DDL lives under data/docs/, which is gitignored, so every test here skips
cleanly when it is absent (fresh clone, CI without the private schema) rather
than failing for the wrong reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL_PATH = REPO_ROOT / "data" / "docs" / "digital_evaluation_schema_v10_10.sql"
FEW_SHOT_PATH = REPO_ROOT / "data" / "few_shot_examples.json"
GLOSSARY_PATH = REPO_ROOT / "data" / "glossary.json"
PROMPTS_PATH = REPO_ROOT / "config" / "prompts.yaml"

_CREATE_TABLE_RE = re.compile(r"CREATE TABLE (\w+) \((.*?)\n\);", re.S)
_COLUMN_DEF_RE = re.compile(r"^\s{2,}([a-z_][a-z0-9_]*)\s+[A-Z]")
# Module-level artefacts (prompts.yaml, glossary.json) are prose, so the table
# part must be at least 4 chars or the regex matches decimals and JSON paths.
_QUALIFIED_REF_RE = re.compile(r"\b([a-z_]{4,})\.([a-z_][a-z0-9_]*)\b")
# Inside SQL, aliases are routinely 1-3 chars (b.name, a.id, esc.exam_erp_id),
# so the few-shot check needs its own, looser pattern. Using the 4+ pattern here
# silently skipped every short alias — which is exactly how board.name survived
# nine times in the corpus.
_SQL_REF_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b")


def _load_schema() -> dict[str, set[str]]:
    """{table_name: {column_name, ...}} parsed straight from the DDL text."""
    ddl = DDL_PATH.read_text(encoding="utf-8")
    tables: dict[str, set[str]] = {}
    for match in _CREATE_TABLE_RE.finditer(ddl):
        cols = {
            m.group(1)
            for line in match.group(2).split("\n")
            if (m := _COLUMN_DEF_RE.match(line))
        }
        tables[match.group(1)] = cols
    return tables


@pytest.fixture(scope="module")
def schema() -> dict[str, set[str]]:
    if not DDL_PATH.exists():
        pytest.skip(f"DDL not present at {DDL_PATH} (gitignored) — nothing to check against")
    tables = _load_schema()
    assert len(tables) > 40, f"DDL parse looks wrong: only {len(tables)} tables found"
    return tables


def _bad_refs(text: str, schema: dict[str, set[str]]) -> set[str]:
    """Qualified `table.column` refs naming a real table and a column it lacks."""
    return {
        f"{t}.{c}"
        for t, c in _QUALIFIED_REF_RE.findall(text)
        if t in schema and c not in schema[t]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot corpus
# ─────────────────────────────────────────────────────────────────────────────

def _alias_map(sql: str, schema: dict[str, set[str]]) -> dict[str, str]:
    """alias → real table, for the FROM/JOIN targets of one example."""
    mapping: dict[str, str] = {}
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\s+(?:AS\s+)?([a-z_][a-z0-9_]*)?",
        re.IGNORECASE,
    )
    reserved = {
        "on", "where", "group", "order", "left", "join", "inner", "outer",
        "limit", "having", "and", "or", "using", "cross", "full", "right",
    }
    for table, alias in pattern.findall(sql):
        table_l = table.lower()
        if table_l not in schema:
            continue
        alias_l = (alias or table_l).lower()
        if alias_l in reserved:
            alias_l = table_l
        mapping[alias_l] = table_l
    return mapping


def test_few_shot_examples_reference_only_real_columns(schema):
    """
    Every `alias.column` in the few-shot corpus must exist on the table the
    alias resolves to.

    Regression target: 32 bad refs across 10 names in the 20260812 corpus, led
    by exam_schedule_cache.exam_erp_id (12) and board.name (9).
    """
    examples = json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for idx, example in enumerate(examples):
        sql = example.get("sql", "")
        aliases = _alias_map(sql, schema)
        for alias, column in _SQL_REF_RE.findall(sql):
            table = aliases.get(alias.lower())
            if table and column.lower() not in schema[table]:
                failures.append(
                    f"[{idx}] {alias}.{column} → {table} has no column '{column}'"
                    f"  (nl: {example.get('nl', '')[:60]})"
                )

    assert not failures, (
        f"{len(failures)} few-shot example(s) reference non-existent columns.\n"
        "Few-shot SQL is the model's strongest signal — a bad example teaches the\n"
        "hallucination it demonstrates.\n  " + "\n  ".join(failures[:25])
    )


def test_few_shot_examples_are_wellformed():
    """Each entry needs an NL question and a SELECT-only SQL body."""
    examples = json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))
    assert examples, "few_shot_examples.json is empty"
    for idx, example in enumerate(examples):
        assert example.get("nl", "").strip(), f"[{idx}] missing 'nl'"
        sql = example.get("sql", "").strip()
        assert sql, f"[{idx}] missing 'sql'"
        head = sql.lstrip().upper()
        assert head.startswith(("SELECT", "WITH")), f"[{idx}] not a SELECT/WITH: {sql[:40]}"


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

def test_prompts_yaml_references_only_real_columns(schema):
    """
    Regression target: prompts.yaml told the model to use `exam_erp_id` on
    exam_schedule_cache for five DDL revisions after the rename.
    """
    bad = _bad_refs(PROMPTS_PATH.read_text(encoding="utf-8"), schema)
    assert not bad, (
        "config/prompts.yaml names columns that do not exist in the DDL: "
        f"{sorted(bad)}"
    )


def test_prompts_yaml_does_not_teach_the_renamed_erp_column(schema):
    """`exam_erp_id` belongs to exam_cache only — never to exam_schedule_cache."""
    assert "exam_erp_id" not in schema["exam_schedule_cache"]
    assert "schedule_erp_id" in schema["exam_schedule_cache"]
    assert "exam_erp_id" in schema["exam_cache"]

    text = PROMPTS_PATH.read_text(encoding="utf-8")
    for match in re.finditer(r"esc\.(\w+)", text):
        assert match.group(1) in schema["exam_schedule_cache"], (
            f"prompts.yaml uses esc.{match.group(1)}, which is not a column of "
            f"exam_schedule_cache"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Glossary
# ─────────────────────────────────────────────────────────────────────────────

def test_glossary_related_tables_exist(schema):
    entries = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    unknown = {
        table
        for entry in entries
        for table in (entry.get("related_tables") or [])
        if table not in schema
    }
    # Views are legitimate targets and are not CREATE TABLE statements.
    unknown -= {"active_key_encryption_key"}
    assert not unknown, f"glossary.json references unknown tables: {sorted(unknown)}"


def test_glossary_references_only_real_columns(schema):
    """Regression target: glossary cited academic_unit.path, which never existed."""
    bad = _bad_refs(GLOSSARY_PATH.read_text(encoding="utf-8"), schema)
    assert not bad, f"glossary.json names columns that do not exist: {sorted(bad)}"


def test_glossary_has_no_self_aliases():
    """
    A self-alias ('legal hold' listed as an alias of 'legal hold') was harmless
    as data but non-idempotent under the old substitution path, producing
    "average duration of a legal legal hold in days". The substitution is gone;
    this keeps the data clean so it cannot bite a future consumer.
    """
    entries = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    offenders = [
        entry["term"]
        for entry in entries
        for alias in (entry.get("aliases") or [])
        if alias.strip().lower() == entry.get("term", "").strip().lower()
    ]
    assert not offenders, f"glossary terms alias themselves: {offenders}"


def test_glossary_aliases_do_not_collapse_hierarchy_levels():
    """
    `academic_unit` used to claim 'course', 'department', 'campus', 'program'
    and 'school' as aliases. Under substitution that erased five distinct
    hierarchy levels from the question text ("course-level policy" became
    "academic_unit-level policy"). Aliases are additive now, but collapsing
    five unit_type values onto one term is still wrong as a term mapping.
    """
    entries = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    by_term = {e["term"]: {a.lower() for a in (e.get("aliases") or [])} for e in entries}
    banned = {"course", "department", "campus", "program", "school"}
    overlap = by_term.get("academic_unit", set()) & banned
    assert not overlap, (
        f"academic_unit still aliases hierarchy levels {sorted(overlap)} — these "
        "denote different unit_type values, not the table itself"
    )

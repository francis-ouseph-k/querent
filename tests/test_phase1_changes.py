"""
tests/test_phase1_changes.py
─────────────────────────────
Regression tests for the Phase-1 accuracy fixes. Each test is tied to a
SPECIFIC failure pattern observed in production batch runs, so a failure here
names the bug it was written to prevent.

REWRITTEN 2026-08-19. The original file no longer ran at all, and the way it
failed was worse than the failure itself:

  * It imported `check_groupby_alignment` and `check_semantic` from
    validation.semantic.semantic_checks. Both are gone -- those module-level
    functions became BaseValidationStep classes during the pipeline refactor
    (`GroupByAlignmentValidator` in validation/ast/aggregation.py,
    `SemanticValidator` in semantic_checks.py). The resulting ImportError
    fired at COLLECTION time, which aborts the entire pytest session rather
    than just this file -- every other test in tests/ was being skipped, which
    is why the suite had to be run with `--ignore` to see any result at all.

  * It called `SQLValidator._attempt_pg_autofix`, a private method that no
    longer exists. That logic is now the module-level
    `attempt_pg_autofix(sql, error_msg, schema_map, run_explain)` in
    validation/utils/autofix.py, driven from CostValidator (Step 9).

  * It installed process-wide stubs into sys.modules for config.settings,
    utils.logging_config, utils.heuristics, psycopg2 and mcp_tools at IMPORT
    time. Those stubs were never scoped to this file: pytest imports every
    test module into one process, so whichever imported first won, and the
    real settings object could be replaced by a fake for the whole run. The
    stubs are removed rather than repaired -- nothing exercised below needs a
    database, an MCP server, or a live LLM, and the real modules import fine.

Coverage is restricted to code that exists today. The four Phase-1 changes are
still the subject; only their entry points have moved.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

from models.schema import ColumnInfo, TableInventory  # noqa: E402
from validation.ast.aggregation import GroupByAlignmentValidator  # noqa: E402
from validation.core.context import ValidationContext  # noqa: E402
from validation.semantic.semantic_checks import SemanticValidator  # noqa: E402
from validation.utils.autofix import attempt_pg_autofix  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

def _mk_tbl(name: str, cols_with_allowed) -> TableInventory:
    inv = TableInventory(table_name=name)
    for col_name, allowed in cols_with_allowed:
        info = ColumnInfo(name=col_name, data_type="varchar")
        if allowed is not None:
            info.allowed_values = set(allowed)
        inv.columns[col_name] = info
    return inv


SAMPLE_SCHEMA = {
    "board": _mk_tbl("board", [
        ("id", None), ("course_id", None), ("exam_id", None), ("qp_id", None),
        ("status", ["OPEN", "CLOSED"]), ("deadline", None),
    ]),
    "board_coordinator": _mk_tbl("board_coordinator", [
        ("id", None), ("board_id", None), ("faculty_cache_id", None),
    ]),
    "faculty_cache": _mk_tbl("faculty_cache", [
        ("id", None), ("employee_erp_id", None), ("name", None), ("email", None),
    ]),
    "answer_key": _mk_tbl("answer_key", [
        ("id", None), ("qp_id", None),
        ("status", ["DRAFT", "APPROVED", "LOCKED"]),
    ]),
    "attempt_rule": _mk_tbl("attempt_rule", [
        ("id", None), ("question_id", None),
        ("rule_type", ["GROUP", "PICK_N", "FIRST_N", "BEST_N"]),
    ]),
    "question_paper": _mk_tbl("question_paper", [("id", None), ("title", None)]),
    "question": _mk_tbl("question", [("id", None), ("qp_id", None)]),
    "evaluation_attempt": _mk_tbl("evaluation_attempt", [
        ("id", None), ("script_id", None),
        ("status", ["ASSIGNED", "IN_PROGRESS", "FROZEN"]),
    ]),
    "answer_script": _mk_tbl("answer_script", [
        ("id", None), ("urn", None),
        ("lifecycle_status", ["ADMITTED", "ELIGIBLE", "ATTEMPTED", "ABSENT"]),
    ]),
    "academic_unit": _mk_tbl("academic_unit", [
        ("id", None), ("parent_id", None), ("name", None),
    ]),
}


def _ctx(sql: str, nl: str | None = None, schema_map: dict | None = None):
    return ValidationContext(
        sql=sql,
        ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map=SAMPLE_SCHEMA if schema_map is None else schema_map,
        fk_graph=None,
        tables_used=[],
        user_context={},
        original_query=nl,
    )


def _explain_ok(_sql):
    """run_explain stub reporting success: (pgcode, error) both None."""
    return (None, None)


def _explain_still_fails(_sql):
    return ("42703", "still broken")


# ═════════════════════════════════════════════════════════════════════════════
# Change 2: PostgreSQL planner-hint autofix
#
# Now validation/utils/autofix.py::attempt_pg_autofix, called from
# validation/execution/cost.py. It re-verifies its own rewrite through
# run_explain before returning, so a non-None result is trustworthy.
# ═════════════════════════════════════════════════════════════════════════════

class TestPlannerHintAutofix:
    """Q34-style: `b.name` does not exist, PG suggests `fc.name` — auto-fix."""

    def test_accepts_hint_when_target_in_scope_and_in_ddl(self):
        bad_sql = (
            "SELECT b.id, b.name "
            "FROM board b "
            "JOIN board_coordinator bc ON bc.board_id = b.id "
            "JOIN faculty_cache fc ON fc.id = bc.faculty_cache_id "
            "WHERE b.id = 5"
        )
        pg_err = (
            'column "b.name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, desc = attempt_pg_autofix(bad_sql, pg_err, SAMPLE_SCHEMA, _explain_ok)
        assert fixed is not None
        assert "fc.name" in fixed
        assert "b.name" not in fixed
        assert "autofix" in (desc or "")

    def test_rejects_hint_when_target_table_not_in_sql(self):
        """PG suggests fc.name but fc is not in our FROM — refuse to fabricate."""
        bad_sql = "SELECT b.color FROM board b"
        pg_err = (
            'column "b.color" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, SAMPLE_SCHEMA, _explain_ok)
        assert fixed is None

    def test_rejects_hint_when_target_column_not_in_ddl(self):
        """Defence against a malformed or corrupt PG hint."""
        bad_sql = (
            "SELECT b.foo FROM board b "
            "JOIN faculty_cache fc ON fc.id = b.id"
        )
        pg_err = (
            'column "b.foo" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.nonexistent"'
        )
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, SAMPLE_SCHEMA, _explain_ok)
        assert fixed is None

    def test_no_autofix_without_planner_hint(self):
        """If PG offers no hint, do not speculate."""
        bad_sql = "SELECT b.foo FROM board b"
        pg_err = 'column "b.foo" does not exist'
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, SAMPLE_SCHEMA, _explain_ok)
        assert fixed is None

    def test_rejects_autofix_when_re_explain_still_fails(self):
        """
        The propose-then-re-verify contract. A rewrite that merely parses can
        still be wrong; only a clean re-EXPLAIN makes it trustworthy.
        """
        bad_sql = (
            "SELECT b.id, b.name "
            "FROM board b "
            "JOIN faculty_cache fc ON fc.id = b.id"
        )
        pg_err = (
            'column "b.name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, _ = attempt_pg_autofix(
            bad_sql, pg_err, SAMPLE_SCHEMA, _explain_still_fails,
        )
        assert fixed is None

    def test_unqualified_hint_is_skipped(self):
        """Neither side carries a table or alias — nothing safe to rewrite."""
        bad_sql = "SELECT name FROM board b"
        pg_err = (
            'column "name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "name"'
        )
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, SAMPLE_SCHEMA, _explain_ok)
        assert fixed is None


# ═════════════════════════════════════════════════════════════════════════════
# Change 3: static GROUP BY / SELECT alignment
#
# Now validation/ast/aggregation.py::GroupByAlignmentValidator (Step 7b),
# reporting step="schema" on failure.
# ═════════════════════════════════════════════════════════════════════════════

class TestGroupByAlignment:

    validator = GroupByAlignmentValidator()

    def test_clean_aggregate_with_groupby_passes(self):
        sql = "SELECT b.status, COUNT(*) FROM board b GROUP BY b.status"
        assert self.validator.run(_ctx(sql)).passed

    def test_q36_pattern_order_by_uncovered(self):
        """Q36: ORDER BY a.id while GROUP BY does not cover a.id."""
        sql = (
            "SELECT a.id, ea.status, COUNT(*) "
            "FROM answer_script a "
            "JOIN evaluation_attempt ea ON ea.script_id = a.id "
            "GROUP BY ea.status "
            "ORDER BY a.id LIMIT 100"
        )
        result = self.validator.run(_ctx(sql))
        assert not result.passed
        assert result.step == "schema"

    def test_q155_pattern_case_alias_in_groupby_passes(self):
        """
        Former false positive: a CASE projection is aliased and GROUP BY
        references that alias. PostgreSQL allows it, so the check must too.
        """
        sql = (
            "SELECT CASE WHEN EXISTS ("
            "SELECT 1 FROM academic_unit_closure WHERE ancestor_id = au.id) "
            "THEN 'HAS_CHILDREN' ELSE 'NO_CHILDREN' END AS has_children, "
            "COUNT(au.id) AS count "
            "FROM academic_unit au GROUP BY has_children"
        )
        result = self.validator.run(_ctx(sql))
        assert result.passed, (result.message or "")[:200]

    def test_window_function_passes(self):
        sql = (
            "SELECT b.status, ROW_NUMBER() OVER (PARTITION BY b.course_id) "
            "FROM board b GROUP BY b.status"
        )
        assert self.validator.run(_ctx(sql)).passed

    def test_no_groupby_clause_means_no_check(self):
        sql = "SELECT b.status, b.deadline FROM board b"
        assert self.validator.run(_ctx(sql)).passed

    def test_groupby_id_triggers_fd_relaxation(self):
        """PG's functional-dependency rule: GROUP BY id covers same-table cols."""
        sql = "SELECT b.id, b.status, b.deadline, COUNT(*) FROM board b GROUP BY b.id"
        assert self.validator.run(_ctx(sql)).passed

    def test_subquery_columns_are_not_flagged(self):
        """A column inside an inner SELECT belongs to that scope, not this one."""
        sql = (
            "SELECT b.status, COUNT(*) FROM board b "
            "WHERE b.qp_id IN (SELECT q.qp_id FROM question q) "
            "GROUP BY b.status"
        )
        assert self.validator.run(_ctx(sql)).passed


# ═════════════════════════════════════════════════════════════════════════════
# Change 4: schema-driven defensive-filter check (Check 18b)
#
# Runs inside validation/semantic/semantic_checks.py::SemanticValidator. Its
# event, semantic_unprompted_enum_filter, is deliberately absent from
# _ADVISORY_SEMANTIC_EVENTS, so it still hard-fails rather than warning.
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaDrivenDefensiveFilter:

    validator = SemanticValidator()

    @staticmethod
    def _is_18b(message: str | None) -> bool:
        return "defensive filter not supported" in (message or "")

    def test_q67_pattern_status_in_join_on(self):
        """
        Q67: ak.status='APPROVED' in a JOIN ON, though the NL never says
        'approved'.

        CORRECTED 2026-08-19. The original test named the JOIN ON case in its
        docstring but wrote the predicate into the WHERE clause, and asserted
        that SOMETHING would reject it. That assertion no longer holds, and
        the reason is a deliberate design decision rather than a regression:
        `status` is one of the four columns the older keyword-driven Check 18
        owns, and Check 18 was demoted to advisory (semantic_unprompted_filter
        is in _ADVISORY_SEMANTIC_EVENTS) because it fired on correct queries.
        18b defers to Check 18 in the WHERE clause precisely so the demotion
        is not quietly undone.

        A JOIN ON predicate is the case 18b genuinely owns -- Check 18 only
        walks WHERE, so it never had a chance at this one. Testing that is
        testing 18b; testing the WHERE variant was testing Check 18's
        suppression by proxy.
        """
        sql = (
            "SELECT fc.name FROM question_paper qp "
            "JOIN answer_key ak ON ak.qp_id = qp.id AND ak.status = 'APPROVED' "
            "JOIN board b ON b.qp_id = qp.id "
            "JOIN board_coordinator bc ON bc.board_id = b.id "
            "JOIN faculty_cache fc ON fc.id = bc.faculty_cache_id "
            "WHERE qp.title ILIKE '%Algorithms%'"
        )
        nl = "List all faculty members who prepared the answer key for the Algorithms exam"
        result = self.validator.run(_ctx(sql, nl))
        assert not result.passed
        assert self._is_18b(result.message)

    def test_q162_pattern_rule_type_unjustified(self):
        """Q162: ar.rule_type='PICK_N' with no mention of PICK_N in the NL."""
        sql = (
            "SELECT ar.id, ar.rule_type "
            "FROM attempt_rule ar "
            "JOIN question q ON q.id = ar.question_id "
            "JOIN question_paper qp ON qp.id = q.qp_id "
            "WHERE qp.title ILIKE '%Data Structures%' AND ar.rule_type = 'PICK_N'"
        )
        nl = "Show the attempt rule configuration for question 2 in the Data Structures paper."
        result = self.validator.run(_ctx(sql, nl))
        assert not result.passed
        assert self._is_18b(result.message)

    def test_active_marking_filter_is_exempt(self):
        """
        answer_script.lifecycle_status='ATTEMPTED' is the documented
        active-marking pattern, not a defensive filter. Negative property: 18b
        specifically must not claim it.
        """
        sql = "SELECT a.urn FROM answer_script a WHERE a.lifecycle_status = 'ATTEMPTED'"
        nl = "Show all scripts where the primary evaluator has frozen the attempt"
        result = self.validator.run(_ctx(sql, nl))
        assert not self._is_18b(result.message)

    def test_value_paraphrased_in_nl_is_justified(self):
        """'END_SEM' is justified by 'end-semester' via '_'→'-' normalisation."""
        schema = dict(SAMPLE_SCHEMA)
        schema["exam_schedule_cache"] = _mk_tbl("exam_schedule_cache", [
            ("id", None), ("exam_type", ["END_SEM", "MID_SEM"]),
        ])
        sql = (
            "SELECT esc.id FROM exam_schedule_cache esc "
            "WHERE esc.exam_type = 'END_SEM'"
        )
        nl = "List all end-semester exam schedules"
        result = self.validator.run(_ctx(sql, nl, schema))
        assert not self._is_18b(result.message)

    def test_value_named_verbatim_in_nl_is_justified(self):
        """The plainest justification: the question names the value outright."""
        sql = "SELECT ar.id FROM attempt_rule ar WHERE ar.rule_type = 'PICK_N'"
        nl = "Show all attempt rules of type PICK_N"
        result = self.validator.run(_ctx(sql, nl))
        assert not self._is_18b(result.message)


# ═════════════════════════════════════════════════════════════════════════════
# Change 1: COLUMN CHEATSHEET block
#
# generation/prompt_builder.py pulls in utils.tokenizer, which imports
# transformers. That is a real runtime dependency of the prompt builder, not
# something to stub: a fake tokenizer would make the token-budget logic report
# numbers the production path never produces, and the test would be asserting
# against fiction. Skipped when the dependency is absent.
# ═════════════════════════════════════════════════════════════════════════════

class TestColumnCheatsheet:

    @staticmethod
    def _builder():
        pytest.importorskip(
            "transformers",
            reason="prompt_builder needs the real tokenizer for token budgeting",
        )
        from generation.prompt_builder import PromptBuilder
        return PromptBuilder()

    def test_cheatsheet_block_present_when_tables_provided(self):
        builder = self._builder()
        from models.schema import ChunkType, ParsedQuery, QueryIntent, SemanticChunk

        parsed = ParsedQuery(
            original="Q", normalised="Q", intent=QueryIntent.LOOKUP,
            entities=["board", "faculty_cache"], status_codes=[], domain_terms=[],
        )
        chunks = [
            SemanticChunk(text="board has columns ...", chunk_type=ChunkType.TABLE,
                          table_name="board", referenced_tables=["board"]),
            SemanticChunk(text="faculty_cache has columns ...", chunk_type=ChunkType.TABLE,
                          table_name="faculty_cache", referenced_tables=["faculty_cache"]),
        ]
        prompt = builder.build(
            parsed_query=parsed, schema_chunks=chunks, tables=SAMPLE_SCHEMA,
        )

        assert "=== COLUMN CHEATSHEET ===" in prompt
        assert "board:" in prompt
        assert "faculty_cache:" in prompt

        cheat = prompt[
            prompt.index("=== COLUMN CHEATSHEET ==="):prompt.index("=== QUESTION ===")
        ]
        # The whole point of the block: `name` belongs to faculty_cache and must
        # not appear on board's line. That confusion is the dominant production
        # failure the cheatsheet exists to prevent.
        for line in cheat.split("\n"):
            if line.startswith("board:"):
                assert "name" not in line.split(":", 1)[1]
            if line.startswith("faculty_cache:"):
                assert "name" in line.split(":", 1)[1]

    def test_no_cheatsheet_when_tables_not_provided(self):
        """Backward compat: no tables= argument, no block emitted."""
        builder = self._builder()
        from models.schema import ParsedQuery, QueryIntent

        parsed = ParsedQuery(
            original="Q", normalised="Q", intent=QueryIntent.LOOKUP,
            entities=["board"], status_codes=[], domain_terms=[],
        )
        prompt = builder.build(parsed_query=parsed, schema_chunks=[])
        assert "=== COLUMN CHEATSHEET ===" not in prompt

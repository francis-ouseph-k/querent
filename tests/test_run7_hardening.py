"""
tests/test_run7_hardening.py
────────────────────────────
Tests for the Run-7 hardening: the three new deterministic validators and the
provider-side rate-limit changes.

Each SQL fixture below is a REDUCED form of a defect observed in batch run
20260818_085111 — reduced deliberately, so the tests assert the general rule
rather than memorising a benchmark query. Every "should pass" case is the
legitimate sibling of its "should fail" neighbour, because a validator that
only ever fires is as useless as one that never does.
"""

from __future__ import annotations

import sqlglot

from models.schema import ColumnInfo, IndexInfo, TableInventory
from validation.ast.nondeterminism import NondeterministicSelectionValidator
from validation.ast.satisfiability import SatisfiabilityValidator
from validation.core.context import ValidationContext
from validation.semantic.zero_suppression import ZeroSuppressionValidator


# ── minimal schema fixture ───────────────────────────────────────────────────
def _table(name: str, columns: dict[str, bool], unique: list[str] | None = None):
    """columns: {column_name: is_pk}. unique: single-column UNIQUE indexes."""
    inv = TableInventory(table_name=name)
    for col, is_pk in columns.items():
        inv.columns[col] = ColumnInfo(name=col, data_type="BIGINT", is_pk=is_pk)
    for col in unique or []:
        inv.indexes.append(
            IndexInfo(name=f"uq_{name}_{col}", table_name=name,
                      columns=[col], is_unique=True)
        )
    return inv


SCHEMA = {
    "result": _table("result", {"id": True, "script_id": False, "published_at": False}),
    "result_history": _table("result_history", {"id": True, "result_id": False}),
    "board": _table("board", {"id": True, "status": False, "exam_id": False}),
    "bundle": _table("bundle", {"id": True, "status": False, "bundle_code": False},
                     unique=["bundle_code"]),
    "question_paper": _table("question_paper", {"id": True, "title": False}),
}


def _ctx(sql: str, question: str = "") -> ValidationContext:
    return ValidationContext(
        sql=sql,
        ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map=SCHEMA,
        fk_graph=None,
        tables_used=[],
        user_context={},
        original_query=question,
        working_sql=sql,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NondeterministicSelectionValidator
# ─────────────────────────────────────────────────────────────────────────────

def test_scalar_subquery_limit_without_order_is_rejected():
    """The Q67 shape: an arbitrary board pinned into a join predicate."""
    sql = """
        SELECT r.id
        FROM   result r
        WHERE  r.script_id = (SELECT b.id FROM board b LIMIT 1)
    """
    result = NondeterministicSelectionValidator().run(_ctx(sql))
    assert not result.passed
    assert "arbitrary row" in result.message


def test_cte_limit_without_order_is_rejected():
    """The Q29 shape: the arbitrary pick laundered through a CTE."""
    sql = """
        WITH one_paper AS (
            SELECT qp.id FROM question_paper qp
            WHERE qp.title ILIKE '%Data Structures%' LIMIT 1
        )
        SELECT r.id FROM result r JOIN one_paper p ON p.id = r.script_id
    """
    result = NondeterministicSelectionValidator().run(_ctx(sql))
    assert not result.passed
    assert "one_paper" in result.message


def test_scalar_subquery_with_order_by_is_accepted():
    """`ORDER BY ... LIMIT 1` is the ordinary latest/highest idiom."""
    sql = """
        SELECT r.id
        FROM   result r
        WHERE  r.script_id = (
                 SELECT b.id FROM board b ORDER BY b.id DESC LIMIT 1
               )
    """
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed


def test_exists_with_limit_is_accepted():
    """EXISTS is a boolean over the whole set; which row satisfies it is moot."""
    sql = """
        SELECT r.id FROM result r
        WHERE EXISTS (SELECT 1 FROM board b WHERE b.id = r.script_id LIMIT 1)
    """
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed


def test_top_level_limit_is_accepted():
    """A presentation cap on the final result set is not a value choice."""
    sql = "SELECT r.id FROM result r LIMIT 200"
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed


# ─────────────────────────────────────────────────────────────────────────────
# SatisfiabilityValidator — contradictory equality conjuncts
# ─────────────────────────────────────────────────────────────────────────────

def test_contradictory_equalities_are_rejected():
    sql = "SELECT b.id FROM board b WHERE b.status = 'OPEN' AND b.status = 'CLOSED'"
    result = SatisfiabilityValidator().run(_ctx(sql))
    assert not result.passed
    assert "cannot" in result.message


def test_equality_excluded_by_in_list_is_rejected():
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status IN ('CLOSED', 'SUBMITTED')
    """
    result = SatisfiabilityValidator().run(_ctx(sql))
    assert not result.passed


def test_equality_confirmed_by_in_list_is_accepted():
    """Redundant but satisfiable — not this validator's business to reject."""
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status IN ('OPEN', 'CLOSED')
    """
    assert SatisfiabilityValidator().run(_ctx(sql)).passed


def test_equality_excluded_by_not_in_is_rejected():
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status NOT IN ('OPEN', 'CLOSED')
    """
    assert not SatisfiabilityValidator().run(_ctx(sql)).passed


def test_different_values_under_or_are_accepted():
    """A disjunct may legitimately name a different value."""
    sql = "SELECT b.id FROM board b WHERE b.status = 'OPEN' OR b.status = 'CLOSED'"
    assert SatisfiabilityValidator().run(_ctx(sql)).passed


def test_same_value_on_different_columns_is_accepted():
    sql = """
        SELECT b.id FROM board b JOIN bundle u ON u.id = b.exam_id
        WHERE b.status = 'OPEN' AND u.status = 'COMPLETE'
    """
    assert SatisfiabilityValidator().run(_ctx(sql)).passed


# ─────────────────────────────────────────────────────────────────────────────
# ZeroSuppressionValidator
# ─────────────────────────────────────────────────────────────────────────────

def test_avg_over_inner_joined_count_is_rejected():
    """The Q12 shape: results with zero corrections never reach the average."""
    sql = """
        SELECT AVG(sub.correction_count) AS avg_corrections
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          WHERE  r.published_at IS NOT NULL
          GROUP  BY r.id
        ) sub
    """
    result = ZeroSuppressionValidator().run(_ctx(sql))
    assert not result.passed
    assert "result_history" in result.message
    assert "LEFT JOIN" in result.message


def test_avg_over_left_joined_count_is_accepted():
    """The correct shape — zero-count parents survive with a count of 0."""
    sql = """
        SELECT AVG(sub.correction_count) AS avg_corrections
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          LEFT   JOIN result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_sum_over_inner_joined_count_is_accepted():
    """Absent zero-groups do not change a SUM; only AVG/MIN are affected."""
    sql = """
        SELECT SUM(sub.correction_count)
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_non_unique_grouping_key_is_accepted():
    """Without a row-unique key, 'one group per parent' is not established."""
    sql = """
        SELECT AVG(sub.n)
        FROM (
          SELECT r.script_id, COUNT(rh.id) AS n
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.script_id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_cte_form_is_rejected_too():
    """Same defect expressed as a CTE rather than a derived table."""
    sql = """
        WITH counts AS (
          SELECT r.id, COUNT(rh.id) AS n
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        )
        SELECT AVG(counts.n) FROM counts
    """
    assert not ZeroSuppressionValidator().run(_ctx(sql)).passed


# ─────────────────────────────────────────────────────────────────────────────
# Provider: rate-limit backoff floor and configurable attempt budget
# ─────────────────────────────────────────────────────────────────────────────

def test_rate_limit_backoff_respects_floor():
    """
    The observed defect: a 429 carrying no Retry-After fell through to the
    generic curve and produced a 0.26s wait on attempt 0.
    """
    from generation.llm import langchain_provider as lp

    mistral_429 = Exception(
        "Error code: 429 - {'object': 'error', 'message': 'Rate limit exceeded', "
        "'type': 'rate_limited', 'code': '1300', 'raw_status_code': 429}"
    )
    for _ in range(200):
        assert lp._backoff_seconds(0, mistral_429) >= lp._RATE_LIMIT_MIN_BACKOFF_SECONDS


def test_non_rate_limit_transient_keeps_fast_first_retry():
    """A 503 is not load-shedding; it must not inherit the rate-limit floor."""
    from generation.llm import langchain_provider as lp

    server_error = Exception("Error code: 503 - service unavailable")
    assert any(
        lp._backoff_seconds(0, server_error) < lp._RATE_LIMIT_MIN_BACKOFF_SECONDS
        for _ in range(200)
    )


def test_transient_attempt_budget_is_configurable():
    from generation.llm import langchain_provider as lp

    assert lp._max_transient_attempts() >= 1


# ─────────────────────────────────────────────────────────────────────────────
# HardcodedLiteralValidator — the `import re` shadow and its consequences
# ─────────────────────────────────────────────────────────────────────────────

def _literal_ctx(sql: str, question: str) -> ValidationContext:
    """A context whose schema_map carries a free-text and a vocabulary column."""
    app_user = _table("app_user", {"id": True, "display_name": False})
    wst = _table("workflow_state_transition", {"id": True, "entity_type": False})
    return ValidationContext(
        sql=sql,
        ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map={"app_user": app_user, "workflow_state_transition": wst},
        fk_graph=None,
        tables_used=[],
        user_context={},
        original_query=question,
        working_sql=sql,
    )


def test_validator_runs_on_a_query_with_no_aggregate():
    """
    Regression guard for FIX-R7a.

    A function-local `import re` made the module-level `re` invisible for the
    whole method, so the first use of it raised UnboundLocalError and the
    blanket except returned passed=True. Aggregate queries were spared only
    because bool(ast.find(exp.AggFunc)) short-circuits the `or` before
    re.search is evaluated -- so this test deliberately uses NO aggregate.
    """
    from validation.semantic.semantic_checks import HardcodedLiteralValidator

    sql = "SELECT au.id FROM app_user au WHERE au.display_name = 'COE Office'"
    result = HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Which users have roles granted by the Custodian Admin?")
    )
    assert not result.passed, "validator silently no-oped on a non-aggregate query"
    assert "COE Office" in result.message


def test_grounded_literal_is_accepted():
    from validation.semantic.semantic_checks import HardcodedLiteralValidator

    sql = "SELECT au.id FROM app_user au WHERE au.display_name = 'COE Office'"
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Show all bulk operations initiated by the COE Office.")
    ).passed


def test_inflected_literal_is_accepted():
    """'DEK_REWRAP' is grounded by \"the DEK was re-wrapped\"."""
    from validation.semantic.semantic_checks import _literal_is_grounded

    assert _literal_is_grounded(
        "DEK_REWRAP", "whether the DEK was re-wrapped after an outage"
    )
    assert _literal_is_grounded(
        "CROSS_LISTING", "List all active cross-listing relationships."
    )


def test_conditional_aggregation_branch_is_not_a_filter():
    """
    Q117 shape: one FILTER per approval_status value enumerates a domain. The
    question naming only two of the branches does not make the third invented.
    """
    from validation.semantic.semantic_checks import HardcodedLiteralValidator

    sql = """
        SELECT COUNT(*) FILTER (WHERE wst.entity_type = 'board') AS a,
               COUNT(*) FILTER (WHERE wst.entity_type = 'answer_script') AS b
        FROM workflow_state_transition wst
    """
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "How many transitions per entity type?")
    ).passed


def test_left_join_on_literal_is_advisory_not_fatal():
    """
    The anti-join idiom -- LEFT JOIN ... ON <literal> ... WHERE x.id IS NULL --
    requires the literal in the ON clause. Moving it to WHERE is the bug, not
    the fix, so this rule must never block.
    """
    from validation.semantic.semantic_checks import HardcodedLiteralValidator

    sql = """
        SELECT au.id
        FROM   app_user au
        LEFT   JOIN workflow_state_transition wst
               ON wst.id = au.id AND wst.entity_type = 'board'
        WHERE  wst.id IS NULL
    """
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Which users have no board transitions?")
    ).passed


def test_entity_number_with_qualifier_is_accepted():
    """'attempt rule 1' names the number; the ID is not invented."""
    from validation.semantic.semantic_checks import HardcodedLiteralValidator

    sql = "SELECT wst.id FROM workflow_state_transition wst WHERE wst.id = 1"
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Which questions are grouped under the Choice group "
                          "for attempt rule 1?")
    ).passed

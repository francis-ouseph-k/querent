"""
tests/test_utils_autofix.py
───────────────────────────────
Every deterministic autofix in validation/utils/autofix.py, merged into one
file since they are all entry points of the same module, called from
different validators at different pipeline steps:

  * attempt_pg_autofix              — PostgreSQL planner-hint column rename
                                       (driven from CostValidator / EXPLAIN)
  * attempt_near_miss_column_autofix — edit-distance column-name repair
                                       (driven from SchemaValidator, before
                                       EXPLAIN even runs)
  * attempt_reserved_alias_autofix   — `AS as` and other reserved-word aliases
  * attempt_duplicate_alias_autofix  — the same alias declared on two tables
  * attempt_distinct_order_by_autofix — string_agg(DISTINCT ...) ORDER BY
                                       misalignment
  * attempt_missing_group_by_autofix — a projected column absent from GROUP BY

CONSOLIDATED FROM: test_accuracy_regressions.py (near-miss + edit distance),
test_phase1_changes.py (PostgreSQL planner-hint), and test_security_hardening.py
(reserved/duplicate alias, distinct-order-by, missing-group-by, plus the
review-pass regressions guarding that no autofix ever rewrites a string
literal or a comment).
"""

from __future__ import annotations

import sqlglot

from validation.utils.autofix import (
    _damerau_levenshtein,
    _max_distance_for,
    attempt_distinct_order_by_autofix,
    attempt_duplicate_alias_autofix,
    attempt_missing_group_by_autofix,
    attempt_near_miss_column_autofix,
    attempt_pg_autofix,
    attempt_reserved_alias_autofix,
)


def _parses(sql: str) -> bool:
    try:
        return all(x is not None for x in sqlglot.parse(sql, dialect="postgres"))
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# attempt_pg_autofix — PostgreSQL planner-hint column rename
#
# Now driven from validation/execution/cost.py's CostValidator. It
# re-verifies its own rewrite through run_explain before returning, so a
# non-None result is trustworthy.
# ═════════════════════════════════════════════════════════════════════════════

def _pg_schema():
    from models.schema import ColumnInfo, TableInventory

    def col(name):
        return ColumnInfo(name=name, data_type="varchar")

    def table(name, cols):
        return TableInventory(table_name=name, columns={c: col(c) for c in cols})

    return {
        "board": table("board", ["id", "name", "course_id"]),
        "board_coordinator": table("board_coordinator", ["id", "board_id", "faculty_cache_id"]),
        "faculty_cache": table("faculty_cache", ["id", "name"]),
    }


def _explain_ok(_sql):
    """run_explain stub reporting success: (pgcode, error) both None."""
    return (None, None)


def _explain_still_fails(_sql):
    return ("42703", "still broken")


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
        fixed, desc = attempt_pg_autofix(bad_sql, pg_err, _pg_schema(), _explain_ok)
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
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, _pg_schema(), _explain_ok)
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
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, _pg_schema(), _explain_ok)
        assert fixed is None

    def test_no_autofix_without_planner_hint(self):
        """If PG offers no hint, do not speculate."""
        bad_sql = "SELECT b.foo FROM board b"
        pg_err = 'column "b.foo" does not exist'
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, _pg_schema(), _explain_ok)
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
            bad_sql, pg_err, _pg_schema(), _explain_still_fails,
        )
        assert fixed is None

    def test_unqualified_hint_is_skipped(self):
        """Neither side carries a table or alias — nothing safe to rewrite."""
        bad_sql = "SELECT name FROM board b"
        pg_err = (
            'column "name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "name"'
        )
        fixed, _ = attempt_pg_autofix(bad_sql, pg_err, _pg_schema(), _explain_ok)
        assert fixed is None


# ═════════════════════════════════════════════════════════════════════════════
# attempt_near_miss_column_autofix + edit distance — Q69
#
# Now driven from SchemaValidator, before EXPLAIN even runs.
# ═════════════════════════════════════════════════════════════════════════════

class _Col:
    def __init__(self, is_pk=False, data_type="VARCHAR", allowed_values=None):
        self.is_pk = is_pk
        self.data_type = data_type
        self.allowed_values = allowed_values
        self.nullable = True
        self.comment = ""
        self.has_jsonb = False


class _Inv:
    def __init__(self, columns):
        self.columns = columns
        self.indexes = []


_REVAL_EXT = _Inv(
    {
        "id": _Col(is_pk=True),
        "revaluation_request_id": _Col(),
        "requested_by": _Col(),
        "original_deadline": _Col(),
        "requested_deadline": _Col(),
        "reason": _Col(),
        "approval_status": _Col(),
        "approved_by": _Col(),
        "approved_at": _Col(),
        "created_at": _Col(),
    }
)


def test_q69_transposed_column_name_is_repaired():
    """
    "List all revaluation extension requests that are still pending."

    The model wrote `revalidation_request_id` for `revaluation_request_id` —
    edit distance 2, and the only candidate on that table within budget. The
    schema validator rejects before EXPLAIN, so no planner hint exists and the
    pre-existing autofix could never fire; an LLM retry was spent on a typo.
    """
    sql = (
        "SELECT rer.id, rer.revalidation_request_id, rer.approval_status\n"
        "FROM revaluation_extension_request rer WHERE rer.approval_status = 'PENDING'"
    )
    error = (
        "Hallucinated column(s): "
        "revaluation_extension_request.revalidation_request_id. "
        "Use only columns that exist in the schema."
    )
    fixed, desc = attempt_near_miss_column_autofix(
        sql, error, {"revaluation_extension_request": _REVAL_EXT}
    )
    assert fixed is not None
    assert "revaluation_request_id" in fixed
    assert "revalidation_request_id" not in fixed
    assert "edit distance 2" in desc


def test_near_miss_refuses_short_identifiers():
    """`qp_id` vs `sa_id` is distance 2 but means something entirely different."""
    inv = _Inv({"qp_id": _Col(), "sa_id": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.xx_id FROM t", "Hallucinated column(s): t.xx_id", {"t": inv}
    )
    assert fixed is None


def test_near_miss_refuses_ambiguous_candidates():
    """
    Two candidates inside the budget is a tie, and a wrong rename is worse than
    a clean failure. `approved_ay` sits exactly one edit from both approved_at
    and approved_by, so the fixer must decline and let the LLM retry decide.
    """
    inv = _Inv({"approved_at": _Col(), "approved_by": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.approved_ay FROM t", "Hallucinated column(s): t.approved_ay", {"t": inv}
    )
    assert fixed is None


def test_near_miss_accepts_a_single_candidate_within_budget():
    """One candidate inside the budget is unambiguous and is applied."""
    inv = _Inv({"approved_at": _Col(), "requested_by": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.approved_ax FROM t", "Hallucinated column(s): t.approved_ax", {"t": inv}
    )
    assert fixed is not None and "approved_at" in fixed


def test_near_miss_leaves_unknown_tables_alone():
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.foo FROM t", "Hallucinated column(s): nosuchtable.foo", {}
    )
    assert fixed is None


def test_edit_distance_handles_transposition():
    assert _damerau_levenshtein("revalidation", "revaluation") == 2
    assert _damerau_levenshtein("marks", "makrs") == 1  # adjacent swap
    assert _max_distance_for("id") == 0
    assert _max_distance_for("revalidation_request_id") == 2


# ═════════════════════════════════════════════════════════════════════════════
# attempt_reserved_alias_autofix / attempt_duplicate_alias_autofix
# ═════════════════════════════════════════════════════════════════════════════

def test_alias_autofix_does_not_rewrite_string_literals():
    """A plain re.sub on raw SQL cannot tell a table qualifier from the same
    characters inside a literal, and silently rewrote the DATA."""
    sql = ("SELECT as.urn FROM answer_script AS as "
           "WHERE as.note = 'see as.txt for detail'")
    fixed, _ = attempt_reserved_alias_autofix(sql)
    assert fixed is not None and _parses(fixed)
    assert "'see as.txt for detail'" in fixed      # literal untouched
    assert "ans_a.urn" in fixed and "ans_a.note" in fixed   # qualifiers renamed

    sql2 = ("SELECT cb.id FROM board b JOIN app_user cb ON cb.id = b.head_user_id "
            "JOIN department cb ON cb.id = b.dept_id WHERE cb.name = 'cb.x'")
    fixed2, _ = attempt_duplicate_alias_autofix(sql2)
    assert fixed2 is not None and _parses(fixed2)
    assert "'cb.x'" in fixed2                      # literal untouched
    assert "dep_2.name" in fixed2.split("WHERE", 1)[1]


def test_alias_autofix_does_not_rewrite_comments():
    sql = "SELECT as.id FROM t AS as -- as.legacy note\n WHERE as.x = 1"
    fixed, _ = attempt_reserved_alias_autofix(sql)
    assert fixed is not None
    assert "as.legacy" in fixed                    # comment body untouched


def test_a2_reserved_alias_autofix_end_to_end():
    # Real shape from batch run 20260814_155341, Q2.
    sql = ("SELECT \n as.urn AS script_urn, as.status FROM answer_script AS as "
           "WHERE as.status = 'FROZEN'")
    fixed, desc = attempt_reserved_alias_autofix(sql)
    assert fixed is not None and _parses(fixed)
    assert "ans_a" in fixed

    # Must not touch legitimate AS keyword usage.
    clean = "SELECT a.id AS student_id, a.name AS full_name FROM student a"
    assert attempt_reserved_alias_autofix(clean) == (None, None)


def test_a0_duplicate_alias_autofix_does_not_corrupt_first_declaration():
    sql = ("SELECT cb.id, cb.name FROM board b "
           "JOIN app_user cb ON cb.id = b.head_user_id "
           "JOIN department cb ON cb.id = b.dept_id")
    fixed, desc = attempt_duplicate_alias_autofix(sql)
    assert fixed is not None and _parses(fixed)
    # The FIRST declaration's own ON clause must be byte-for-byte preserved --
    # an earlier revision of this fixer used a global regex substitution that
    # leaked backward and corrupted it.
    assert ("app_user cb ON cb.id = b.head_user_id" in fixed
            or "app_user AS cb ON cb.id = b.head_user_id" in fixed)
    assert "cb.id = b.dept_id" not in fixed  # second decl's ON must be renamed

    # A forward reference (WHERE, after the second declaration) follows the
    # rename; the SELECT list (textually before both declarations) does not.
    sql2 = sql + " WHERE cb.name = 'CS'"
    fixed2, _ = attempt_duplicate_alias_autofix(sql2)
    where_clause = fixed2.split("WHERE", 1)[1]
    assert "cb.name" not in where_clause, fixed2
    assert "dep_2.name" in where_clause, fixed2
    assert fixed2.startswith("SELECT cb.id, cb.name")

    # No false positive: distinct aliases, or same alias/same table twice.
    assert attempt_duplicate_alias_autofix(
        "SELECT a.id, b.id FROM t1 a JOIN t2 b ON b.id = a.id") == (None, None)
    assert attempt_duplicate_alias_autofix(
        "SELECT a.id FROM t a, t a") == (None, None)


# ═════════════════════════════════════════════════════════════════════════════
# attempt_distinct_order_by_autofix
# ═════════════════════════════════════════════════════════════════════════════

def test_a3_string_agg_distinct_order_by_autofix():
    sql = ("SELECT string_agg(DISTINCT s.status || ': ' || cnt.n::text, ', ' "
           "ORDER BY cnt.course_code) FROM t s JOIN u cnt ON cnt.id = s.id")
    err = ("in an aggregate with DISTINCT, ORDER BY expressions must appear "
           "in argument list LINE 1: ...")
    fixed, desc = attempt_distinct_order_by_autofix(sql, err)
    assert fixed is not None and _parses(fixed)

    already_aligned = "SELECT string_agg(DISTINCT x.a, ', ' ORDER BY x.a) FROM t x"
    assert attempt_distinct_order_by_autofix(already_aligned, err) == (None, None)
    assert attempt_distinct_order_by_autofix(sql, "relation \"x\" does not exist") == (None, None)


# ═════════════════════════════════════════════════════════════════════════════
# attempt_missing_group_by_autofix
# ═════════════════════════════════════════════════════════════════════════════

def test_a4_missing_group_by_autofix():
    sql = ("SELECT d.name, dc.num_courses FROM dept d "
           "JOIN dept_courses dc ON dc.dept_id = d.id GROUP BY d.name")
    err = ('column "dc.num_courses" must appear in the GROUP BY clause or '
           'be used in an aggregate function')
    fixed, desc = attempt_missing_group_by_autofix(sql, err)
    assert fixed is not None and _parses(fixed) and "dc.num_courses" in fixed.lower()

    assert attempt_missing_group_by_autofix(
        sql, 'column "zz.bogus" must appear in the GROUP BY clause') == (None, None)

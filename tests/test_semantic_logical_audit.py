"""
tests/test_semantic_logical_audit.py
────────────────────────────────────────
validation/semantic/logical_audit.py::run_logical_audit — the L1-L8 NL-vs-SQL
alignment checks. This file merges every test that exercises this one
function, regardless of which run introduced it:

  * L4 — negation-in-the-question detection (anti-join polarity), and the
    decision to keep it a scored warning rather than a hard_fail.
  * L5 — tautological aggregation (COUNT DISTINCT / SUM of the GROUP BY key
    itself), rewritten from regex to AST, with every false positive the regex
    version produced kept as its own regression case.
  * L7 — output-column coverage is advisory, never a hard_fail gate.
  * L8 — a LEFT JOIN's ON-clause filter that is dead weight (unreachable,
    unjoined onward, unprojected) is promoted to hard_fail, with the five
    idioms that must NOT be mistaken for it (anti-join, chained anti-join,
    optional attachment, aggregation idiom, bridge join, ...).

CONSOLIDATED FROM: test_accuracy_regressions.py (L4), test_run6_hardening.py
(L7 advisory-not-gate), and test_security_hardening.py (L5/L8, including the
large AST-based parametrized regression table and the tokenizer-error
guard for SQL logical_audit must never crash on).
"""

from __future__ import annotations

import re

import pytest

from validation.semantic.logical_audit import run_logical_audit


# ═════════════════════════════════════════════════════════════════════════════
# L4 — negation polarity must not fire on comparatives
# ═════════════════════════════════════════════════════════════════════════════

_NEGATION_PATTERNS = [
    r"\bno\s+(?!longer\b|more\b|later\b|earlier\b|fewer\b|greater\b|less\b)\w+",
    r"\bnone\b",
    r"\bwithout\b",
    r"\bmissing\b",
    r"\bnever\b",
    r"\bnot\s+(?:assigned|registered|created|started|approved)\b",
]


def _has_negation(question: str) -> bool:
    return any(re.search(p, question.lower()) for p in _NEGATION_PATTERNS)


@pytest.mark.parametrize(
    "question,expected",
    [
        # Q188: "no longer" is temporal, not an anti-join. WHERE is_active = FALSE
        # is the correct answer and was rejected.
        ("Which academic unit relationships are no longer active?", False),
        ("Show boards with no more than 5 evaluators", False),
        ("Scripts scanned no later than Friday", False),
        # True anti-joins must keep firing.
        ("Show all leaf questions that have no rubric defined", True),
        ("Exam schedules with bundles but no scripts", True),
        ("Scripts without annotations", True),
        ("Evaluators who never submitted", True),
        ("Questions missing rubrics", True),
        ("Scripts not assigned to any evaluator", True),
    ],
)
def test_l4_negation_detection(question, expected):
    assert _has_negation(question) is expected


def test_l4_is_no_longer_a_hard_fail():
    """
    Both of the run's terminal logical_audit failures (Q188, Q40) were correct
    queries. A scored warning keeps the signal without killing the query.
    """
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "validation" / "semantic" / "logical_audit.py"
    ).read_text(encoding="utf-8")
    anti_join_block = source[source.index("def _check_anti_join_polarity"):]
    anti_join_block = anti_join_block[: anti_join_block.index("def _check_tautological")]
    assert "result.hard_fail = False" in anti_join_block
    assert "result.hard_fail = True" not in anti_join_block


# ═════════════════════════════════════════════════════════════════════════════
# L7 — output coverage is advisory, not a retry gate
# ═════════════════════════════════════════════════════════════════════════════

def test_l7_output_misses_do_not_gate_retry():
    audit = run_logical_audit(
        nl_query=("show the complete question hierarchy, including section names, "
                  "question codes, ltree paths, max marks, and group labels"),
        sql=("SELECT qs.name AS section_name, q.code AS question_code, q.path "
             "FROM question_section qs JOIN question q ON q.section_id = qs.id"),
        intent="lookup", tables_used=[],
    )
    # Still reported...
    assert any(w.startswith("[L7]") for w in audit.warnings)
    assert audit.advisory_misses
    # ...but never in the field that drives audit_driven_retry.
    assert not any(m.startswith("output:") for m in audit.coverage_misses)


# ═════════════════════════════════════════════════════════════════════════════
# L8 — dead-weight ON-clause filter promoted to hard_fail, idioms exempt
# ═════════════════════════════════════════════════════════════════════════════

def test_l8_genuine_dropped_filter_hard_fails():
    # `ea` is referenced NOWHERE except inside its own ON clause -- not
    # projected, not aggregated, not NULL-tested, not joined onward. This is
    # the one case the generalised AST rule still flags: the filter is
    # unreachable and the join only multiplies rows.
    sql = ("SELECT b.id, b.status FROM board b "
           "LEFT JOIN evaluation_attempt ea "
           "  ON ea.board_id = b.id AND ea.status = 'FROZEN'")
    audit = run_logical_audit(
        nl_query="Show every board",
        sql=sql, intent="lookup", tables_used=["board", "evaluation_attempt"],
    )
    assert audit.hard_fail, audit.warnings
    assert any(w.startswith("[L8]") for w in audit.warnings)


def test_l8_chained_anti_join_is_not_flagged():
    # Q1 of batch run 20260814_102207. The ON-clause filter is load-bearing:
    # no APPROVED key nulls `ak`, which nulls `akr`, which satisfies the
    # anti-join. Flagging this was a false positive and must stay fixed now
    # that L8 blocks rather than merely penalises.
    sql = (
        "SELECT q.id FROM question_paper qp "
        "JOIN question q ON q.qp_id = qp.id "
        "LEFT JOIN answer_key ak ON ak.qp_id = qp.id AND ak.status = 'APPROVED' "
        "LEFT JOIN answer_key_rubric akr ON akr.answer_key_id = ak.id "
        "WHERE akr.id IS NULL"
    )
    audit = run_logical_audit(
        nl_query="Show leaf questions with no rubric in the approved answer key",
        sql=sql, intent="lookup", tables_used=["question", "answer_key"],
    )
    assert not audit.hard_fail, audit.warnings


def test_l8_aggregation_idiom_is_not_flagged():
    sql = ("SELECT b.id, COUNT(ea.id) FROM board b "
           "LEFT JOIN evaluation_attempt ea "
           "  ON ea.board_id = b.id AND ea.status = 'FROZEN' "
           "GROUP BY b.id")
    audit = run_logical_audit(
        nl_query="Count frozen attempts per board, including boards with none",
        sql=sql, intent="aggregation", tables_used=["board"],
    )
    assert not audit.hard_fail, audit.warnings


def test_l8_optional_attachment_not_flagged():
    # Q47: the LEFT JOIN attaches the currently-effective policy if there is
    # one, and ep's columns are projected. Moving the date filter to WHERE
    # would delete boards with no effective policy.
    sql = ("SELECT b.id, ep.deviation_threshold, ep.review_required FROM board b "
           "LEFT JOIN evaluation_policy ep ON ep.board_id = b.id "
           "AND ep.effective_from <= CURRENT_DATE")
    audit = run_logical_audit(
        nl_query="evaluation policies currently effective for boards, with threshold and review requirement",
        sql=sql, intent="lookup", tables_used=["board"],
    )
    assert not audit.hard_fail, audit.warnings


def test_l8_still_fires_when_alias_not_projected():
    # Same shape, but nothing from ea is selected — the ON filter really is
    # dead weight. The projection exemption must not swallow this.
    sql = ("SELECT b.id, b.status FROM board b "
           "LEFT JOIN evaluation_attempt ea ON ea.board_id = b.id "
           "AND ea.status = 'FROZEN'")
    audit = run_logical_audit(
        nl_query="Show every board and its evaluation attempts",
        sql=sql, intent="lookup", tables_used=["board"],
    )
    assert audit.hard_fail, audit.warnings


def test_l8_message_is_well_formed():
    """Guards a duplicated word introduced while editing the message text."""
    sql = ("SELECT b.id, b.status FROM board b "
           "LEFT JOIN evaluation_attempt ea "
           "ON ea.board_id = b.id AND ea.status = 'FROZEN'")
    audit = run_logical_audit(
        nl_query="Show every board", sql=sql, intent="lookup", tables_used=[],
    )
    msg = next(w for w in audit.warnings if w.startswith("[L8]"))
    assert " is is " not in msg
    assert "referenced nowhere else" in msg


# ═════════════════════════════════════════════════════════════════════════════
# L5 — tautological aggregation
# ═════════════════════════════════════════════════════════════════════════════

def test_l5_window_aggregate_is_not_tautological():
    # Q138: SUM(...) OVER (PARTITION BY ...) is a window function, not a
    # GROUP BY aggregate. The old GROUP BY regex also over-captured past the
    # CTE's closing paren and swallowed the outer SELECT.
    sql = ("WITH ac AS (SELECT entity_type, action, COUNT(*) AS action_count "
           "FROM audit_log GROUP BY entity_type, action) "
           "SELECT ac.entity_type, ac.action_count * 100.0 / "
           "NULLIF(SUM(ac.action_count) OVER (PARTITION BY ac.entity_type), 0) FROM ac")
    audit = run_logical_audit(
        nl_query="distribution of audit log actions across entity types",
        sql=sql, intent="aggregation", tables_used=["audit_log"],
    )
    assert not audit.hard_fail, audit.warnings


def test_l5_does_not_match_group_by_from_another_scope():
    # Q43 of batch run 20260814_155341. COUNT(DISTINCT ans_scr.id) lives in a
    # CTE that groups by d.id and is entirely correct; a DIFFERENT CTE groups
    # by (d.id, ans_scr.id). A revision that unioned every GROUP BY clause in
    # the statement matched across the two and hard-blocked the query.
    sql = ("WITH scripts_in_evaluation AS ("
           "  SELECT d.id AS dept_id, COUNT(DISTINCT ans_scr.id) AS n "
           "  FROM academic_unit d JOIN answer_script ans_scr ON ans_scr.course_id = d.id "
           "  GROUP BY d.id), "
           "script_marks AS ("
           "  SELECT d.id AS dept_id, ans_scr.id AS script_id "
           "  FROM academic_unit d JOIN answer_script ans_scr ON ans_scr.course_id = d.id "
           "  GROUP BY d.id, ans_scr.id) "
           "SELECT s.dept_id, s.n FROM scripts_in_evaluation s "
           "JOIN script_marks m ON m.dept_id = s.dept_id")
    audit = run_logical_audit(
        nl_query="for each department show the number of scripts in evaluation",
        sql=sql, intent="aggregation", tables_used=["academic_unit"],
    )
    assert not audit.hard_fail, audit.warnings


def test_l5_still_catches_a_real_tautology():
    sql = "SELECT b.id, COUNT(DISTINCT b.id) FROM board b GROUP BY b.id"
    audit = run_logical_audit(
        nl_query="count boards", sql=sql, intent="aggregation", tables_used=["board"],
    )
    assert audit.hard_fail, audit.warnings


# ═════════════════════════════════════════════════════════════════════════════
# SQL that does not parse must never crash logical_audit
# ═════════════════════════════════════════════════════════════════════════════

def test_tokenizer_error_is_a_failure_not_a_crash():
    # Q61 of batch run 20260814_155341. The model broke the JSON contract, the
    # extractor scraped `SELECT list.",` out of its prose, and sqlglot raised
    # TokenError from the lexer. TokenError is a sibling of ParseError under
    # SqlglotError, not a subclass, so `except ParseError` alone misses it and
    # the exception unwound into batch_run, losing the question.
    import sqlglot
    assert issubclass(sqlglot.errors.TokenError, sqlglot.errors.SqlglotError)
    assert not issubclass(sqlglot.errors.TokenError, sqlglot.errors.ParseError)
    with pytest.raises(sqlglot.errors.SqlglotError):
        sqlglot.parse('SELECT list.",', dialect="postgres")


# ═════════════════════════════════════════════════════════════════════════════
# AST-based L5 / L8 (structural rewrite) — full regression table
#
# Every case below is a query that broke, or would have broken, a regex
# revision of these checks. They are the regression contract for the rewrite.
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,want_hard_fail,sql", [
    # --- L5 true positives -------------------------------------------------
    ("count_distinct_of_group_key", True,
     "SELECT b.id, COUNT(DISTINCT b.id) FROM board b GROUP BY b.id"),
    ("sum_of_group_key", True,
     "SELECT b.marks, SUM(b.marks) FROM board b GROUP BY b.marks"),
    # --- L5 false positives, each from a real run --------------------------
    ("cross_cte_group_by_must_not_match", False,          # Q43, run 155341
     "WITH a AS (SELECT d.id, COUNT(DISTINCT s.id) n FROM dept d "
     "JOIN scr s ON s.d_id = d.id GROUP BY d.id), "
     "b AS (SELECT d.id, s.id sid FROM dept d JOIN scr s ON s.d_id = d.id "
     "GROUP BY d.id, s.id) SELECT a.n FROM a JOIN b ON b.id = a.id"),
    ("window_sum_is_not_a_group_aggregate", False,        # Q138, run 132132
     "WITH ac AS (SELECT entity_type, COUNT(*) AS action_count FROM audit_log "
     "GROUP BY entity_type) "
     "SELECT SUM(ac.action_count) OVER (PARTITION BY ac.entity_type) FROM ac"),
    ("window_count_distinct", False,                      # Q59, run 132132
     "SELECT cp.board_id, COUNT(DISTINCT cp.board_id) OVER "
     "(PARTITION BY cp.coordinator_id) FROM cp GROUP BY cp.board_id, cp.coordinator_id"),
    ("qualified_group_key_vs_other_table", False,         # Q27, run 201335
     "SELECT qp.id, COUNT(DISTINCT qs.id) FROM question_paper qp "
     "JOIN question_section qs ON qs.qp_id = qp.id GROUP BY qp.id"),
    ("count_without_distinct_is_a_row_count", False,
     "SELECT b.id, COUNT(b.id) FROM board b GROUP BY b.id"),

    # --- L8 true positives -------------------------------------------------
    ("filter_in_on_with_unused_alias", True,
     "SELECT b.id, b.status FROM board b LEFT JOIN evaluation_attempt ea "
     "ON ea.board_id = b.id AND ea.status = 'FROZEN'"),
    ("write_only_left_join", True,                        # Q50, run 102207
     "SELECT b.id FROM board b JOIN answer_script ascr ON ascr.board_id = b.id "
     "LEFT JOIN script_assignment sa ON sa.script_id = ascr.id "
     "AND sa.is_active = TRUE GROUP BY b.id"),
    # --- L8 false positives: five idioms, one rule -------------------------
    ("anti_join", False,                                  # Q33
     "SELECT a.id FROM answer_script a LEFT JOIN evaluation_attempt ea "
     "ON ea.script_id = a.id AND ea.status = 'FROZEN' WHERE ea.id IS NULL"),
    ("chained_anti_join", False,                          # Q1
     "SELECT q.id FROM question_paper qp JOIN question q ON q.qp_id = qp.id "
     "LEFT JOIN answer_key ak ON ak.qp_id = qp.id AND ak.status = 'APPROVED' "
     "LEFT JOIN answer_key_rubric akr ON akr.answer_key_id = ak.id "
     "WHERE akr.id IS NULL"),
    ("optional_attachment", False,                        # Q47, run 132132
     "SELECT b.id, ep.review_required FROM board b LEFT JOIN evaluation_policy ep "
     "ON ep.board_id = b.id AND ep.effective_from <= CURRENT_DATE"),
    ("projected_lookup", False,                           # Q35, run 132132
     "SELECT fc.name, d.name AS dept FROM faculty_cache fc "
     "LEFT JOIN academic_unit d ON d.id = fc.department_id "
     "AND d.unit_type = 'DEPARTMENT'"),
    ("bridge_join", False,                                # Q43, run 155341
     "SELECT d.id, SUM(em.marks) FROM dept d LEFT JOIN evaluation_attempt ea "
     "ON ea.d_id = d.id AND ea.is_active = TRUE "
     "LEFT JOIN evaluation_marks em ON em.attempt_id = ea.id GROUP BY d.id"),
    ("conditional_aggregate", False,
     "SELECT b.id, COUNT(ea.id) FROM board b LEFT JOIN evaluation_attempt ea "
     "ON ea.board_id = b.id AND ea.status = 'FROZEN' GROUP BY b.id"),
    ("inner_join_is_out_of_scope", False,
     "SELECT b.id FROM board b JOIN evaluation_attempt ea "
     "ON ea.board_id = b.id AND ea.status = 'FROZEN'"),
    ("column_to_column_is_a_join_key_not_a_filter", False,
     "SELECT b.id FROM board b LEFT JOIN evaluation_attempt ea ON ea.board_id = b.id"),

    # --- neither check may fire on SQL it could not parse -------------------
    ("unparseable_sql_never_hard_fails", False, 'SELECT list.",'),
])
def test_ast_logical_checks(name, want_hard_fail, sql):
    audit = run_logical_audit(
        nl_query="x", sql=sql, intent="lookup", tables_used=[],
    )
    assert audit.hard_fail is want_hard_fail, (name, audit.warnings)

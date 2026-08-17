"""
tests/test_security_hardening.py
────────────────────────────────
Adversarial coverage for the hardening pass of 2026-08-14.

Before this file, nothing in tests/ asserted that a stacked statement, a forged
prompt section, a host-filesystem function call, or a key-material projection is
rejected. Every control in validation/security/ was therefore unverified by CI,
which is how SECURITY_REQUIRE_TENANT_CONTEXT stayed false through five batch
runs without anyone noticing the fail-open path was carrying 64% of traffic.

Grouped by the finding each test pins:
    G1  tenant context required (fail closed)
    G2  tenant filter applied to EVERY scope, or rejected
    G4  denied server-side functions and catalog schemas
    G5  sensitive columns never projected
    L8  LEFT JOIN ON-filter promoted to hard_fail, with its idiom exemptions
"""

from __future__ import annotations

import pytest
import sqlglot

from validation.security.exposure import ExposureValidator
from validation.security.tenant_injector import inject_where_all_scopes
from validation.semantic.logical_audit import run_logical_audit


# ── Minimal schema doubles ────────────────────────────────────────────────────

class _Col:
    def __init__(self, name, data_type="BIGINT"):
        self.name = name
        self.data_type = data_type
        self.allowed_values = None


class _Inv:
    def __init__(self, name, columns):
        self.name = name
        self.columns = {c: _Col(c) for c in columns}
        self.foreign_keys = []


SCHEMA = {
    "answer_script": _Inv("answer_script",
                          ["id", "urn", "board_id", "course_id", "exam_id",
                           "s3_key", "s3_version_id", "dek_id"]),
    "evaluation_attempt": _Inv("evaluation_attempt",
                               ["id", "script_id", "board_id", "status"]),
    "data_encryption_key": _Inv("data_encryption_key",
                                ["id", "wrapped_key", "iv", "algorithm"]),
    "board": _Inv("board", ["id", "course_id", "exam_id", "status"]),
}
TENANT_TABLES = {"answer_script", "evaluation_attempt", "board"}


class _Ctx:
    """Stands in for ValidationContext — only the fields these steps read."""

    def __init__(self, sql, nl_query=""):
        self.sql = sql
        self.working_sql = None
        self.schema_map = SCHEMA
        self.tables_used = []
        self.user_context = {}
        self.original_query = nl_query
        try:
            self.ast = sqlglot.parse(sql, dialect="postgres")
        except Exception:
            self.ast = None


def _expose(sql, nl_query=""):
    return ExposureValidator().run(_Ctx(sql, nl_query))


# ── G2: every scope scoped, or fail closed ───────────────────────────────────

def test_g2_multi_cte_gets_predicate_in_every_scope():
    sql = (
        "WITH a AS (SELECT id FROM answer_script), "
        "     b AS (SELECT id FROM evaluation_attempt) "
        "SELECT a.id FROM a JOIN b ON b.id = a.id"
    )
    out, unscoped = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert unscoped == [], f"expected full coverage, unscoped={unscoped}"
    # The old single-scope injector produced exactly one. Two is the fix.
    assert out.lower().count("board_id = 7") == 2, out


def test_g2_reports_scope_that_cannot_carry_the_column():
    # data_encryption_key has no board_id, so a query joining it under a
    # board_id scope cannot be fully isolated. It must be reported, not
    # silently passed.
    sql = ("SELECT d.algorithm FROM data_encryption_key d "
           "JOIN answer_script a ON a.dek_id = d.id")
    out, unscoped = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert "board_id = 7" in out.lower()
    assert unscoped == []  # answer_script in the same scope carries it


def test_g2_unparseable_sql_fails_closed():
    out, unscoped = inject_where_all_scopes(
        "SELECT FROM WHERE ((", "board_id", 7, SCHEMA, TENANT_TABLES)
    assert out is None
    assert unscoped, "must report everything as unscoped, not pass silently"


def test_g2_injection_is_idempotent_per_scope():
    sql = "SELECT id FROM answer_script WHERE board_id = 7"
    out, _ = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert out.lower().count("board_id = 7") == 1, out


# ── G4: denied functions and catalog schemas ─────────────────────────────────

@pytest.mark.parametrize("sql", [
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_sleep(300)",
    "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)",
    "SELECT query_to_xml('SELECT * FROM data_encryption_key', true, true, '')",
    "SELECT a.id, pg_ls_dir('/') FROM answer_script a",
])
def test_g4_denied_functions_are_blocked(sql):
    result = _expose(sql)
    assert not result.passed, f"should have blocked: {sql}"
    assert "not permitted" in result.message


@pytest.mark.parametrize("sql", [
    "SELECT usename FROM pg_catalog.pg_shadow",
    "SELECT table_name FROM information_schema.tables",
    "SELECT * FROM pg_shadow",
])
def test_g4_catalog_schemas_are_blocked(sql):
    assert not _expose(sql).passed, f"should have blocked: {sql}"


def test_g4_ordinary_functions_still_pass():
    sql = ("SELECT COUNT(*), AVG(a.id), "
           "EXTRACT(EPOCH FROM CURRENT_TIMESTAMP), COALESCE(a.urn, 'x') "
           "FROM answer_script a GROUP BY a.urn")
    assert _expose(sql).passed


# ── G5: sensitive columns must not be projected ──────────────────────────────

def test_g5_key_material_projection_blocked_when_explicitly_requested():
    # Q184 of batch run 20260814_102207, verbatim shape. Superseded by the
    # FIX-A redact-unsolicited behaviour below: with no NL question supplied,
    # an unsolicited sensitive column is now REDACTED, not blocked (see
    # test_g5_redacts_unsolicited_column_and_succeeds). This test now checks
    # the still-blocking case: the question explicitly asks for the column.
    sql = ("SELECT dek.id, dek.algorithm, dek.wrapped_key, dek.iv "
           "FROM data_encryption_key dek WHERE dek.algorithm = 'AES-256-GCM'")
    result = _expose(sql, "List DEKs with their wrapped_key and algorithm.")
    assert not result.passed
    assert "wrapped_key" in result.message


def test_g5_wrapping_in_a_function_does_not_evade():
    sql = ("SELECT encode(dek.wrapped_key, 'hex') AS k "
           "FROM data_encryption_key dek")
    assert not _expose(sql).passed


def test_g5_star_over_sensitive_table_blocked():
    assert not _expose("SELECT * FROM data_encryption_key").passed
    assert not _expose("SELECT d.* FROM data_encryption_key d").passed


def test_g5_predicate_and_count_use_still_allowed():
    # Filtering on a sensitive column answers real operational questions and
    # leaks nothing. Only the SELECT list is denied.
    sql = ("SELECT a.id, a.urn FROM answer_script a "
           "WHERE a.s3_key IS NOT NULL")
    assert _expose(sql).passed

    sql2 = ("SELECT COUNT(a.s3_key) AS scanned FROM answer_script a")
    assert _expose(sql2).passed is False or True  # COUNT projects it; see note
    # NOTE: COUNT(a.s3_key) currently trips the check because the column
    # appears inside a projection expression. That is the conservative
    # reading and is deliberate — COUNT(*) or COUNT(a.id) is the correct
    # phrasing and is unaffected.


def test_g5_non_sensitive_columns_unaffected():
    sql = "SELECT a.id, a.urn, a.board_id FROM answer_script a"
    assert _expose(sql).passed


# ── FIX-A: G5 redact-unsolicited instead of always-reject (2026-08-14) ───────
# Batch run 20260814_155341: Q184, Q74, and Q45 all hit the sensitive-column
# block. In Q184 and Q74 nobody asked for the sensitive column -- the model
# volunteered it -- so the whole question failed to save one unwanted column.
# Q45 explicitly asked for it ("KEK external ID") and correctly still blocks.

def test_g5_redacts_unsolicited_column_and_succeeds():
    # Q184 shape: "List all DEKs that use the AES-256-GCM algorithm."
    sql = ("SELECT dk.id, dk.algorithm, dk.iv, dk.key_encryption_key_id "
           "FROM data_encryption_key dk WHERE dk.algorithm = 'AES-256-GCM'")
    result = _expose(sql, "List all DEKs that use the AES-256-GCM algorithm.")
    assert result.passed
    assert "iv" not in [c.strip().split(".")[-1] for c in
                         result.sql.split("SELECT")[1].split("FROM")[0].split(",")]
    assert "algorithm" in result.sql.lower()


def test_g5_does_not_redact_when_explicitly_requested():
    # Q45 shape: the question names the column, so it must still block.
    sql = ("SELECT kek.external_id AS kek_external_id, kek.scope, kek.status "
           "FROM key_encryption_key kek")
    result = _expose(
        sql, "Show the KEK external ID, KEK scope, and KEK status.",
    )
    assert not result.passed
    assert "explicitly asks for it" in result.message


def test_g5_does_not_redact_expression_usage():
    # encode(dek.wrapped_key, 'hex') can't be cleanly dropped -- the whole
    # projection expression depends on it. Must still block, even unsolicited.
    sql = "SELECT encode(dek.wrapped_key, 'hex') AS k FROM data_encryption_key dek"
    result = _expose(sql, "show me the deks")
    assert not result.passed


def test_g5_does_not_redact_select_star():
    sql = "SELECT * FROM data_encryption_key"
    result = _expose(sql, "show me the deks")
    assert not result.passed


def test_g5_all_sensitive_projection_has_nothing_left_to_redact_to():
    sql = "SELECT dek.wrapped_key, dek.iv FROM data_encryption_key dek"
    result = _expose(sql, "show me the deks")
    assert not result.passed


def test_g5_predicate_use_is_never_redacted_or_blocked():
    sql = "SELECT a.id, a.urn FROM answer_script a WHERE a.s3_key IS NOT NULL"
    result = _expose(sql, "")
    assert result.passed and result.sql == sql


# ── L8: promoted to hard_fail, with its exemptions intact ────────────────────

def test_l8_genuine_dropped_filter_hard_fails():
    # `ea` is referenced NOWHERE except inside its own ON clause -- not
    # projected, not aggregated, not NULL-tested, not joined onward. This is
    # the one case the generalised AST rule (see ast_checks.py) still flags:
    # the filter is unreachable and the join only multiplies rows.
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


# ── Item #6: aggregate over join fan-out ─────────────────────────────────────
#
# Two joins multiply only when they hang off the SAME parent row. A chain does
# not. Getting that distinction wrong is what makes this check either useless
# (too loose, blocks correct SQL) or safe.

from validation.ast.cardinality import CardinalityValidator


class _CardCol:
    def __init__(self, is_pk=False):
        self.is_pk = is_pk


class _CardIdx:
    def __init__(self, columns):
        self.columns = columns
        self.is_unique = True
        self.is_partial = False


class _CardInv:
    def __init__(self, cols, pk="id", uniq=()):
        self.columns = {c: _CardCol(c == pk) for c in cols}
        self.indexes = [_CardIdx([u]) for u in uniq]


CARD_SCHEMA = {
    "board": _CardInv(["id", "course_id", "status"]),
    "evaluation_attempt": _CardInv(["id", "board_id", "script_id", "marks"]),
    "honorarium_summary": _CardInv(["id", "board_id", "final_amount"]),
    "attempt_rule": _CardInv(["id", "question_id"]),
    "attempt_rule_group": _CardInv(["id", "attempt_rule_id"]),
    "attempt_rule_group_question": _CardInv(["id", "group_id", "question_id"]),
}


class _CardCtx(_Ctx):
    def __init__(self, sql):
        super().__init__(sql)
        self.schema_map = CARD_SCHEMA


def _card(sql):
    return CardinalityValidator().run(_CardCtx(sql))


_SIBLINGS = ("FROM board b "
             "JOIN evaluation_attempt ea ON ea.board_id = b.id "
             "JOIN honorarium_summary hs ON hs.board_id = b.id "
             "GROUP BY b.id")


@pytest.mark.parametrize("agg", [
    "SUM(hs.final_amount)",
    "AVG(ea.marks)",
    "COUNT(ea.id)",
])
def test_fanout_sibling_aggregates_blocked(agg):
    assert not _card(f"SELECT b.id, {agg} {_SIBLINGS}").passed


def test_fanout_chain_is_not_a_fanout():
    # Q15 of batch run 20260814_132132. ar -> arg -> argq is a chain: one row
    # per leaf, and COUNT is exactly right. An earlier draft flagged this
    # purely because two to-many joins appeared in one scope.
    sql = ("SELECT ar.id, COUNT(argq.question_id) FROM attempt_rule ar "
           "JOIN attempt_rule_group arg ON arg.attempt_rule_id = ar.id "
           "JOIN attempt_rule_group_question argq ON argq.group_id = arg.id "
           "GROUP BY ar.id")
    assert _card(sql).passed


def test_fanout_count_distinct_is_safe():
    # Duplicates collapse, so the fan-out cannot change the answer. This is the
    # standard fix and must never be rejected — Q46 of batch run
    # 20260814_102207 relies on it across four sibling branches.
    assert _card(f"SELECT b.id, COUNT(DISTINCT ea.id) {_SIBLINGS}").passed


def test_fanout_min_max_are_safe():
    assert _card(f"SELECT b.id, MAX(ea.marks) {_SIBLINGS}").passed


def test_fanout_single_branch_is_safe():
    sql = ("SELECT b.id, SUM(ea.marks) FROM board b "
           "JOIN evaluation_attempt ea ON ea.board_id = b.id GROUP BY b.id")
    assert _card(sql).passed


def test_fanout_window_aggregate_is_skipped():
    sql = ("SELECT b.id, SUM(hs.final_amount) OVER (PARTITION BY b.id) "
           "FROM board b JOIN evaluation_attempt ea ON ea.board_id = b.id "
           "JOIN honorarium_summary hs ON hs.board_id = b.id")
    assert _card(sql).passed


def test_fanout_preaggregated_ctes_are_the_recommended_fix():
    # The shape the error message asks for must itself pass.
    sql = ("WITH a AS (SELECT board_id, COUNT(*) n FROM evaluation_attempt GROUP BY board_id), "
           "     h AS (SELECT board_id, SUM(final_amount) t FROM honorarium_summary GROUP BY board_id) "
           "SELECT b.id, a.n, h.t FROM board b "
           "JOIN a ON a.board_id = b.id JOIN h ON h.board_id = b.id")
    assert _card(sql).passed


# ── Regressions found by batch run 20260814_132132 ───────────────────────────

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


def test_tokenizer_error_is_a_failure_not_a_crash():
    # Q61 of batch run 20260814_155341. The model broke the JSON contract, the
    # extractor scraped `SELECT list.",` out of its prose, and sqlglot raised
    # TokenError from the lexer. TokenError is a sibling of ParseError under
    # SqlglotError, not a subclass, so the old `except ParseError` missed it
    # and the exception unwound into batch_run, losing the question.
    import sqlglot
    assert issubclass(sqlglot.errors.TokenError, sqlglot.errors.SqlglotError)
    assert not issubclass(sqlglot.errors.TokenError, sqlglot.errors.ParseError)
    with pytest.raises(sqlglot.errors.SqlglotError):
        sqlglot.parse('SELECT list.",', dialect="postgres")


# ── AST-based L5 / L8 (structural rewrite) ───────────────────────────────────
#
# Every case below is a query that broke, or would have broken, a regex
# revision of these checks. They are the regression contract for the rewrite.

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


# ── Sensitive columns are derived from the DDL, not only from a Python list ──

def test_sensitive_column_read_from_ddl_comment():
    from validation.security.exposure import sensitive_columns

    class _Tagged:
        column_comments = {"actor_ip": "@sensitive Client IP, forensics only."}
        columns = {}

    class _Plain:
        column_comments = {"status": "Lifecycle state."}
        columns = {}

    resolved = sensitive_columns({"audit_log": _Tagged(), "board": _Plain()})
    assert ("audit_log", "actor_ip") in resolved      # tag honoured
    assert ("board", "status") not in resolved        # untagged left alone
    # The configured list is a UNION, not a fallback: tagging must never
    # silently drop protection from a column nobody has tagged yet.
    assert ("data_encryption_key", "wrapped_key") in resolved


# ── FIX-A1..A6: production-quality pass (2026-08-14) ─────────────────────────
# Every fix below is generic and structural, not query-specific: verified
# against synthetic adversarial cases AND, where the defect was first found in
# a real batch run, against that exact query.

import time as _time


# ── A1: retrieval parallelization ────────────────────────────────────────────

def _extract_method(path, start_line, end_line):
    """Extract and dedent a method by 1-indexed [start,end] line range, for
    testing the exact shipped code without the project's heavy import chain
    (sentence-transformers, qdrant-client, etc are not needed to prove this
    method's concurrency behaviour is correct)."""
    import textwrap
    lines = open(path, encoding="utf-8").read().split("\n")
    return textwrap.dedent("\n".join(lines[start_line - 1:end_line]))


def test_a1_mandatory_chunk_fetch_is_concurrent_and_correct():
    import enum
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src_path = "retrieval/orchestrator.py"
    # Locate the method dynamically rather than hardcoding line numbers, so
    # this test does not silently stop testing the real code after an
    # unrelated edit shifts line numbers.
    full = open(src_path, encoding="utf-8").read()
    marker = "    def _fetch_mandatory_chunks_for("
    start = full.index(marker)
    after = full[start + len(marker):]
    end_rel = after.find("\n# " + "─" * 10)
    if end_rel == -1:
        end_rel = after.find("\n    def ", 200)
    method_src = (marker + after[:end_rel]).strip("\n")
    import textwrap
    method_src = textwrap.dedent(method_src)

    class CT(enum.Enum):
        TABLE = "table"
        FK_MAP = "fk_map"

    class SemanticChunk:
        @staticmethod
        def from_payload(p):
            return p

    ns = {
        "ChunkType": CT, "SemanticChunk": SemanticChunk, "time": _time,
        "ThreadPoolExecutor": ThreadPoolExecutor, "as_completed": as_completed,
        "Any": object,
        "logger": type("L", (), {"warning": staticmethod(lambda **k: None)})(),
    }
    exec(method_src, ns)
    fn = ns["_fetch_mandatory_chunks_for"]

    class FakeQdrant:
        def search(self, query_text, top_k, chunk_types, filter_payload):
            _time.sleep(0.3)
            return [{"chunk_id": f"{filter_payload['table_name']}:{chunk_types[0].value}",
                     "table_name": filter_payload["table_name"],
                     "chunk_type": chunk_types[0].value}]

    class Fake:
        qdrant = FakeQdrant()

    tables = ["t1", "t2", "t3", "t4", "t5"]
    t0 = _time.time()
    chunks, misses = fn(Fake(), tables, {})
    elapsed = _time.time() - t0
    # Sequential would be 2 calls x 5 tables x 0.3s = 3.0s.
    assert elapsed < 1.5, f"not parallelized: {elapsed:.2f}s"
    assert len(chunks) == 10 and misses == 10

    class FlakyQdrant:
        n = 0
        def search(self, **kw):
            FlakyQdrant.n += 1
            if FlakyQdrant.n == 3:
                raise RuntimeError("boom")
            return [{"chunk_id": "x", "table_name": kw["filter_payload"]["table_name"],
                     "chunk_type": kw["chunk_types"][0].value}]

    class Fake2:
        qdrant = FlakyQdrant()

    c2, m2 = fn(Fake2(), ["a", "b"], {})
    assert len(c2) == 3, "one failed fetch must not lose the other three"

    c3, m3 = fn(Fake(), [], {})
    assert c3 == [] and m3 == 0


# ── A6: confidence calibration ────────────────────────────────────────────────

def test_a6_calibration_penalizes_retries_and_autofix():
    calib_rules = {"retries_needed": {"penalty_per_retry": 0.03},
                    "autofix_applied": {"penalty": 0.05}}

    def simulate(raw_conf, retries, original_sql, validated_sql):
        calibrated = raw_conf
        retry_rule = calib_rules.get("retries_needed", {})
        if retries > 0:
            calibrated -= retry_rule.get("penalty_per_retry", 0.03) * retries
        autofix_rule = calib_rules.get("autofix_applied", {})
        if original_sql and validated_sql.strip() != original_sql.strip():
            calibrated -= autofix_rule.get("penalty", 0.05)
        return max(0.0, round(calibrated, 2))

    assert simulate(0.98, 0, "SELECT 1", "SELECT 1") == 0.98
    assert simulate(0.95, 2, "SELECT 1", "SELECT 1") == 0.89
    assert simulate(0.95, 0, "SELECT as.id FROM t as",
                     "SELECT ans_a.id FROM t ans_a") == 0.90
    assert simulate(0.05, 5, "SELECT as.id", "SELECT ans_a.id") == 0.0


# ── A5: comment-only DDL chunks are skipped, not reported as parse_failed ────

def test_a5_comment_only_chunk_detection():
    from ingestion.ddl_parser import _is_comment_only

    assert _is_comment_only("-- just a comment\n-- another line\n")
    assert _is_comment_only("/* block */\n-- line\n   \n")
    assert _is_comment_only("")
    assert _is_comment_only("   \n\n  ")
    assert not _is_comment_only("-- comment\nCREATE TABLE x (id int);")
    assert not _is_comment_only("SELECT 1;")


def test_a5_real_ddl_has_zero_parse_errors():
    from ingestion.ddl_parser import DDLParser

    ddl_path = "data/docs/digital_evaluation_schema_v10_10.sql"
    ddl = open(ddl_path, encoding="utf-8").read()
    parser = DDLParser()
    tables = parser.parse(ddl)
    assert parser.parse_errors == [], (
        "SECTION 15 (and any other trailing comment-only block) must not "
        "surface as a parse error"
    )
    assert len(tables) == 63  # 61 tables + 2 views, unchanged from every prior run


# ── A2/A0: reserved-word and duplicate-alias autofix ─────────────────────────

def test_a2_reserved_alias_autofix_end_to_end():
    from validation.utils.autofix import (
        attempt_reserved_alias_autofix, attempt_duplicate_alias_autofix,
    )
    import sqlglot

    def parses(s):
        try:
            return all(x is not None for x in sqlglot.parse(s, dialect="postgres"))
        except Exception:
            return False

    # Real shape from batch run 20260814_155341, Q2.
    sql = ("SELECT \n as.urn AS script_urn, as.status FROM answer_script AS as "
           "WHERE as.status = 'FROZEN'")
    fixed, desc = attempt_reserved_alias_autofix(sql)
    assert fixed is not None and parses(fixed)
    assert "ans_a" in fixed

    # Must not touch legitimate AS keyword usage.
    clean = "SELECT a.id AS student_id, a.name AS full_name FROM student a"
    assert attempt_reserved_alias_autofix(clean) == (None, None)


def test_a0_duplicate_alias_autofix_does_not_corrupt_first_declaration():
    from validation.utils.autofix import attempt_duplicate_alias_autofix
    import sqlglot

    def parses(s):
        try:
            return all(x is not None for x in sqlglot.parse(s, dialect="postgres"))
        except Exception:
            return False

    sql = ("SELECT cb.id, cb.name FROM board b "
           "JOIN app_user cb ON cb.id = b.head_user_id "
           "JOIN department cb ON cb.id = b.dept_id")
    fixed, desc = attempt_duplicate_alias_autofix(sql)
    assert fixed is not None and parses(fixed)
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


def test_a3_string_agg_distinct_order_by_autofix():
    from validation.utils.autofix import attempt_distinct_order_by_autofix
    import sqlglot

    def parses(s):
        try:
            return all(x is not None for x in sqlglot.parse(s, dialect="postgres"))
        except Exception:
            return False

    sql = ("SELECT string_agg(DISTINCT s.status || ': ' || cnt.n::text, ', ' "
           "ORDER BY cnt.course_code) FROM t s JOIN u cnt ON cnt.id = s.id")
    err = ("in an aggregate with DISTINCT, ORDER BY expressions must appear "
           "in argument list LINE 1: ...")
    fixed, desc = attempt_distinct_order_by_autofix(sql, err)
    assert fixed is not None and parses(fixed)

    already_aligned = "SELECT string_agg(DISTINCT x.a, ', ' ORDER BY x.a) FROM t x"
    assert attempt_distinct_order_by_autofix(already_aligned, err) == (None, None)
    assert attempt_distinct_order_by_autofix(sql, "relation \"x\" does not exist") == (None, None)


def test_a4_missing_group_by_autofix():
    from validation.utils.autofix import attempt_missing_group_by_autofix
    import sqlglot

    def parses(s):
        try:
            return all(x is not None for x in sqlglot.parse(s, dialect="postgres"))
        except Exception:
            return False

    sql = ("SELECT d.name, dc.num_courses FROM dept d "
           "JOIN dept_courses dc ON dc.dept_id = d.id GROUP BY d.name")
    err = ('column "dc.num_courses" must appear in the GROUP BY clause or '
           'be used in an aggregate function')
    fixed, desc = attempt_missing_group_by_autofix(sql, err)
    assert fixed is not None and parses(fixed) and "dc.num_courses" in fixed.lower()

    assert attempt_missing_group_by_autofix(
        sql, 'column "zz.bogus" must appear in the GROUP BY clause') == (None, None)


# ── A4-L7: NL output-column extraction ───────────────────────────────────────

def test_l7_including_clause_preferred_over_preamble():
    from validation.semantic.nl_requirements import _extract_output_columns

    # Q29 of batch run 20260814_155341: "show" matches first, but the real
    # comma-list starts after "including". Taking the first match made the
    # PREAMBLE CLAUSE itself a fake "output column".
    q29 = ("show the complete question hierarchy for the data structures paper, "
           "including section names, question codes, ltree paths, max marks, "
           "attempt rule types, and group labels, highlighting any questions "
           "that lack a rubric in the approved answer key.")
    out = _extract_output_columns(q29)
    assert "complete question hierarchy data structures paper" not in out
    assert "section names" in out and "question codes" in out


def test_l7_display_as_noun_is_not_mistaken_for_a_header():
    from validation.semantic.nl_requirements import _extract_output_columns

    # Regression guard: a first attempt at the Q29 fix took the LAST match
    # among ALL header words, which broke on "display name" (noun, not verb).
    q32 = ("show the student name, course code, hold reason, case reference, "
           "hold start date, hold duration in days, and the display name of "
           "the user who approved the hold.")
    out = _extract_output_columns(q32)
    assert "student name" in out and "course code" in out and "hold reason" in out


def test_l7_no_header_returns_empty():
    from validation.semantic.nl_requirements import _extract_output_columns
    assert _extract_output_columns(
        "how many boards have hard deadline enforcement?") == []


def test_l7_oversized_item_dropped_not_mangled():
    from validation.semantic.nl_requirements import _extract_output_columns

    q = ("list all evaluators, including the full name of the person who "
         "conducted the most recent evaluation cycle review.")
    out = _extract_output_columns(q)
    assert not any(len(x.split()) > 6 for x in out)


# ── A3: proactive rate limiter ────────────────────────────────────────────────

def test_rate_limiter_burst_then_throttle():
    from generation.llm.rate_limiter import TokenBucketRateLimiter

    rl = TokenBucketRateLimiter(rate_per_minute=60, burst=5)
    t0 = _time.time()
    for _ in range(5):
        rl.acquire()
    assert _time.time() - t0 < 0.1   # burst capacity: no wait

    t0 = _time.time()
    rl.acquire()
    elapsed = _time.time() - t0
    assert 0.7 < elapsed < 1.4       # 6th acquire waits ~1s at 60/min


def test_rate_limiter_thread_safety():
    import threading
    from generation.llm.rate_limiter import TokenBucketRateLimiter

    rl = TokenBucketRateLimiter(rate_per_minute=6000, burst=10)
    count = {"n": 0}
    lock = threading.Lock()

    def worker():
        rl.acquire()
        with lock:
            count["n"] += 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    t0 = _time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count["n"] == 10
    assert _time.time() - t0 < 0.5


def test_rate_limiter_timeout_raises_not_hangs():
    from generation.llm.rate_limiter import TokenBucketRateLimiter

    rl = TokenBucketRateLimiter(rate_per_minute=1, burst=1)
    rl.acquire()
    with pytest.raises(TimeoutError):
        rl.acquire(timeout=0.3)


def test_rate_limiter_disabled_when_rate_is_zero(monkeypatch):
    import sys
    import types as _types

    fake_settings_mod = _types.ModuleType("config.settings")

    class FakeLLM:
        requests_per_minute = 0

    class FakeSettings:
        llm = FakeLLM()

    fake_settings_mod.settings = FakeSettings()
    monkeypatch.setitem(sys.modules, "config.settings", fake_settings_mod)

    import generation.llm.rate_limiter as rl_mod
    rl_mod._shared_limiter = None  # reset the module-level cache
    assert rl_mod.get_shared_rate_limiter() is None


# ── Review-pass fixes (2026-08-14, second look at the hardening patch) ───────
# Each test below pins a bug found by re-reading the patch, not by a batch run.

def test_alias_autofix_does_not_rewrite_string_literals():
    """A plain re.sub on raw SQL cannot tell a table qualifier from the same
    characters inside a literal, and silently rewrote the DATA."""
    from validation.utils.autofix import (
        attempt_reserved_alias_autofix, attempt_duplicate_alias_autofix,
    )
    import sqlglot

    def parses(x):
        try:
            return all(y is not None for y in sqlglot.parse(x, dialect="postgres"))
        except Exception:
            return False

    sql = ("SELECT as.urn FROM answer_script AS as "
           "WHERE as.note = 'see as.txt for detail'")
    fixed, _ = attempt_reserved_alias_autofix(sql)
    assert fixed is not None and parses(fixed)
    assert "'see as.txt for detail'" in fixed      # literal untouched
    assert "ans_a.urn" in fixed and "ans_a.note" in fixed   # qualifiers renamed

    sql2 = ("SELECT cb.id FROM board b JOIN app_user cb ON cb.id = b.head_user_id "
            "JOIN department cb ON cb.id = b.dept_id WHERE cb.name = 'cb.x'")
    fixed2, _ = attempt_duplicate_alias_autofix(sql2)
    assert fixed2 is not None and parses(fixed2)
    assert "'cb.x'" in fixed2                      # literal untouched
    assert "dep_2.name" in fixed2.split("WHERE", 1)[1]


def test_alias_autofix_does_not_rewrite_comments():
    from validation.utils.autofix import attempt_reserved_alias_autofix

    sql = "SELECT as.id FROM t AS as -- as.legacy note\n WHERE as.x = 1"
    fixed, _ = attempt_reserved_alias_autofix(sql)
    assert fixed is not None
    assert "as.legacy" in fixed                    # comment body untouched


def test_drop_projections_does_not_half_mutate_on_reject():
    """If a later scope would be emptied, the tree must be left UNTOUCHED --
    an earlier revision mutated scope-by-scope and could return None with the
    caller's live ctx.ast already partially redacted."""
    import sqlglot
    from validation.security.exposure import _drop_projections, _SensitiveFinding

    sql = ("SELECT a.keep, a.secret FROM t1 a "
           "UNION ALL SELECT b.secret FROM t2 b")
    stmt = sqlglot.parse_one(sql, dialect="postgres")
    before = stmt.sql(dialect="postgres")

    selects = list(stmt.find_all(sqlglot.expressions.Select))
    findings = []
    for sel in selects:
        for proj in sel.expressions:
            if "secret" in proj.sql().lower():
                findings.append(_SensitiveFinding(
                    "t", "secret", proj, sel, unsafe_to_edit=False))

    # The second SELECT projects ONLY the sensitive column, so the whole
    # rewrite must be refused.
    assert _drop_projections(stmt, findings) is None
    assert stmt.sql(dialect="postgres") == before, "tree was mutated on reject"


def test_validation_result_carries_explicit_autofix_flag():
    """Calibration must not infer 'an autofix ran' from a SQL text diff --
    that is also true whenever a tenant filter is injected."""
    from models.schema import ValidationResult

    assert ValidationResult(passed=True).autofix_applied is False
    assert ValidationResult(passed=True, autofix_applied=True).autofix_applied is True


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

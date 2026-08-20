"""
tests/test_semantic_checks.py
─────────────────────────────────
validation/semantic/semantic_checks.py — Step 7 (`SemanticValidator`, twelve
schema-driven checks including 18b: unprompted enum filter) and Step 8
(`HardcodedLiteralValidator`: a hardcoded literal ID/name not grounded in the
question). Both classes live in the same source file and are merged here for
the same reason.

CONSOLIDATED FROM: test_phase1_changes.py (SemanticValidator / 18b, "Change
4") and test_run7_hardening.py (HardcodedLiteralValidator, including the
`import re` shadow regression and its `_literal_is_grounded` helper).
"""

from __future__ import annotations

from models.schema import ColumnInfo, TableInventory
from validation.semantic.semantic_checks import (
    HardcodedLiteralValidator,
    SemanticValidator,
    _literal_is_grounded,
)

from conftest import make_ctx


# ═════════════════════════════════════════════════════════════════════════════
# SemanticValidator — Check 18b: schema-driven defensive-filter detection
#
# Runs inside Step 7. Its event, semantic_unprompted_enum_filter, is
# deliberately absent from _ADVISORY_SEMANTIC_EVENTS, so it still hard-fails
# rather than warning.
# ═════════════════════════════════════════════════════════════════════════════

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
    "answer_script": _mk_tbl("answer_script", [
        ("id", None), ("urn", None),
        ("lifecycle_status", ["ADMITTED", "ELIGIBLE", "ATTEMPTED", "ABSENT"]),
    ]),
}


class TestSchemaDrivenDefensiveFilter:

    validator = SemanticValidator()

    def _ctx(self, sql, nl, schema_map=None):
        return make_ctx(sql, schema_map if schema_map is not None else SAMPLE_SCHEMA,
                         original_query=nl)

    @staticmethod
    def _is_18b(message: str | None) -> bool:
        return "defensive filter not supported" in (message or "")

    def test_q67_pattern_status_in_join_on(self):
        """
        Q67: ak.status='APPROVED' in a JOIN ON, though the NL never says
        'approved'.

        `status` is one of the four columns the older keyword-driven Check 18
        owns, and Check 18 was demoted to advisory (semantic_unprompted_filter
        is in _ADVISORY_SEMANTIC_EVENTS) because it fired on correct queries.
        18b defers to Check 18 in the WHERE clause precisely so the demotion
        is not quietly undone -- a JOIN ON predicate is the case 18b genuinely
        owns, since Check 18 only walks WHERE.
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
        result = self.validator.run(self._ctx(sql, nl))
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
        result = self.validator.run(self._ctx(sql, nl))
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
        result = self.validator.run(self._ctx(sql, nl))
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
        result = self.validator.run(self._ctx(sql, nl, schema))
        assert not self._is_18b(result.message)

    def test_value_named_verbatim_in_nl_is_justified(self):
        """The plainest justification: the question names the value outright."""
        sql = "SELECT ar.id FROM attempt_rule ar WHERE ar.rule_type = 'PICK_N'"
        nl = "Show all attempt rules of type PICK_N"
        result = self.validator.run(self._ctx(sql, nl))
        assert not self._is_18b(result.message)


# ═════════════════════════════════════════════════════════════════════════════
# HardcodedLiteralValidator — Step 8
# ═════════════════════════════════════════════════════════════════════════════

def _literal_table(name: str, columns: dict[str, bool]) -> TableInventory:
    inv = TableInventory(table_name=name)
    for col in columns:
        inv.columns[col] = ColumnInfo(name=col, data_type="varchar")
    return inv


def _literal_ctx(sql: str, question: str):
    """A context whose schema_map carries a free-text and a vocabulary column."""
    app_user = _literal_table("app_user", {"id": True, "display_name": False})
    wst = _literal_table("workflow_state_transition", {"id": True, "entity_type": False})
    return make_ctx(
        sql, {"app_user": app_user, "workflow_state_transition": wst},
        original_query=question, working_sql=sql,
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
    sql = "SELECT au.id FROM app_user au WHERE au.display_name = 'COE Office'"
    result = HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Which users have roles granted by the Custodian Admin?")
    )
    assert not result.passed, "validator silently no-oped on a non-aggregate query"
    assert "COE Office" in result.message


def test_grounded_literal_is_accepted():
    sql = "SELECT au.id FROM app_user au WHERE au.display_name = 'COE Office'"
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Show all bulk operations initiated by the COE Office.")
    ).passed


def test_inflected_literal_is_accepted():
    """'DEK_REWRAP' is grounded by \"the DEK was re-wrapped\"."""
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
    sql = "SELECT wst.id FROM workflow_state_transition wst WHERE wst.id = 1"
    assert HardcodedLiteralValidator().run(
        _literal_ctx(sql, "Which questions are grouped under the Choice group "
                          "for attempt rule 1?")
    ).passed


# ═════════════════════════════════════════════════════════════════════════════
# Check 8: "average duration" is TYPE-AWARE
#
# Q17 of run 20260819: "What is the average duration of a legal hold in days?"
# over script_hold.hold_start_date / hold_end_date, both DATE.
#
# Check 8 demanded EXTRACT(EPOCH FROM (end - start)) / 86400 unconditionally.
# DateArithmeticValidator rejects exactly that over DATE operands, because
# DATE - DATE yields an INTEGER and EXTRACT has no integer signature. No SQL
# could satisfy both checks, so the question was unanswerable by construction
# and burned its full retry budget on a contradiction rather than a defect.
#
# These tests pin BOTH directions: the DATE form must now be accepted, and the
# TIMESTAMP form must still be required. A fix that simply deleted Check 8
# would pass the first of these and fail the second.
# ═════════════════════════════════════════════════════════════════════════════

def _duration_schema():
    from models.schema import ColumnInfo, TableInventory

    def col(name, data_type, *, pk=False):
        return ColumnInfo(name=name, data_type=data_type, nullable=True, is_pk=pk)

    def table(name, cols):
        return TableInventory(table_name=name, columns={c.name: c for c in cols})

    return {
        "script_hold": table("script_hold", [
            col("id", "bigint", pk=True),
            col("hold_start_date", "date"),
            col("hold_end_date", "date"),
        ]),
        "evaluation_attempt": table("evaluation_attempt", [
            col("id", "bigint", pk=True),
            col("started_at", "timestamptz"),
            col("frozen_at", "timestamptz"),
        ]),
    }


_DATE_NL = "What is the average duration of a legal hold in days?"
_TS_NL = "What is the average time between attempt assignment and freeze?"


def test_date_subtraction_is_accepted_for_average_duration():
    """The Q17 answer. Plain DATE subtraction is already a day count."""
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = ("SELECT AVG(COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date) "
           "AS avg_days FROM script_hold sh")
    assert avg_duration_epoch_error(_DATE_NL, sql, _duration_schema()) is None


def test_timestamp_subtraction_without_epoch_is_still_rejected():
    """The rule's original purpose must survive the fix."""
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = "SELECT AVG(ea.frozen_at - ea.started_at) FROM evaluation_attempt ea"
    msg = avg_duration_epoch_error(_TS_NL, sql, _duration_schema())
    assert msg is not None
    assert "EXTRACT(EPOCH" in msg


def test_timestamp_subtraction_with_epoch_passes():
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = ("SELECT AVG(EXTRACT(EPOCH FROM (ea.frozen_at - ea.started_at)) / 3600) "
           "FROM evaluation_attempt ea")
    assert avg_duration_epoch_error(_TS_NL, sql, _duration_schema()) is None


def test_message_explains_both_type_cases():
    """
    The retry loop only recovers from an error it can act on. A message naming
    only the TIMESTAMP form is what drove the model into the type error.
    """
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = "SELECT AVG(ea.frozen_at - ea.started_at) FROM evaluation_attempt ea"
    msg = avg_duration_epoch_error(_TS_NL, sql, _duration_schema())
    assert "DATE" in msg and "TIMESTAMP" in msg


def test_two_argument_call_keeps_legacy_behaviour():
    """
    fine_tuning/preprocess/quality.py calls this with two arguments and has no
    schema to type against. Without a schema_map the conservative original
    behaviour is correct and must be preserved.
    """
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = ("SELECT AVG(COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date) "
           "FROM script_hold sh")
    assert avg_duration_epoch_error(_DATE_NL, sql) is not None


def test_unrelated_question_is_untouched():
    from validation.semantic.semantic_checks import avg_duration_epoch_error
    sql = "SELECT COUNT(*) FROM script_hold sh"
    assert avg_duration_epoch_error("How many holds are active?", sql,
                                    _duration_schema()) is None

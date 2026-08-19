"""
tests/test_ast_cardinality.py
───────────────────────────────
validation/ast/cardinality.py::CardinalityValidator — rejects an aggregate
computed over the fan-out of two sibling to-many joins hanging off the same
parent row, while leaving a genuine chain (one row per leaf) alone.

CONSOLIDATED FROM: test_security_hardening.py ("Item #6: aggregate over join
fan-out").
"""

from __future__ import annotations

import pytest

from validation.ast.cardinality import CardinalityValidator

from conftest import make_ctx


class _Col:
    def __init__(self, is_pk=False):
        self.is_pk = is_pk


class _Idx:
    def __init__(self, columns):
        self.columns = columns
        self.is_unique = True
        self.is_partial = False


class _Inv:
    def __init__(self, cols, pk="id", uniq=()):
        self.columns = {c: _Col(c == pk) for c in cols}
        self.indexes = [_Idx([u]) for u in uniq]


CARD_SCHEMA = {
    "board": _Inv(["id", "course_id", "status"]),
    "evaluation_attempt": _Inv(["id", "board_id", "script_id", "marks"]),
    "honorarium_summary": _Inv(["id", "board_id", "final_amount"]),
    "attempt_rule": _Inv(["id", "question_id"]),
    "attempt_rule_group": _Inv(["id", "attempt_rule_id"]),
    "attempt_rule_group_question": _Inv(["id", "group_id", "question_id"]),
}


def _card(sql):
    return CardinalityValidator().run(make_ctx(sql, CARD_SCHEMA))


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

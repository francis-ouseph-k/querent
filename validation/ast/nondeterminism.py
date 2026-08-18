"""
validation/ast/nondeterminism.py
────────────────────────────────
NondeterministicSelectionValidator — rejects SQL whose answer depends on which
row PostgreSQL happened to return first.

THE DEFECT

    LEFT JOIN evaluator_pool ep ON ep.faculty_cache_id = fc.id
      AND ep.board_id = (SELECT b.id FROM board b JOIN ... LIMIT 1)

`LIMIT 1` with no `ORDER BY` does not mean "the one board". It means "whichever
board the planner emitted first" — a value that can change between runs, after
a VACUUM, or when the plan flips from a seq scan to an index scan. The query
runs, returns rows, and is reported as a success; re-running it later returns a
different answer with no error and no warning.

The same defect wearing a different hat:

    WITH ds_paper AS (
      SELECT qp.id FROM question_paper qp
      WHERE qp.title ILIKE '%Data Structures%' LIMIT 1
    )

Here the arbitrary pick is laundered through a CTE before being joined, so the
whole report silently describes one arbitrarily chosen paper out of several.

WHY THIS ONE CAN HARD-FAIL

It is decided entirely from the AST. It does not read the question, does not
compare NL words to SQL words, and does not depend on any value in any specific
schema — it is the standard SQL fact that `LIMIT` without `ORDER BY` leaves row
selection to the planner. There is no reading of any question under which "give
me an arbitrary one of these, and a different one tomorrow" is the intent.

WHAT IT DELIBERATELY DOES NOT FLAG

  * A top-level `LIMIT` on the final result set. "Show me 200 scripts" is a
    presentation cap, and the rows it returns are all legitimate answers; the
    defect here is about a value that FEEDS a predicate or a join.
  * `EXISTS (SELECT 1 ... LIMIT 1)`. EXISTS is a boolean over the whole set —
    which row satisfies it cannot change the answer, so the LIMIT is a
    (redundant) optimisation, not a choice.
  * Any subquery that already carries an ORDER BY. `ORDER BY x DESC LIMIT 1`
    is the ordinary "latest/highest" idiom and is deterministic to the extent
    the ordering key is unique — which is the author's call, not this check's.
  * `LIMIT` inside a window/derived table that is not consumed as a value.

The rewrite the message asks for is the one a reviewer would ask for: state the
ORDER BY that decides the tie, or correlate the subquery so it yields one row
by construction.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


# Contexts in which a subquery's value is consumed as a SCALAR or as a
# membership set — i.e. WHICH row came back changes the result of the enclosing
# expression. exp.Exists is deliberately absent: see module docstring.
_VALUE_CONSUMING_PARENTS = (
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE,
    exp.In, exp.Any, exp.All,
    exp.Binary, exp.Condition,
)


def _limit_count(select_node: exp.Select) -> int | None:
    """The integer row cap on this SELECT, or None when absent/non-literal."""
    limit = select_node.args.get("limit")
    if limit is None:
        return None
    target = limit.expression if hasattr(limit, "expression") else None
    if not isinstance(target, exp.Literal) or target.is_string:
        return None
    try:
        return int(target.name)
    except (TypeError, ValueError):
        return None


def _is_ordered(select_node: exp.Select) -> bool:
    """True when this SELECT states its own row order."""
    return select_node.args.get("order") is not None


def _under_exists(node: exp.Expression) -> bool:
    """True when the node sits inside an EXISTS / NOT EXISTS test."""
    return node.find_ancestor(exp.Exists) is not None


class NondeterministicSelectionValidator(BaseValidationStep):
    """Rejects arbitrary row picks that feed a predicate, a join, or a CTE."""

    name = "NondeterministicSelectionValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue

                finding = self._check_ctes(stmt) or self._check_subqueries(stmt)
                if finding is None:
                    continue

                where, offender = finding
                logger.warning(
                    component="sql_validator",
                    event="nondeterministic_row_selection",
                    location=where,
                    predicate=offender,
                    sql_preview=sql[:120],
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        f"`{offender}` picks an arbitrary row: it uses LIMIT "
                        f"with no ORDER BY, and the value it produces feeds "
                        f"{where}. PostgreSQL does not guarantee which row "
                        f"LIMIT returns, so this query can give a different "
                        f"answer on the same data after a plan change. "
                        f"Either add an ORDER BY that states which row you "
                        f"mean (e.g. the most recent, the highest version), "
                        f"or correlate the subquery to the outer row so it "
                        f"yields exactly one match by construction and the "
                        f"LIMIT can be dropped."
                    ),
                    sql=sql,
                )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="nondeterminism_check_error",
                error=str(exc),
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    # ── CTE bodies ────────────────────────────────────────────────────────
    @staticmethod
    def _check_ctes(stmt: exp.Expression) -> tuple[str, str] | None:
        """
        A CTE capped by LIMIT with no ORDER BY. Its rows are consumed by
        whatever joins it, so an arbitrary pick propagates into the result.
        """
        for cte in stmt.find_all(exp.CTE):
            body = cte.this
            if not isinstance(body, exp.Select):
                continue
            count = _limit_count(body)
            if count is None or _is_ordered(body):
                continue
            name = cte.alias_or_name or "a CTE"
            return (
                f"the `{name}` CTE, which the rest of the statement joins to",
                f"WITH {name} AS (... LIMIT {count})",
            )
        return None

    # ── scalar / IN subqueries ────────────────────────────────────────────
    @staticmethod
    def _check_subqueries(stmt: exp.Expression) -> tuple[str, str] | None:
        for sub in stmt.find_all(exp.Subquery):
            body = sub.this
            if not isinstance(body, exp.Select):
                continue
            count = _limit_count(body)
            if count is None or _is_ordered(body):
                continue
            if _under_exists(sub):
                continue

            parent = sub.parent
            if parent is None or not isinstance(parent, _VALUE_CONSUMING_PARENTS):
                # A derived table in FROM is a row source, not a value; the
                # arbitrary-pick argument does not apply the same way and the
                # CTE rule above already covers the laundering case.
                continue

            location = "a comparison" if isinstance(parent, exp.Binary) else "a predicate"
            if isinstance(parent, exp.In):
                location = "an IN list"
            return (location, sub.sql(dialect="postgres")[:120])
        return None

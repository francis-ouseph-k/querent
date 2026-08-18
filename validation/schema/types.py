"""
validation/schema/types.py
──────────────────────────
validate_types — the type/enum sub-check of Schema validation (step 4, reports
`schema`). Runs after tables and columns are confirmed to exist.

Checks that comparisons are type-sensible (e.g. not comparing a text column to an
integer literal) and, most usefully, that a literal compared against a column with
a CHECK constraint is one of the allowed values — so `status = 'ONGOING'` fails
when the enum only permits {'OPEN','CLOSED',...}. Enum membership is derived from
the CHECK constraints captured by the DDL parser, so it stays in sync with the
schema without a hand-maintained list.
"""

import sqlglot.expressions as exp
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger
from validation.utils.blocklist import (
    classify_pg_type as _classify_pg_type,
    check_literal_type_compat as _check_literal_type_compat,
)

logger = get_logger(__name__)

def validate_types(ctx: ValidationContext) -> ValidationResult | None:
    """
    Performs best-effort column data type and CHECK constraint enum checks.
    """
    type_errors: list[str] = []
    enum_errors: list[str] = []
    # Advisory: array-column values absent from the vocabulary OBSERVED in the
    # DDL's seed rows. Seed data is evidence, not a constraint, so these never
    # hard-fail — they are surfaced as a correction hint.
    vocab_warnings: list[str] = []
    sql = ctx.working_sql or ctx.sql
    if ctx.ast is None:
        return None

    try:
        _comparison_types = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ, exp.Is)

        for stmt in ctx.ast:
            if stmt is None:
                continue

            cte_names = set()
            for cte in stmt.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

            for cmp_node in stmt.find_all(*_comparison_types):
                left, right = cmp_node.left, cmp_node.right
                pairs: list[tuple[exp.Expression, exp.Expression]] = []
                
                if isinstance(left, exp.Column):
                    pairs.append((left, right))
                if isinstance(right, exp.Column):
                    pairs.append((right, left))

                for col_node, val_node in pairs:
                    col_name = (col_node.name or "").lower()
                    tbl_part = (col_node.table or "").lower()
                    if not col_name:
                        continue

                    resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                        list(ctx.sql_tables - cte_names)[0]
                        if len(ctx.sql_tables - cte_names) == 1 else None
                    )
                    if not resolved or resolved not in ctx.schema_map:
                        continue

                    col_info = ctx.schema_map[resolved].columns.get(col_name)
                    if col_info is None:
                        continue

                    if (
                        getattr(col_info, "allowed_values", None) is not None
                        and isinstance(cmp_node, (exp.EQ, exp.NEQ))
                        and isinstance(val_node, exp.Literal)
                        and val_node.is_string
                        and val_node.this not in col_info.allowed_values
                    ):
                        enum_errors.append(
                            f"{resolved}.{col_name} = '{val_node.this}' is not a "
                            f"valid value for this column. Allowed values: "
                            f"{', '.join(sorted(col_info.allowed_values))}"
                        )
                        continue

                    family = _classify_pg_type(col_info.data_type)
                    if family is None:
                        continue

                    err = _check_literal_type_compat(family, val_node)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                        )

            for in_node in stmt.find_all(exp.In):
                col_node = in_node.this
                if not isinstance(col_node, exp.Column):
                    continue
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()
                if not col_name:
                    continue

                resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                    list(ctx.sql_tables - cte_names)[0]
                    if len(ctx.sql_tables - cte_names) == 1 else None
                )
                if not resolved or resolved not in ctx.schema_map:
                    continue

                col_info = ctx.schema_map[resolved].columns.get(col_name)
                if col_info is None:
                    continue

                # FIX-E2: `family` may be None for a type this classifier does not
                # model, but an unmodelled type can still carry a CHECK
                # constraint. Resolve it without gating on the type family so the
                # enum test is not skipped for such columns.
                family = _classify_pg_type(col_info.data_type)

                # FIX-E2. Enum membership was checked for `col = 'X'` / `col <> 'X'`
                # but not for `col IN ('X','Y')`, so an invalid value hid inside a
                # list. Observed: honorarium_summary.role_in_board IN
                # ('EVALUATOR','REVIEWER','THIRD') passed validation, though the
                # CHECK constraint permits only EVALUATOR / REVIEWER /
                # BOARD_COORDINATOR. The query returned a silently truncated
                # per-role breakdown. IN is the same membership assertion as =,
                # so it gets the same check, over every element of the list.
                #
                # A subquery IN (`col IN (SELECT ...)`) has no literal elements to
                # test; `in_node.expressions` is empty there, so it is naturally
                # skipped rather than special-cased.
                allowed = getattr(col_info, "allowed_values", None)
                for val_node in in_node.expressions:
                    if (
                        allowed is not None
                        and isinstance(val_node, exp.Literal)
                        and val_node.is_string
                        and val_node.this not in allowed
                    ):
                        enum_errors.append(
                            f"{resolved}.{col_name} IN (... '{val_node.this}' ...) "
                            f"is not a valid value for this column. Allowed "
                            f"values: {', '.join(sorted(allowed))}"
                        )
                        continue

                    if family is None:
                        continue
                    err = _check_literal_type_compat(family, val_node)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                        )

            for between_node in stmt.find_all(exp.Between):
                col_node = between_node.this
                if not isinstance(col_node, exp.Column):
                    continue
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()
                if not col_name:
                    continue

                resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                    list(ctx.sql_tables - cte_names)[0]
                    if len(ctx.sql_tables - cte_names) == 1 else None
                )
                if not resolved or resolved not in ctx.schema_map:
                    continue

                col_info = ctx.schema_map[resolved].columns.get(col_name)
                if col_info is None:
                    continue

                family = _classify_pg_type(col_info.data_type)
                if family is None:
                    continue

                for bound in (between_node.args.get("low"), between_node.args.get("high")):
                    if bound is None:
                        continue
                    err = _check_literal_type_compat(family, bound)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}) "
                            f"BETWEEN bound: {err}"
                        )

    except Exception as exc:
        logger.warning(
            component="sql_validator",
            event="type_check_error",
            error=str(exc),
            note="Type-compatibility check skipped due to AST error",
        )

    # ── Array-containment vocabulary (advisory) ───────────────────────────
    # Postgres cannot express "every element of this array is one of N values"
    # as a CHECK..IN, so array columns holding controlled vocabularies carry no
    # constraint. ddl_parser harvests their real vocabulary from the schema's
    # own seed INSERTs into ColumnInfo.observed_values; this is the consumer.
    #
    # Fires on `col @> ARRAY[...]`, `col && ARRAY[...]`, and `x = ANY(col)`.
    # WARNS rather than rejects: seed rows are a sample, so an unseen value may
    # be legitimately new. But when a query asks for roles and supplies values
    # that appear nowhere in the role vocabulary, that is worth saying — it is
    # the difference between an empty result that means "none" and one that
    # means "you asked the wrong question".
    try:
        for _stmt in (ctx.ast or []):
          if _stmt is None:
            continue
          for arr_node in _stmt.find_all(exp.Array):
              literals = [
                  e.this for e in arr_node.expressions
                  if isinstance(e, exp.Literal) and e.is_string
              ]
              if not literals:
                  continue
              # Climb to the enclosing predicate. The array is frequently
              # wrapped in a cast (`ARRAY[...]::VARCHAR[]`), so arr_node.parent
              # is a Cast rather than the comparison; looking only one level up
              # finds no column at all. Walk up until a node referencing a
              # column OUTSIDE the array itself appears.
              cols = []
              _node = arr_node.parent
              _inside = set(id(c) for c in arr_node.find_all(exp.Column))
              for _ in range(5):
                  if _node is None:
                      break
                  _cands = [c for c in _node.find_all(exp.Column)
                            if id(c) not in _inside]
                  if _cands:
                      cols = _cands
                      break
                  _node = _node.parent
              for col_node in cols:
                  tbl_part = (col_node.table or "").lower()
                  col_name = (col_node.name or "").lower()
                  resolved = ctx.alias_map.get(tbl_part) if tbl_part else None
                  if not resolved or resolved not in ctx.schema_map:
                      continue
                  col_info = ctx.schema_map[resolved].columns.get(col_name)
                  observed = getattr(col_info, "observed_values", None) if col_info else None
                  if not observed:
                      continue
                  unknown = [v for v in literals if v not in observed]
                  if unknown and len(unknown) == len(literals):
                      # EVERY supplied value is outside the vocabulary — that is a
                      # vocabulary mix-up, not a new value. A partial mismatch is
                      # left alone; adding one new role to an existing set is
                      # exactly the legitimate case seed data cannot rule out.
                      vocab_warnings.append(
                          f"{resolved}.{col_name} is matched against "
                          f"{', '.join(repr(v) for v in unknown)}, but none of "
                          f"those appear in this column's vocabulary. Values seen "
                          f"in the schema's seed data: "
                          f"{', '.join(sorted(observed))}"
                      )
    except Exception:
        pass

    if vocab_warnings and not enum_errors:
        first = vocab_warnings[0]
        logger.warning(
            component="sql_validator",
            event="array_vocabulary_mismatch",
            detail=first[:200],
        )
        return ValidationResult(
            passed=False,
            step="schema",
            message=(
                f"Vocabulary mismatch: {first}. "
                f"Tip — check whether the values belong to a different column "
                f"(a status or type column often shares similar-looking names). "
                f"If they are genuinely new values not yet present in any row, "
                f"the query may still be correct."
            ),
            sql=sql,
        )

    if enum_errors:
        first = enum_errors[0]
        return ValidationResult(
            passed=False,
            step="schema",
            message=(
                f"Invalid value: {first}. "
                f"Tip — this column is constrained to a fixed set of values "
                f"by a CHECK constraint; only the listed values can ever exist "
                f"in the data."
                + (f" (and {len(enum_errors)-1} more)" if len(enum_errors) > 1 else "")
            ),
            sql=sql,
        )

    if type_errors:
        first = type_errors[0]
        return ValidationResult(
            passed=False,
            step="schema",
            message=(
                f"Type mismatch: {first}. "
                f"Tip — integer/bigint columns need unquoted numeric literals "
                f"(e.g. board_id = 5, not board_id = '5'). "
                f"If you are filtering by a human-readable label like a course "
                f"code or name, use the .code or .name VARCHAR column instead "
                f"of the numeric .id column."
                + (f" (and {len(type_errors)-1} more)" if len(type_errors) > 1 else "")
            ),
            sql=sql,
        )

    return None
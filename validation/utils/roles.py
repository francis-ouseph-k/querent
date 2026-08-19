"""
validation/utils/roles.py
─────────────────────────
Role resolution for polymorphic parent tables.

WHY THIS EXISTS

`JoinValidator._referent_entity` resolves a column to the TABLE it identifies.
That is the right granularity for most schemas and the wrong granularity for
any schema with a polymorphic parent — a single table whose rows mean different
things depending on a discriminator column.

This schema has one: `academic_unit` holds campuses, schools, departments,
programs and courses, discriminated by `unit_type`. Consequently
`board.course_id`, `faculty_cache.department_id`, `scanner_device.campus_id`
and `exam_cache.program_id` ALL resolve to `academic_unit`, so equating any two
of them passes the table-level domain check untouched:

    LEFT JOIN active_boards  ab ON ab.course_id = dc.dept_id     -- Q43
    LEFT JOIN scripts_in_eval se ON se.course_id = dc.dept_id

Both sides are `academic_unit` ids. Both columns exist. EXPLAIN is happy. The
join matches a course id against a department id, so every aggregate hanging
off it is silently zero or garbage.

WHERE THE ROLE COMES FROM

Two sources, deliberately kept separate, because conflating them manufactures
false positives:

  1. An FK column's role is declared in its own DDL comment:

         COMMENT ON COLUMN board.course_id IS
             'FK to academic_unit where unit_type=COURSE. ...';

     The role of an FK column is a property of the COLUMN and never of the row
     it happens to be joined against.

  2. A primary-key reference's role is pinned by an in-scope equality on the
     discriminator:

         FROM academic_unit d ... WHERE d.unit_type = 'DEPARTMENT'

     Here `d.id` identifies a DEPARTMENT for the duration of that scope.

Mixing the two directions is exactly the trap. In Q43's own first CTE:

    JOIN academic_unit c ON c.parent_id = d.id AND c.unit_type = 'COURSE'

`c` is pinned to COURSE, but `c.parent_id` carries no declared role, and a
course's parent is legitimately a DEPARTMENT. Reading the alias pin onto the FK
column would reject a correct join. So rule 1 never consults alias pins and
rule 2 never applies to FK columns. When either side is unknown this module
returns None and every caller stays silent.

WHAT COUNTS AS A DISCRIMINATOR

Derived from the DDL, never hardcoded. A column D of table T is a
discriminator when it is not part of T's primary key and either

  * D is a foreign key into a catalog table whose single-column primary key is
    itself named D (`academic_unit.unit_type` -> `unit_type_config.unit_type`), or
  * D carries a CHECK (...IN...) list, i.e. ColumnInfo.allowed_values is set.

Any schema with a polymorphic parent expressed either way gets this check for
free; a schema with none is unaffected because no discriminator is found.
"""

from __future__ import annotations

import re

import sqlglot.expressions as exp

# 'FK to academic_unit where unit_type=COURSE', 'FK to academic_unit
# (unit_type = COURSE)', "... where unit_type = 'COURSE'". The value is
# required to be an upper-case identifier because that is what a controlled
# vocabulary looks like; prose following the clause is ignored.
_ROLE_IN_COMMENT = re.compile(
    r"\b(\w+)\s*=\s*'?([A-Z][A-Z0-9_]{1,})'?",
)


def discriminator_columns(inventory) -> set[str]:
    """
    Columns of `inventory` that partition its rows into kinds. See module
    docstring for the two DDL shapes that qualify.

    `schema_map` is needed to test the catalog-table shape; when it is not
    supplied only the CHECK-list shape is recognised.
    """
    out: set[str] = set()
    columns = getattr(inventory, "columns", {}) or {}
    for name, col in columns.items():
        if getattr(col, "is_pk", False):
            continue
        if getattr(col, "allowed_values", None):
            out.add(name.lower())
    return out


def discriminator_columns_for(inventory, schema_map: dict) -> set[str]:
    """discriminator_columns() plus the FK-into-same-named-catalog-PK shape."""
    out = discriminator_columns(inventory)
    for fk in getattr(inventory, "foreign_keys", []) or []:
        from_col = (fk.from_col or "").lower()
        to_table = (fk.to_table or "").lower()
        to_col = (fk.to_col or "").lower()
        if not from_col or not to_table:
            continue
        target = schema_map.get(to_table)
        if target is None:
            continue
        target_pk = {
            n.lower()
            for n, c in (getattr(target, "columns", {}) or {}).items()
            if getattr(c, "is_pk", False)
        }
        # A catalog table: one column, and the referencing column carries the
        # same name, which is what makes the value readable as a "kind".
        if target_pk == {to_col} and from_col == to_col:
            out.add(from_col)
    return out


def declared_role(inventory, column: str, target_table: str, schema_map: dict) -> str | None:
    """
    The role an FK column declares in its own DDL comment, or None.

    Only a `<discriminator>=<VALUE>` clause naming a discriminator of the FK's
    TARGET table is accepted, so an unrelated `=` in prose cannot be misread as
    a role.
    """
    comments = getattr(inventory, "column_comments", {}) or {}
    comment = comments.get(column) or ""
    if not comment:
        col_info = (getattr(inventory, "columns", {}) or {}).get(column)
        comment = getattr(col_info, "comment", "") or ""
    if not comment:
        return None

    target = schema_map.get((target_table or "").lower())
    if target is None:
        return None
    discriminators = discriminator_columns_for(target, schema_map)
    if not discriminators:
        return None

    for match in _ROLE_IN_COMMENT.finditer(comment):
        if match.group(1).lower() in discriminators:
            return match.group(2).upper()
    return None


def _equality_literal(pred: exp.Expression) -> tuple[str, str, str] | None:
    """(alias, column, literal) for `alias.col = 'LITERAL'`, either way round."""
    if not isinstance(pred, exp.EQ):
        return None
    for column_side, value_side in ((pred.left, pred.right), (pred.right, pred.left)):
        if not isinstance(column_side, exp.Column) or not column_side.table:
            continue
        if not isinstance(value_side, exp.Literal) or not value_side.is_string:
            continue
        return (
            column_side.table.lower(),
            (column_side.name or "").lower(),
            str(value_side.this),
        )
    return None


def pinned_roles(
    select_node: exp.Select,
    alias_map: dict[str, str],
    schema_map: dict,
) -> dict[str, str]:
    """
    {alias: ROLE} for every alias this scope pins to a single discriminator
    value via an equality against a string literal, in WHERE or in any ON.

    An alias pinned to two different values is dropped rather than guessed at:
    the caller must not act on an ambiguous role.
    """
    found: dict[str, set[str]] = {}

    def collect(node: exp.Expression | None) -> None:
        if node is None:
            return
        for eq in node.find_all(exp.EQ):
            hit = _equality_literal(eq)
            if hit is None:
                continue
            alias, column, value = hit
            table = alias_map.get(alias)
            inventory = schema_map.get(table) if table else None
            if inventory is None:
                continue
            if column not in discriminator_columns_for(inventory, schema_map):
                continue
            found.setdefault(alias, set()).add(value.upper())

    collect(select_node.args.get("where"))
    for join in select_node.args.get("joins") or []:
        collect(join.args.get("on"))

    return {alias: next(iter(values)) for alias, values in found.items() if len(values) == 1}

"""
validation/utils/blocklist.py
─────────────────────────────
Column blocklist and type-compatibility constants for SQL validation.

This module contains:
- PostgreSQL type family constants and classifier
- Column blocklist for commonly hallucinated columns
- Literal type-compatibility checker

Separated from sql_validator.py to keep curated data (blocklist entries)
independent from validation logic.  The blocklist grows with each batch
evaluation audit; keeping it in its own file makes diffs cleaner.
"""

from __future__ import annotations

import sqlglot.expressions as exp
from utils.heuristics import HEURISTICS

# ── Type-compatibility check ──────────────────────────────────────────────────
# Maps DDL data_type prefix → set of sqlglot literal kinds that are compatible.
# sqlglot literal kinds: exp.Literal.is_string (True=string, False=number),
# exp.Boolean, exp.Null, exp.Cast.
# Prefixes matched case-insensitively against ColumnInfo.data_type.
#
# Rules:
#   INTEGER family  → only numeric literals (not quoted strings)
#   VARCHAR/TEXT    → only string literals
#   BOOLEAN         → only TRUE/FALSE
#   DATE/TIMESTAMP  → strings only (ISO date strings are valid)
#   DECIMAL/NUMERIC → numeric literals only
#   JSONB/JSON      → strings only (JSON is passed as quoted string)
#   CHAR            → string literals only
#
# We do NOT flag:
#   - Casts: column::int = '5'::int — valid, sqlglot emits exp.Cast
#   - NULL comparisons — always valid
#   - Subqueries / column = column — no literal to check
#   - Arrays (BIGINT[]) — skip, too complex
#   - Parameters ($1) — skip
#
_INTEGER_TYPES  = {"bigserial", "bigint", "integer", "int", "int4", "int8",
                   "smallint", "serial", "serial4", "serial8", "int2"}
_NUMERIC_TYPES  = {"decimal", "numeric", "float", "float4", "float8",
                   "real", "double"}
_STRING_TYPES   = {"varchar", "text", "char", "character varying",
                   "character", "citext", "uuid", "inet", "name"}
_DATE_TYPES     = {"date", "timestamp", "timestamptz", "time", "timetz",
                   "interval"}
_BOOL_TYPES     = {"boolean", "bool"}
_SKIP_TYPES     = {"jsonb", "json", "bytea", "xml", "array", "[]"}


# ── Column blocklist — intercepts commonly hallucinated columns ───────────────
# Hoisted to module level (C-2 fix): this dict was previously defined
# inside _validate_columns() and rebuilt on every validation call. Now allocated
# once at import time.
#
# Keys: (table_name, column_name) tuples.
# Values: Corrective instruction strings fed to the LLM's self-correction loop.
COLUMN_BLOCKLIST: dict[tuple[str, str], str] = {
    (item['table'], item['column']): item['message']
    for item in HEURISTICS.get('column_blocklist', [])
}


def classify_pg_type(data_type: str) -> str | None:
    """
    Return a broad type family for a DDL data_type string, or None to skip.
    """
    dt = data_type.lower().split("(")[0].strip()  # strip precision: DECIMAL(6,2)→decimal
    if dt.endswith("[]"):
        return None   # array — skip
    if dt in _INTEGER_TYPES or dt.startswith("serial"):
        return "integer"
    if dt in _NUMERIC_TYPES or dt.startswith("decimal") or dt.startswith("numeric"):
        return "numeric"
    if dt in _STRING_TYPES or dt.startswith("varchar") or dt.startswith("char"):
        return "string"
    if dt in _DATE_TYPES or dt.startswith("timestamp") or dt.startswith("time"):
        return "date"
    if dt in _BOOL_TYPES:
        return "boolean"
    for skip in _SKIP_TYPES:
        if dt.startswith(skip):
            return None
    return None  # unknown — skip rather than false-positive


def check_literal_type_compat(
    col_family: str,
    literal: exp.Expression,
) -> str | None:
    """
    Return an error string if literal is incompatible with col_family, else None.
    Handles exp.Literal, exp.Boolean, exp.Null.

    Intentionally one-way on integer/numeric columns:
      - integer/numeric column + quoted non-numeric string → ERROR
      - string/date column   + bare numeric literal        → NOT flagged
        (PostgreSQL implicitly casts integers to text; flagging this would
         cause false positives on VARCHAR columns like exam_id.)

    Fixed:
      Issue 2: '123.0' no longer a false positive — int(float(val)) used.
      Issue 3: '+123' no longer a false positive — lstrip("+-").
      Issue 5: dead `not isinstance(literal, exp.Boolean)` guard removed.
    """
    if isinstance(literal, exp.Null):
        return None   # NULL is always compatible
    if isinstance(literal, (exp.Cast, exp.Anonymous, exp.Column,
                             exp.Subquery, exp.Placeholder)):
        return None   # cast / subquery / parameter — skip

    if isinstance(literal, exp.Boolean):
        if col_family != "boolean":
            return f"boolean literal used on {col_family} column"
        return None

    if isinstance(literal, exp.Literal):
        is_str = literal.is_string   # True = quoted string, False = number

        # ── Integer columns ───────────────────────────────────────────────
        # Allow '123', '123.0', '-123', '+123' (PostgreSQL casts these).
        # Reject 'MBA101', 'abc', etc.
        if col_family == "integer" and is_str:
            val = literal.this
            try:
                int(float(val.lstrip("+-")))   # handles '123.0', '+123', '-5'
            except ValueError:
                return (
                    f"string literal '{val}' used on integer/bigint column — "
                    f"use an unquoted integer (e.g. 123), or filter by a text "
                    f"column (e.g. .code or .name) if matching by label"
                )

        # ── Numeric / decimal columns ─────────────────────────────────────
        if col_family == "numeric" and is_str:
            val = literal.this
            try:
                float(val)
            except ValueError:
                return (
                    f"string literal '{val}' used on numeric/decimal column — "
                    f"use an unquoted number (e.g. 25.50)"
                )

        # ── Boolean columns ───────────────────────────────────────────────
        # exp.Boolean already handled above; this covers Literal('TRUE') etc.
        if col_family == "boolean" and is_str:
            val = literal.this.upper()
            if val not in ("TRUE", "FALSE", "T", "F", "1", "0", "YES", "NO"):
                return f"non-boolean string literal '{val}' on boolean column"

        # string/date columns: not flagged — see docstring.
        return None

    return None  # unknown node type — skip

import re
import sqlglot

_PG_COLUMN_NOT_EXIST_RE = re.compile(
    r'column\s+"?([\w.]+)"?\s+does not exist', re.IGNORECASE
)
_PG_HINT_DID_YOU_MEAN_RE = re.compile(
    r'Perhaps you meant to reference the column\s+"([^"]+)"', re.IGNORECASE
)
_PG_MISSING_FROM_RE = re.compile(
    r'missing FROM-clause entry for table\s+"?(\w+)"?', re.IGNORECASE
)
_PG_OPERATOR_RE = re.compile(
    r'operator does not exist:\s+(.+)', re.IGNORECASE
)
_PG_TABLE_DUP_RE = re.compile(
    r'table name\s+"?(\w+)"?\s+specified more than once', re.IGNORECASE
)
_PG_INVALID_INPUT_RE = re.compile(
    r'invalid input syntax for type (\w+):\s+"([^"]*)"', re.IGNORECASE
)

def classify_pg_error(error_msg: str) -> tuple[str, str]:
    """
    Reclassify a PostgreSQL error from EXPLAIN into a validator step label
    and an actionable correction message.
    """
    m = _PG_COLUMN_NOT_EXIST_RE.search(error_msg)
    if m:
        col = m.group(1)
        hint = _PG_HINT_DID_YOU_MEAN_RE.search(error_msg)
        if hint:
            msg = (
                f"Column '{col}' does not exist. The Postgres planner suggests "
                f"'{hint.group(1)}' — use that, or verify the column belongs to "
                f"the qualifying table in your FROM/JOIN clauses."
            )
        else:
            msg = (
                f"Column '{col}' does not exist. Verify that the column is "
                f"defined on the table referenced by its alias prefix, or that "
                f"the alias prefix matches a table declared in FROM/JOIN."
            )
        return "schema", msg

    m = _PG_MISSING_FROM_RE.search(error_msg)
    if m:
        tbl = m.group(1)
        return "schema", (
            f"Table or alias '{tbl}' is referenced (e.g. in an ON clause or "
            f"SELECT list) but is not declared in the FROM/JOIN clauses. "
            f"Add the missing table to the FROM list, or check JOIN ordering — "
            f"a JOIN's ON clause cannot reference a table that appears later "
            f"in the JOIN chain."
        )

    m = _PG_OPERATOR_RE.search(error_msg)
    if m:
        op_expr = m.group(1).strip()
        return "schema", (
            f"Type mismatch in comparison ({op_expr}). One side is a string "
            f"column (VARCHAR/TEXT) and the other is a numeric column "
            f"(BIGINT/INTEGER). Check whether you are joining an ERP VARCHAR "
            f"identifier (e.g. exam_erp_id) against a BIGINT surrogate "
            f"primary key (e.g. .id). Use the surrogate id for joins."
        )

    m = _PG_TABLE_DUP_RE.search(error_msg)
    if m:
        tbl = m.group(1)
        return "syntax", (
            f"Alias '{tbl}' is used for more than one table in the same "
            f"query. Give each occurrence a unique alias (e.g. 'au_user', "
            f"'au_dept')."
        )

    m = _PG_INVALID_INPUT_RE.search(error_msg)
    if m:
        pg_type, bad_value = m.group(1), m.group(2)
        return "syntax", (
            f"A {pg_type} column was compared against the literal '{bad_value}' "
            f"which is not a valid {pg_type}. This is often a leftover :param "
            f"placeholder — replace it with a concrete value derived from the "
            f"question, or join through a text column (e.g. .code, .name) if "
            f"the user gave a label."
        )

    return "cost", f"EXPLAIN revealed a SQL error: {error_msg}"

def outer_query_has_limit(sql: str) -> bool:
    """Return True if the outermost SELECT already has a LIMIT clause."""
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
        if stmt is None:
            return False
        outer = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
        if outer is None:
            return False
        return outer.args.get("limit") is not None
    except Exception:
        return False

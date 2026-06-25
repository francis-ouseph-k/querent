"""
validation/blocklist.py
────────────────────────
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

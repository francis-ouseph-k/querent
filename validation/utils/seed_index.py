"""
validation/utils/seed_index.py
──────────────────────────────
Whole-row index over the DDL's own seed INSERTs.

`ColumnInfo.observed_values` already records, per column, which values the seed
data contains. That is the right structure for "is this literal plausible" and
the wrong one for "do these literals go together", because a set of values per
column cannot say which values CO-OCCUR on a row.

Co-occurrence is the whole content of a configuration/catalog table. The DDL
seeds

    INSERT INTO relationship_type_config
        (relationship_type, display_name, from_unit_type, to_unit_type, cardinality)
    VALUES ('CROSS_LISTING', 'Course cross-listed in Dept', 'COURSE', 'DEPARTMENT', ...)

which states that a CROSS_LISTING edge runs COURSE -> DEPARTMENT. A query that
pins `relationship_type = 'CROSS_LISTING'` and then asserts the from-side is a
DEPARTMENT contradicts a fact the schema file itself carries. Every table and
column exists, every literal is a legitimate catalog value, the join domains
are correct and EXPLAIN is happy — nothing else in the pipeline can see it.
The query returns zero rows and reads as a true finding of "none".

Scope and honesty: this indexes DECLARED reference data — rows shipped in the
DDL alongside the CREATE TABLE statements — not application data. Seeded
catalog tables are configuration, and configuration in the DDL is as
authoritative as a CHECK constraint. Rows seeded into transactional tables are
samples and callers must not reason from their absence.
"""

from __future__ import annotations

import re

_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)\s*"
    r"(?:OVERRIDING\s+SYSTEM\s+VALUE\s*)?VALUES\s*(.+?);",
    re.IGNORECASE | re.DOTALL,
)


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep`, ignoring separators inside quotes, parens or brackets."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == "'":
                # '' is an escaped quote inside a string literal.
                if i + 1 < len(text) and text[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_quote = False
            buf.append(ch)
        elif ch == "'":
            in_quote = True
            buf.append(ch)
        elif ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _row_tuples(values_blob: str) -> list[str]:
    """The `( ... )` groups of a VALUES payload, as raw inner text."""
    rows: list[str] = []
    depth = 0
    in_quote = False
    start = -1
    i = 0
    while i < len(values_blob):
        ch = values_blob[i]
        if in_quote:
            if ch == "'":
                if i + 1 < len(values_blob) and values_blob[i + 1] == "'":
                    i += 2
                    continue
                in_quote = False
        elif ch == "'":
            in_quote = True
        elif ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                rows.append(values_blob[start:i])
                start = -1
        i += 1
    return rows


def _scalar(token: str) -> str | None:
    """
    The literal value of a VALUES element, or None when it is not a plain
    scalar (an expression, a cast, an ARRAY, a function call).
    """
    token = token.strip()
    if not token:
        return None
    if token.startswith("'"):
        end = token.rfind("'")
        if end <= 0:
            return None
        # Reject anything trailing the literal — '{"a":1}'::jsonb is not a
        # scalar this index should claim to understand.
        if token[end + 1:].strip():
            return None
        return token[1:end].replace("''", "'")
    if token.upper() in {"NULL", "TRUE", "FALSE"}:
        return token.upper()
    if re.fullmatch(r"-?\d+(\.\d+)?", token):
        return token
    return None


def build_seed_index(seed_statements: list[str] | None) -> dict[str, list[dict[str, str]]]:
    """
    {table_name: [ {column: value, ...}, ... ]} for every seed INSERT that
    supplies an explicit column list.

    A positional INSERT is skipped rather than resolved against column order:
    a mis-aligned row would be worse than no row at all. Non-scalar elements
    are omitted from their row, so a row is a partial but never a wrong record.
    """
    index: dict[str, list[dict[str, str]]] = {}
    for chunk in seed_statements or []:
        for match in _INSERT_RE.finditer(chunk):
            table = match.group(1).lower()
            columns = [c.strip().lower() for c in _split_top_level(match.group(2))]
            if not columns:
                continue
            for raw_row in _row_tuples(match.group(3)):
                values = _split_top_level(raw_row)
                if len(values) != len(columns):
                    continue
                row: dict[str, str] = {}
                for column, token in zip(columns, values):
                    scalar = _scalar(token)
                    if scalar is not None:
                        row[column] = scalar
                if row:
                    index.setdefault(table, []).append(row)
    return index

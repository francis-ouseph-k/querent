"""
ingestion/ddl_parser.py
───────────────────────
Parses a PostgreSQL DDL file into structured TableInventory objects.

WHY THIS WAS REWRITTEN (review finding #4):
  The original regex-based implementation broke on the real DDL in five ways:
  nested parentheses, DECIMAL(6,2) commas, ON DELETE clause leaking into FK
  to_col, functional indexes with inner parentheses, and missing dollar-quote
  handling.  All five are resolved by using sqlglot's DDL AST.

ARCHITECTURE:

  Pass 0 — preprocess: character-level state machine strips top-level DML
            (INSERT/UPDATE/DELETE/COPY), yielding isolated structural chunks.
            Seed vocabulary (INSERT values) is preserved in seed_statements.
  Pass 1 — parse each structural chunk individually with ErrorLevel.RAISE so
            parse failures are isolated and logged, not silently discarded.
  Pass 2 — CREATE TABLE / CREATE VIEW / MATERIALIZED VIEW  (sqlglot AST)
  Pass 3 — ALTER TABLE ADD CONSTRAINT FK  (sqlglot AST)
  Pass 4 — CREATE INDEX  (sqlglot AST)
  Pass 5 — COMMENT ON TABLE / COLUMN  (targeted regex on clean DDL only)

KNOWN RESOLVED BUGS:

  BUG-A (OVERRIDING SYSTEM VALUE):
    sqlglot does not parse INSERT ... OVERRIDING SYSTEM VALUE VALUES, emitting
    ~900 stderr warnings per ingest.  Top-level DML is now stripped before
    sqlglot sees any input.  Function bodies inside $$...$$ are preserved.

  BUG-B (adjacent string literal concatenation in COMMENT ON):
    COMMENT ON COLUMN t.c IS 'part1 ' 'part2 '; — the original regex matched
    only the first fragment.  The new regex captures all adjacent fragments
    and _decode_pg_string() joins and unescapes them.

SQLGLOT VERSION NOTES (v20 → v26+):

  v26 renamed the partition property node:
    PartitionedByProperty  →  PartitionedOfProperty  (PARTITION OF parent)
  Both names are checked in _process_create_table() for forward compatibility.

  fk_node.expressions contain Identifier nodes in both v20 and v26.
  _process_alter_table_fk() uses _node_name() which handles both
  Identifier and Column nodes safely.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import sqlglot
import sqlglot.expressions as exp

from models.schema import ColumnInfo, ForeignKey, IndexInfo, TableInventory
from utils.logging_config import get_logger

import io as _io
import logging as _logging
import sys as _sys

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 0 — DDL preprocessor
# ─────────────────────────────────────────────────────────────────────────────


def _split_ddl(ddl_text: str) -> tuple[list[str], list[str]]:
    """
    Split DDL text into structural chunks and DML chunks using a
    character-level state machine.

    Returns:
        struct_chunks — DDL statements (CREATE, ALTER, COMMENT, etc.)
        dml_chunks    — seed-data DML (INSERT, UPDATE, DELETE, COPY)
                        preserved for vocabulary extraction by the caller

    State tracked (no nesting counters needed — all delimiters are paired):
        in_dollar_quote  — inside $tag$ ... $tag$
        in_single_quote  — inside '...'
        in_line_comment  — after -- until end of line
        in_block_comment — inside /* ... */

    Semicolons at the top level (outside all delimiters) end a chunk.
    Stripped DML chunks are replaced with equal-length whitespace in
    struct_chunks so line numbers in subsequent error messages stay valid.
    """
    in_dollar_quote = False
    dollar_tag = ""
    in_single_quote = False
    in_line_comment = False
    in_block_comment = False

    chunks: list[str] = []
    current: list[str] = []
    i = 0
    n = len(ddl_text)

    while i < n:
        ch = ddl_text[i]

        # ── Comment transitions ───────────────────────────────────────
        if in_line_comment:
            current.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            current.append(ch)
            if ch == "*" and i + 1 < n and ddl_text[i + 1] == "/":
                current.append("/")
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue

        # ── Dollar-quote transitions ──────────────────────────────────
        if in_dollar_quote:
            current.append(ch)
            if ch == "$" and ddl_text[i : i + len(dollar_tag)] == dollar_tag:
                for extra in ddl_text[i + 1 : i + len(dollar_tag)]:
                    current.append(extra)
                i += len(dollar_tag)
                in_dollar_quote = False
                dollar_tag = ""
            else:
                i += 1
            continue

        # ── Single-quote transitions ──────────────────────────────────
        if in_single_quote:
            current.append(ch)
            if ch == "'" and i + 1 < n and ddl_text[i + 1] == "'":
                current.append("'")
                i += 2
            elif ch == "'":
                in_single_quote = False
                i += 1
            else:
                i += 1
            continue

        # ── Top-level: detect delimiter starts ────────────────────────
        if ch == "-" and i + 1 < n and ddl_text[i + 1] == "-":
            in_line_comment = True
            current.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and ddl_text[i + 1] == "*":
            in_block_comment = True
            current.append(ch)
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            current.append(ch)
            i += 1
            continue

        if ch == "$":
            j = i + 1
            while j < n and (ddl_text[j].isalnum() or ddl_text[j] == "_"):
                j += 1
            if j < n and ddl_text[j] == "$":
                dollar_tag = ddl_text[i : j + 1]
                in_dollar_quote = True
                for extra in dollar_tag:
                    current.append(extra)
                i = j + 1
                continue

        if ch == ";":
            current.append(ch)
            chunks.append("".join(current))
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    if current:
        chunks.append("".join(current))

    # ── Classify: structural vs DML ───────────────────────────────────
    # DML: first meaningful token (ignoring comments and whitespace) is
    # INSERT, UPDATE, DELETE, or COPY.
    _DML_START = re.compile(
        # Skip leading whitespace, line comments, AND block comments
        # before checking for the DML keyword.
        # Review fix: original regex only skipped line comments and
        # whitespace, missing /* seed */ INSERT ... patterns.
        r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/\s*)|\s)*(INSERT|UPDATE|DELETE|COPY)\b",
        re.IGNORECASE | re.DOTALL,
    )

    struct_chunks: list[str] = []
    dml_chunks: list[str] = []

    for chunk in chunks:
        if _DML_START.match(chunk):
            dml_chunks.append(chunk.strip())
            # Replace DML with whitespace to keep line numbers stable
            struct_chunks.append(re.sub(r"[^\n]", " ", chunk))
        else:
            struct_chunks.append(chunk)

    return struct_chunks, dml_chunks


def _strip_exclude_constraints(ddl_chunk: str) -> str:
    """
    Remove EXCLUDE USING <method> (...) constraints from a CREATE TABLE chunk.

    Uses a character-level state machine rather than regex so that:
      - Single-quoted strings inside the EXCLUDE body (e.g. '[)' in daterange
        range-bounds) are correctly skipped — a ')' inside a string literal
        does not prematurely terminate the constraint body match.
      - Nested function calls (e.g. daterange(..., COALESCE(...), '[)')) are
        handled correctly regardless of nesting depth.

    Only removes EXCLUDE constraints; all other table content is preserved.
    """
    result = ddl_chunk

    # Anchor: ", [optional comments/whitespace] CONSTRAINT name EXCLUDE USING method ("
    # The \s* was insufficient — some EXCLUDE constraints are preceded by a
    # -- line comment between the trailing comma and the CONSTRAINT keyword.
    _EXCL_ANCHOR = re.compile(
        r",(?:\s|--[^\n]*\n)*CONSTRAINT\s+\w+\s+EXCLUDE\s+USING\s+\w+\s*\(",
        re.IGNORECASE,
    )

    while True:
        m = _EXCL_ANCHOR.search(result)
        if not m:
            break

        excl_start = m.start()  # position of the leading ','
        paren_open = m.end() - 1  # position of the opening '('

        # Walk forward from the opening '(' to find its matching ')'.
        depth = 0
        in_sq = False
        j = paren_open
        n = len(result)

        while j < n:
            c = result[j]
            if in_sq:
                if c == "'" and j + 1 < n and result[j + 1] == "'":
                    j += 2  # escaped single-quote: skip both
                    continue
                if c == "'":
                    in_sq = False
            else:
                if c == "'":
                    in_sq = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1  # include the closing ')'
                        break
            j += 1

        # Excise the entire EXCLUDE clause
        result = result[:excl_start] + result[j:]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# COMMENT ON regex helpers (Pass 5 only)
# ─────────────────────────────────────────────────────────────────────────────
# WHY regex: sqlglot does not expose COMMENT ON as a first-class DDL node
# with typed attributes for target and text.  Regex is simpler here.
#
# WHY on clean_ddl: running the regex on the raw DDL could match COMMENT ON
# syntax inside dollar-quoted function bodies (e.g. inside a dynamic EXECUTE
# string).  After DML-stripping those bodies survive but the regex is now
# applied to the same pre-processed text that sqlglot sees.
#
# BUG-B fix: single-quoted branch uses ((?:'(?:[^']|'')*'\\s*)+) to capture
# adjacent fragments.  _decode_pg_string() joins them and unescapes ''.


def _decode_pg_string(raw: str) -> str:
    """
    Concatenate adjacent PostgreSQL single-quoted string literals and
    unescape '' → '.

    Examples:
        _decode_pg_string("'hello '")         → "hello "
        _decode_pg_string("'foo ' 'bar'")      → "foo bar"
        _decode_pg_string("'it''s fine'")      → "it's fine"
    """
    parts = re.findall(r"'((?:[^']|'')*)'", raw, re.DOTALL)
    return "".join(p.replace("''", "'") for p in parts)


# TABLE / VIEW / MATERIALIZED VIEW — group layout:
#   Branch 1 ($$):      gr.1=table  gr.2=body
#   Branch 2 (single):  gr.3=table  gr.4=raw_literals
_RE_COMMENT_TABLE = re.compile(
    r"COMMENT\s+ON\s+(?:MATERIALIZED\s+VIEW|TABLE|VIEW)\s+([\w.\"]+)\s+IS\s+"
    r"\$\$([^$]*(?:\$(?!\$)[^$]*)*)\$\$"
    r"|COMMENT\s+ON\s+(?:MATERIALIZED\s+VIEW|TABLE|VIEW)\s+([\w.\"]+)\s+IS\s+"
    r"((?:'(?:[^']|'')*'\s*)+)",
    re.DOTALL | re.IGNORECASE,
)

# COLUMN — group layout:
#   Branch 1 ($$):      gr.1=table  gr.2=col  gr.3=body
#   Branch 2 (single):  gr.4=table  gr.5=col  gr.6=raw_literals
_RE_COMMENT_COLUMN = re.compile(
    r"COMMENT\s+ON\s+COLUMN\s+([\w.\"]+)\.([\w\"]+)\s+IS\s+"
    r"\$\$([^$]*(?:\$(?!\$)[^$]*)*)\$\$"
    r"|COMMENT\s+ON\s+COLUMN\s+([\w.\"]+)\.([\w\"]+)\s+IS\s+"
    r"((?:'(?:[^']|'')*'\s*)+)",
    re.DOTALL | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Identifier helpers
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def _clean_name(name: str) -> str:
    """
    Normalise a SQL identifier: strip schema prefix, quotes, whitespace,
    lowercase.

    @lru_cache: _clean_name is called thousands of times over a large schema
    (every column, FK endpoint, index column).  Caching avoids redundant
    string operations for repeated identifiers.

    Examples:
        'public.answer_script'  →  'answer_script'
        '"evaluation_attempt"'  →  'evaluation_attempt'
        '  Board  '             →  'board'
    """
    if not name:
        return ""
    name = name.strip().strip('"').lower()
    if "." in name:
        name = name.split(".")[-1]
    return name


def _node_name(node: exp.Expression | None) -> str:
    """
    Extract a normalised string name from a sqlglot Table, Column,
    or Identifier node.

    Handles both exp.Identifier and exp.Column safely — sqlglot changed the
    node type used for FK column lists between versions (v20 used Identifier,
    later versions use Column).  Falling back to str(node) covers any future
    representation changes.
    """
    if node is None:
        return ""
    if hasattr(node, "name") and node.name:
        return _clean_name(node.name)
    return _clean_name(str(node))


# ─────────────────────────────────────────────────────────────────────────────
# Column extraction from a sqlglot Create node
# ─────────────────────────────────────────────────────────────────────────────


def _extract_columns(
    create_node: exp.Create,
    table_name: str,
) -> tuple[dict[str, ColumnInfo], list[ForeignKey]]:
    """
    Extract column definitions and FK constraints from a CREATE TABLE node.

    Returns (columns, fks).  FKs here are inline definitions only; ALTER TABLE
    FKs are collected separately in Pass 3 and deduplicated.
    """
    columns: dict[str, ColumnInfo] = {}
    fks: list[ForeignKey] = []

    schema_node = create_node.find(exp.Schema)
    if not schema_node:
        return columns, fks

    # ── Primary key columns ───────────────────────────────────────────
    pk_cols: set[str] = set()

    # Table-level PRIMARY KEY (id, name) — exp.PrimaryKey with Column children
    for pk in create_node.find_all(exp.PrimaryKey):
        for col in pk.find_all(exp.Column):
            pk_cols.add(_node_name(col))

    # Column-level: id BIGSERIAL PRIMARY KEY — modeled as
    # PrimaryKeyColumnConstraint nested inside a ColumnConstraint on the
    # ColumnDef.  find_all(exp.PrimaryKey) does NOT find these.
    for col_def in schema_node.find_all(exp.ColumnDef):
        if col_def.find(exp.PrimaryKeyColumnConstraint):
            c = _node_name(col_def.this)
            if c:
                pk_cols.add(c)

    # ── Column definitions ────────────────────────────────────────────
    for col_def in schema_node.find_all(exp.ColumnDef):
        col_name = _node_name(col_def.this)
        if not col_name:
            continue

        dtype_node = col_def.find(exp.DataType)
        col_type = dtype_node.sql("postgres").upper() if dtype_node else "UNKNOWN"

        is_nullable = not bool(col_def.find(exp.NotNullColumnConstraint))

        # Review fix: PK columns are implicitly NOT NULL in PostgreSQL even
        # when the DDL omits the explicit constraint.
        if col_name in pk_cols:
            is_nullable = False

        has_jsonb = (
            dtype_node is not None and dtype_node.this == exp.DataType.Type.JSONB
        )

        default_node = col_def.find(exp.DefaultColumnConstraint)
        default = default_node.this.sql() if default_node else ""

        # Inline CHECK(col IN (...)) — captures the column's legal enum
        # value set so the validator can catch fabricated literals (e.g.
        # user_type = 'COE_STAFF') that pass type-family checks but can
        # never exist in the data because the CHECK constraint forbids them.
        # None means "no such constraint on this column" — distinct from
        # an empty set, which would mean "constrained to nothing".
        allowed_values: set[str] | None = None
        check_node = col_def.find(exp.CheckColumnConstraint)
        if check_node:
            in_node = check_node.find(exp.In)
            if in_node:
                allowed_values = {
                    e.this for e in in_node.expressions
                    if isinstance(e, exp.Literal) and e.is_string
                }
                if not allowed_values:
                    allowed_values = None  # malformed/non-string CHECK..IN — skip rather than over-constrain

        # Inline FK: REFERENCES target(col) on the column definition
        ref_node = col_def.find(exp.Reference)
        if ref_node:
            ref_schema = ref_node.find(exp.Schema)
            if ref_schema:
                to_table_node = ref_schema.find(exp.Table)
                if to_table_node:
                    to_table = _node_name(to_table_node)
                    table_ident = to_table_node.find(exp.Identifier)
                    to_cols = [
                        i.name
                        for i in ref_schema.find_all(exp.Identifier)
                        if i is not table_ident
                    ]
                    to_col = _clean_name(to_cols[0]) if to_cols else ""
                    if to_table and to_col:
                        fks.append(
                            ForeignKey(
                                from_table=table_name,
                                from_col=col_name,
                                to_table=to_table,
                                to_col=to_col,
                            )
                        )

        columns[col_name] = ColumnInfo(
            name=col_name,
            data_type=col_type,
            nullable=is_nullable,
            is_pk=col_name in pk_cols,
            default=default,
            has_jsonb=has_jsonb,
            allowed_values=allowed_values,
        )

    # ── Table-level FK constraints ────────────────────────────────────
    for fk_node in schema_node.find_all(exp.ForeignKey):
        from_cols = [_node_name(c) for c in fk_node.find_all(exp.Column)]
        ref_node = fk_node.find(exp.Reference)
        if not ref_node:
            continue
        to_table = _node_name(ref_node.find(exp.Table))
        to_cols = [_node_name(c) for c in ref_node.find_all(exp.Column)]
        if not to_table or not from_cols or not to_cols:
            continue
        
        # Review fix: zip() silently truncates mismatched FK definitions.
        if len(from_cols) != len(to_cols):
            logger.warning(
                component="ddl_parser",
                event="fk_column_count_mismatch",
                table=table_name,
                from_cols=from_cols,
                to_cols=to_cols,
            )
            continue
        
        for from_col, to_col in zip(from_cols, to_cols):
            fks.append(
                ForeignKey(
                    from_table=table_name,
                    from_col=from_col,
                    to_table=to_table,
                    to_col=to_col,
                )
            )

    return columns, fks


# ─────────────────────────────────────────────────────────────────────────────
# Public parser
# ─────────────────────────────────────────────────────────────────────────────


class DDLParser:
    """
    Parses a PostgreSQL DDL file into a dict of { table_name: TableInventory }.

    After calling parse() or parse_file(), the following attributes are set:
        tables          — { table_name: TableInventory }
        seed_statements — raw DML strings from the DDL (INSERT/UPDATE/etc.)
                          These carry business vocabulary (enum values, lookup
                          table contents) useful for NL→SQL retrieval.
        parse_errors    — list of (chunk_preview, error_message) tuples for
                          any structural statement that sqlglot could not parse.

    Usage:
        parser = DDLParser()
        tables = parser.parse(Path("schema.sql").read_text())
        # Access preserved seed data:
        for stmt in parser.seed_statements:
            ...
    """

    def __init__(self) -> None:
        self.seed_statements: list[str] = []
        self.parse_errors: list[tuple[str, str]] = []
        self._seen_fk_keys: set[tuple[str, str, str, str]] = set()

    def parse(self, ddl_text: str) -> dict[str, TableInventory]:
        """
        Full parse. Returns { table_name: TableInventory }.
        Resets seed_statements, parse_errors, and seen-FK tracking.
        """
        self.seed_statements = []
        self.parse_errors = []
        self._seen_fk_keys = set()
        tables: dict[str, TableInventory] = {}

        # ── Pass 0: split DDL into structural and DML chunks ──────────────
        struct_chunks, dml_chunks = _split_ddl(ddl_text)

        # Preserve seed data for vocabulary/retrieval use
        self.seed_statements = dml_chunks
        if dml_chunks:
            logger.debug(
                component="ddl_parser",
                event="dml_split",
                structural=len(struct_chunks),
                dml_preserved=len(dml_chunks),
            )

        # ── Pass 1: parse each structural chunk individually ──────────────
        # Parsing per-chunk with ErrorLevel.RAISE means a failed statement
        # is logged and skipped — it does not abort ingestion of the rest.
        #
        # WARN fallback: some valid PostgreSQL constructs are unsupported by
        # sqlglot (EXCLUDE USING GIST WITH = operator).  Stripping them before
        # parsing allows the rest of the CREATE TABLE to parse correctly.
        # If WARN still returns a Command node, the statement is genuinely
        # unsupported and is logged and skipped.
        #
        # EXCLUDE USING GIST stripping uses a state machine (not regex) to
        # correctly handle the ')' inside date-range string literals such as
        # '[)' — a regex-based approach fails because it cannot distinguish a
        # ')' inside a single-quoted string from a real closing parenthesis.

        statements: list[exp.Expression] = []
        for chunk in struct_chunks:
            if not chunk.strip():
                continue

            parse_chunk = _strip_exclude_constraints(chunk)

            stmt: exp.Expression | None = None
            try:
                stmt = sqlglot.parse_one(
                    parse_chunk,
                    dialect="postgres",
                    error_level=sqlglot.ErrorLevel.RAISE,
                )
            except sqlglot.errors.SqlglotError as exc:
                preview = chunk.strip()[:120].replace("\n", " ")

                try:

                    # sqlglot emits some parser diagnostics through its logger even when
                    # ErrorLevel.WARN successfully returns a partial AST.
                    _sq_log = _logging.getLogger("sqlglot")
                    _sq_lvl = _sq_log.level

                    try:
                        _sq_log.setLevel(_logging.CRITICAL)

                        _old_stderr = _sys.stderr
                        _sys.stderr = _io.StringIO()

                        stmt = sqlglot.parse_one(
                            parse_chunk,
                            dialect="postgres",
                            error_level=sqlglot.ErrorLevel.WARN,
                        )

                    finally:
                        _sys.stderr = _old_stderr
                        _sq_log.setLevel(_sq_lvl)

                except Exception:
                    stmt = None                    

                if isinstance(stmt, exp.Command):
                    stmt = None

                if stmt is not None:
                    logger.warning(
                        component="ddl_parser",
                        event="partial_parse",
                        preview=preview,
                        error=str(exc)[:200],
                        note="partial AST recovered via WARN fallback",
                    )
                    self.parse_errors.append((preview, f"partial:{exc}"))
                else:
                    logger.warning(
                        component="ddl_parser",
                        event="parse_failed",
                        preview=preview,
                        error=str(exc)[:200],
                    )
                    self.parse_errors.append((preview, str(exc)))

            if stmt is not None:
                statements.append(stmt)

        if self.parse_errors:
            logger.warning(
                component="ddl_parser",
                event="parse_errors_summary",
                count=len(self.parse_errors),
                note="Schema objects in failed statements were not ingested",
            )

        # ── Pass 2: CREATE TABLE and CREATE VIEW ──────────────────────────
        for stmt in statements:
            if not isinstance(stmt, exp.Create):
                continue
            kind = (stmt.args.get("kind") or "").upper()
            if kind == "TABLE":
                self._process_create_table(stmt, tables)
            elif kind in ("VIEW", "MATERIALIZED VIEW"):
                self._process_create_view(stmt, tables)

        logger.info(
            component="ddl_parser",
            event="tables_parsed",
            tables=sum(1 for t in tables.values() if not t.is_view),
            views=sum(1 for t in tables.values() if t.is_view),
        )

        # ── Pass 3: ALTER TABLE ADD CONSTRAINT FK ─────────────────────────
        alter_count = 0
        for stmt in statements:
            if (
                isinstance(stmt, exp.Alter)
                and (stmt.args.get("kind") or "").upper() == "TABLE"
            ):
                alter_count += self._process_alter_table_fk(stmt, tables)

        logger.info(component="ddl_parser", event="alter_fks_parsed", count=alter_count)

        # ── Pass 4: CREATE INDEX ──────────────────────────────────────────
        index_count = 0
        for stmt in statements:
            if (
                isinstance(stmt, exp.Create)
                and (stmt.args.get("kind") or "").upper() == "INDEX"
            ):
                index_count += self._process_create_index(stmt, tables)

        logger.info(component="ddl_parser", event="indexes_parsed", count=index_count)

        # ── Pass 5: COMMENT ON TABLE / COLUMN ────────────────────────────
        # Run on clean_ddl (struct_chunks joined) so the regex never matches
        # COMMENT ON syntax inside dollar-quoted function bodies.
        clean_ddl = "".join(struct_chunks)
        comment_count = self._extract_comments(clean_ddl, tables)
        logger.info(
            component="ddl_parser", event="comments_parsed", count=comment_count
        )

        logger.info(
            component="ddl_parser", event="parse_complete", total_objects=len(tables)
        )
        return tables

    # ─────────────────────────────────────────────────────────────────
    # Pass handlers
    # ─────────────────────────────────────────────────────────────────

    def _process_create_table(
        self,
        stmt: exp.Create,
        tables: dict[str, TableInventory],
    ) -> None:
        """Extract table name, columns, and inline FKs from a CREATE TABLE."""

        # ── Partition detection (must come BEFORE schema_node check) ──────
        # Two separate concepts:
        #   PartitionedByProperty  = CREATE TABLE t (...) PARTITION BY LIST(col)
        #     → t is the PARENT that owns child partitions; NOT itself a partition
        #   PartitionedOfProperty  = CREATE TABLE child PARTITION OF parent ...
        #     → child IS a partition; parent is the referenced table
        #
        # v20–v25 used PartitionedByProperty for BOTH; v26 introduced the
        # separate PartitionedOfProperty class.  We check the correct one.
        #
        # Child partition tables have NO exp.Schema node (no column list) so
        # they must be registered BEFORE the schema_node early-exit below.

        partition_of = ""
        is_partition = False

        props_node = stmt.find(exp.Properties)
        if props_node:
            for prop in props_node.expressions:
                # PartitionedOfProperty (v26+): child PARTITION OF parent
                if isinstance(prop, getattr(exp, "PartitionedOfProperty", type(None))):
                    is_partition = True
                    # args['this'] is the parent Table node in v26
                    parent_node = prop.args.get("this") or prop.find(exp.Table)
                    if parent_node:
                        partition_of = _node_name(parent_node)
                    break

                # v20–v25 fallback: PartitionedByProperty sometimes used for
                # PARTITION OF children (not to be confused with PARTITION BY
                # parent definitions which also use this class in v26).
                # Only treat as a partition child if it contains a Table
                # reference (= parent name) in its sql() output.
                if isinstance(prop, exp.PartitionedByProperty):
                    sql_lower = prop.sql().lower()
                    if "partition of" in sql_lower or "partition_of" in sql_lower:
                        is_partition = True
                        parent_node = prop.find(exp.Table)
                        if parent_node:
                            partition_of = _node_name(parent_node)
                    break

        # ── Determine table name ──────────────────────────────────────────
        # For partition children (no Schema node), the table name is on the
        # Create node itself, not inside a Schema node.
        schema_node = stmt.find(exp.Schema)

        if schema_node:
            table_node = schema_node.find(exp.Table)
        else:
            # Partition child: find the FIRST Table node which is the child name
            all_tables = list(stmt.find_all(exp.Table))
            table_node = all_tables[0] if all_tables else None

        if not table_node:
            return

        table_name = _node_name(table_node)
        if not table_name:
            return

        if is_partition:
            logger.debug(
                component="ddl_parser",
                event="partition_detected",
                table=table_name,
                parent=partition_of,
            )

        if not schema_node:
            # Partition child with no column definitions — register with empty
            # columns but correct partition metadata so the graph builder can
            # link it to its parent.
            tables[table_name] = TableInventory(
                table_name=table_name,
                columns={},
                foreign_keys=[],
                is_partition=is_partition,
                partition_of=partition_of,
            )
            return

        columns, fks = _extract_columns(stmt, table_name)
        unique_fks = self._dedup_fks(fks)

        tables[table_name] = TableInventory(
            table_name=table_name,
            columns=columns,
            foreign_keys=unique_fks,
            is_partition=is_partition,
            partition_of=partition_of,
        )

    def _process_create_view(
        self,
        stmt: exp.Create,
        tables: dict[str, TableInventory],
    ) -> None:
        """Register a CREATE VIEW or CREATE MATERIALIZED VIEW."""
        view_node = stmt.find(exp.Table)
        if not view_node:
            return
        view_name = _node_name(view_node)
        if not view_name:
            return
        if view_name not in tables:
            tables[view_name] = TableInventory(table_name=view_name, is_view=True)
        else:
            tables[view_name].is_view = True

    def _process_alter_table_fk(
        self,
        stmt: exp.Alter,
        tables: dict[str, TableInventory],
    ) -> int:
        """
        Extract FK constraints from ALTER TABLE ADD CONSTRAINT ... FOREIGN KEY.

        Returns the number of new FK edges added (excluding duplicates).

        Note on from_cols extraction: sqlglot versions differ in whether
        fk_node.expressions contains Identifier or Column nodes.  Using
        _node_name() handles both by checking the .name attribute first and
        falling back to str() — no isinstance guard needed.
        """
        source_table = _node_name(stmt.find(exp.Table))
        if not source_table:
            return 0

        if source_table not in tables:
            tables[source_table] = TableInventory(table_name=source_table)

        count = 0
        
        for action in stmt.find_all(exp.AddConstraint):
            fk_node = action.find(exp.ForeignKey)
            if not fk_node:
                continue

            constraint_node = action.find(exp.Constraint)
            constraint_name = _node_name(constraint_node) if constraint_node else ""

            # from_cols: direct children of the ForeignKey node.
            # _node_name() works for both Identifier (v20) and Column (v26+).
            from_cols = [_node_name(e) for e in fk_node.expressions if _node_name(e)]

            ref_node = fk_node.find(exp.Reference)
            if not ref_node:
                continue

            ref_schema = ref_node.find(exp.Schema)
            if not ref_schema:
                continue

            to_table_node = ref_schema.find(exp.Table)
            if not to_table_node:
                continue

            to_table = _node_name(to_table_node)
            table_ident = to_table_node.find(exp.Identifier)
            to_cols = [
                i.name
                for i in ref_schema.find_all(exp.Identifier)
                if i is not table_ident
            ]

            # Review fix: zip() silently truncates mismatched FK definitions.
            if len(from_cols) != len(to_cols):
                logger.warning(
                    component="ddl_parser",
                    event="fk_column_count_mismatch",
                    table=source_table,
                    constraint=constraint_name,
                    from_cols=from_cols,
                    to_cols=to_cols,
                )
                continue        

            if not to_table or not from_cols or not to_cols:
                continue
 
            for from_col, to_col in zip(from_cols, to_cols):
                fk = ForeignKey(
                    from_table=_clean_name(source_table),
                    from_col=_clean_name(from_col),
                    to_table=_clean_name(to_table),
                    to_col=_clean_name(to_col),
                    constraint_name=constraint_name,
                )
                key = (fk.from_table, fk.from_col, fk.to_table, fk.to_col)
                if key not in self._seen_fk_keys:
                    self._seen_fk_keys.add(key)
                    tables[source_table].foreign_keys.append(fk)
                    count += 1

        return count

    def _process_create_index(
        self,
        stmt: exp.Create,
        tables: dict[str, TableInventory],
    ) -> int:
        """Extract index metadata from a CREATE [UNIQUE] INDEX node."""
        index_node = stmt.find(exp.Index)
        if not index_node:
            return 0

        index_name = _node_name(index_node.find(exp.Identifier)) or ""
        table_node = index_node.find(exp.Table)
        table_name = _node_name(table_node) if table_node else ""

        if not table_name or table_name not in tables:
            return 0

        # Index method (USING btree / gin / gist / hash).
        # params_node.args["using"] is a Var node in v26+; str() gives the name.
        method = "btree"
        params_node = index_node.args.get("params")
        if params_node and params_node.args.get("using"):
            method = str(params_node.args["using"]).lower().strip('"')

        is_unique = bool(stmt.args.get("unique"))

        # Index column / expression list.
        # Scoped to params_node only — the fallback find_all(exp.Column) on the
        # entire index node also picks up columns from the WHERE clause of a
        # partial index, polluting the column list.  If params_node is absent
        # we log a warning and skip rather than ingest incorrect data.
        cols: list[str] = []
        if params_node and params_node.args.get("columns"):
            for ordered in params_node.args["columns"]:
                col_sql = (
                    ordered.this.sql("postgres")
                    if hasattr(ordered, "this")
                    else str(ordered)
                )
                cols.append(col_sql)

        if not cols:
            logger.debug(
                component="ddl_parser",
                event="index_no_columns",
                index=index_name,
                table=table_name,
                note="params_node missing or empty — index skipped",
            )
            return 0

        condition = ""
        if params_node and params_node.args.get("where"):
            condition = params_node.args["where"].sql("postgres")

        tables[table_name].indexes.append(
            IndexInfo(
                name=index_name,
                table_name=table_name,
                columns=cols,
                is_unique=is_unique,
                is_partial=bool(condition),
                method=method,
                condition=condition,
            )
        )
        return 1

    def _extract_comments(
        self,
        clean_ddl: str,
        tables: dict[str, TableInventory],
    ) -> int:
        """
        Extract COMMENT ON TABLE / VIEW / COLUMN statements using targeted regex.

        Run on clean_ddl (DML stripped, struct_chunks joined) so the regex
        cannot false-positive against COMMENT ON syntax inside dollar-quoted
        function bodies.

        Both dollar-quoted ($$...$$) and single-quoted ('...' with optional
        adjacent fragment concatenation) comment strings are handled.
        """
        count = 0

        # Table / View / Materialized View comments
        for match in _RE_COMMENT_TABLE.finditer(clean_ddl):
            if match.group(1):  # $$ branch
                table_name = _clean_name(match.group(1))
                comment = match.group(2).strip()
            else:  # single-quote branch
                table_name = _clean_name(match.group(3))
                comment = _decode_pg_string(match.group(4))

            if table_name in tables:
                tables[table_name].comment = comment
                count += 1

        # Column comments
        for match in _RE_COMMENT_COLUMN.finditer(clean_ddl):
            if match.group(1):  # $$ branch
                table_name = _clean_name(match.group(1))
                col_name = _clean_name(match.group(2))
                comment = match.group(3).strip()
            else:  # single-quote branch
                table_name = _clean_name(match.group(4))
                col_name = _clean_name(match.group(5))
                comment = _decode_pg_string(match.group(6))

            if table_name in tables:
                tables[table_name].column_comments[col_name] = comment
                if col_name in tables[table_name].columns:
                    tables[table_name].columns[col_name].comment = comment
                count += 1

        return count

    def _dedup_fks(self, fks: list[ForeignKey]) -> list[ForeignKey]:
        """
        Return only FKs whose (from_table, from_col, to_table, to_col) key
        has not been seen in this parse session.  Deduplication is necessary
        because a FK can appear both inline in CREATE TABLE and in a subsequent
        ALTER TABLE ADD CONSTRAINT.
        """
        result: list[ForeignKey] = []
        for fk in fks:
            key = (fk.from_table, fk.from_col, fk.to_table, fk.to_col)
            if key not in self._seen_fk_keys:
                self._seen_fk_keys.add(key)
                result.append(fk)
        return result

    # ─────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────

    def parse_file(self, path: str | Path) -> dict[str, TableInventory]:
        """Read a DDL file from disk and parse it."""
        ddl_text = Path(path).read_text(encoding="utf-8")
        logger.info(
            component="ddl_parser",
            event="reading_file",
            path=str(path),
            bytes=len(ddl_text),
        )
        return self.parse(ddl_text)
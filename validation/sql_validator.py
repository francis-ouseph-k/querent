"""
validation/sql_validator.py
────────────────────────────
10-step SQL validation pipeline.

Step 1   — Syntax:    sqlglot parses in PostgreSQL 16 dialect
Step 1.1 — Parameter: detects LLM placeholders ($1, :param)
Step 1.5 — AST Rules: identifies reserved keyword alias usage
Step 2   — Schema:    all tables_used exist + column-level hallucination check
Step 2.5 — Type:      validates literal type compatibility against schema
Step 3   — Safety:    no DML (UPDATE/DELETE/TRUNCATE/DROP); no Cartesian joins
Step 4   — Security:  tenant filter injected idempotently if missing
Step 5   — Cost:      PostgreSQL EXPLAIN — reject if estimated cost > threshold
Step 7   — Semantic:  heuristic logic checks (anti-join, percentage, avg-per pattern)
Step 8   — Hardcoded: detects suspiciously hardcoded integer literal IDs

Design: each step is a separate method. The pipeline halts at the first
failure and returns the failing step name + message. The retry loop uses
this to build a targeted correction prompt.

FIXES IN THIS VERSION
─────────────────────
C2  — _step_security(): tenant filter presence check was a substring search
      ("board_id" in sql.lower()).  Any SQL that mentioned the column name —
      even in SELECT board_id or WHERE board_id = 999 — was treated as already
      filtered and skipped.  Fix: AST-walk exp.EQ nodes to verify a genuine
      equality predicate matching the context value is present before skipping
      injection.

H1  — _step_safety() Cartesian check: from_node.args.get("expressions") is
      always None in sqlglot 26+.  The check was dead code.  Fix: detect
      implicit Cartesian joins by looking for exp.Join nodes with no ON and
      no USING clause.  The regex fallback is retained for unparseable SQL.

H2  — _step_safety() Layer 2 blocked-keyword regex matched keywords inside
      string literals (e.g. WHERE action_type = 'UPDATE' was blocked).
      Fix: run the regex only when the AST parse succeeded and Layer 1 already
      cleared the statement as SELECT-only.  A statement the AST confirmed is a
      valid SELECT cannot contain top-level DML keywords; the regex is now only
      a secondary guard for genuinely unparseable input.

H3  — _inject_where(): injected predicate was unqualified ("board_id = 5").
      When multiple joined tables both have board_id, PostgreSQL raises 42702
      (ambiguous column).  Fix: resolve the alias of the first FROM/JOIN table
      that has the predicate column and qualify the predicate accordingly
      (e.g. "a.board_id = 5").

H6  — _step_schema(): only table names were validated; hallucinated column
      names passed through silently in dry-run mode (when EXPLAIN is skipped).
      Fix: walk exp.Column nodes, resolve table alias → table name, check each
      column against schema_map[table].columns.  CTE output columns and
      function-call arguments are excluded from the check to avoid
      false-positives.

M7  — _step_cost(): EXPLAIN ran on the SQL without LIMIT (LIMIT is appended
      later in _execute()).  Cost estimates for large-table scans were inflated,
      causing false rejects near the threshold.  Fix: append
      LIMIT <max_rows> to the SQL string before EXPLAIN when the outermost
      SELECT has no LIMIT clause.

FIX-V1 — Four logger calls in _step_security() and _inject_where() were
          missing component="sql_validator", inconsistent with the rest of the
          file and breaking log filtering by component=.
          Affected calls: tenant_filter_injected (×2), tenant_filter_rls,
          tenant_filter_unavailable, tenant_filter_ast_failed.

REVIEW CLARIFICATION (Step 5 gate) — see comment at the gate inside validate()
below. No behavior change; clarifies that the gate checks whether a
connection mechanism was configured, not whether it currently succeeds.

REVIEW FIX (NEW-M3) — _step_cost()'s direct psycopg2 path now wraps
self._get_conn() in a try/except, since runner.py's _get_connection() can
raise PoolTimeoutError when the connection pool is exhausted past its
acquire timeout. The cost check degrades to skip-the-check (passed=True)
on any connection failure here, consistent with how a None connection is
already treated elsewhere in this method.
"""

from __future__ import annotations
import time
from typing import Any

import psycopg2
import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp

from config.settings import settings
from mcp_tools.client import call_postgres_explain, MCPCallError
from models.schema import ValidationResult
from utils.logging_config import get_logger
from validation.blocklist import (
    COLUMN_BLOCKLIST as _COLUMN_BLOCKLIST,
    classify_pg_type as _classify_pg_type,
    check_literal_type_compat as _check_literal_type_compat,
)
from validation.semantic_checks import (
    check_semantic,
    check_hardcoded_literals,
)

logger = get_logger(__name__)

import re
# Module‑level regex constants for reserved‑keyword alias detection
RESERVED_WORDS = r'(?:as|in|group|order|table|select|where|having|limit|offset|from|join|union|with|all|and|any|case|check|exists|using)'

# Detect alias declarations in FROM/JOIN clauses, supporting optional schema and quoted identifiers.
DECL_ALIAS_RE = re.compile(
    rf'''(?x)
        \b(?:from|join)\s+
        (?:\"?[a-z_][a-z0-9_]*\"?\.)?\"?[a-z_][a-z0-9_]*\"?\s+
        (?:as\s+)?\"?(?P<alias>{RESERVED_WORDS})\"?
        (?=\s+(?:on|where|join|left|right|inner|outer|cross|full|group|order|limit|having|;|union|intersect|except)\b|$)
    '''
)

# Fallback – toxic prefixes that appear as "as.column" etc.
TOXIC_ALIAS_RE = re.compile(r'\b(?P<alias>(?:as|in|group|order|having|limit|offset|from|join))\.[a-z_][a-z0-9_]*\b')

# Sub‑query alias detection – catches "FROM (SELECT ...) AS <reserved>"
SUBQUERY_ALIAS_RE = re.compile(
    rf'''(?i)\bfrom\s*\(\s*select.*?\)\s+as\s+\"?(?P<alias>{RESERVED_WORDS})\"?\b''',
    flags=re.DOTALL,
)

# Cartesian regex — fallback only for unparseable SQL (H2 fix: not run on valid parses)
_CARTESIAN_PATTERN = re.compile(r"FROM\s+\w+\s*,\s*\w+", re.IGNORECASE)

# Parameter placeholder regex — detects :param_name and $N patterns that
# the LLM sometimes generates instead of literal values.  These cause
# "there is no parameter $1" errors at EXPLAIN time.
# Negative lookbehind (?<!:) avoids matching PostgreSQL's :: cast operator.
# Negative lookbehind (?<!\w) avoids matching inside identifiers.
_PLACEHOLDER_RE = re.compile(r"(?<!:)(?<!\w):\w+|\$\d+")

# _BLOCKED_PATTERN is owned by ValidationSettings in config/settings.py.
# Access via: settings.validation.blocked_pattern

# Type constants, column blocklist, and helper functions are in validation/blocklist.py.
# Imported above as: _COLUMN_BLOCKLIST, _classify_pg_type, _check_literal_type_compat


# ── Postgres-error → validator-step reclassifier ───────────────────────────
# Patterns matched against PostgreSQL error text returned by EXPLAIN.
# Used by _step_cost.check_pgcode() to route errors back into the retry
# loop with an actionable step label and correction hint.
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


def _classify_pg_error(error_msg: str) -> tuple[str, str]:
    """
    Reclassify a PostgreSQL error from EXPLAIN into a validator step label
    and an actionable correction message.

    Returns (step, message). The step label routes the failure through the
    retry-context builder in pipeline/runner.py — schema-step failures get
    relevant table chunks, syntax-step failures get a syntax hint, and
    plain cost failures keep the original behaviour.
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

    # Fallback — keep the original cost-step classification
    return "cost", f"EXPLAIN revealed a SQL error: {error_msg}"


def _outer_query_has_limit(sql: str) -> bool:
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


class SQLValidator:
    """
    Validates generated SQL through all 10 checkpoints before execution.

    Usage:
        validator = SQLValidator(schema_map=tables, db_conn=conn)
        result    = validator.validate(sql, tables_used=["board", "evaluation_attempt"])
    """

    def __init__(
        self,
        schema_map: dict[str, Any],   # table_name → TableInventory
        get_connection: Any = None,   # callable → psycopg2 connection | None
        release_conn:   Any = None,   # callable(conn) → None
        db_dsn:         str | None = None,   # deprecated — use get_connection instead
        fk_graph:       Any = None,   # nx.DiGraph
    ) -> None:
        self.schema_map    = schema_map
        self._get_conn     = get_connection
        self._release_conn = release_conn
        self.db_dsn        = db_dsn
        self.fk_graph      = fk_graph

        # FIX-NEW-C3: derive tenant-scoped tables dynamically from the schema map.
        self._tenant_scoped_tables: set[str] = {
            name for name, inv in schema_map.items()
            if "board_id" in inv.columns or "course_id" in inv.columns
        }
        logger.info(
            component="sql_validator",
            event="tenant_tables_derived",
            count=len(self._tenant_scoped_tables),
            tables=sorted(self._tenant_scoped_tables),
        )

    def validate(
        self,
        sql:            str,
        tables_used:    list[str]  = None,
        user_context:   dict | None = None,
        original_query: str | None = None,
    ) -> ValidationResult:
        """
        Run all validation steps in order.
        Returns immediately on first failure.

        original_query: the user's natural language question, used by Step 7
        (semantic heuristic) to detect logic mismatches between the question
        and the generated SQL. Optional — when None, Step 7 is skipped.
        """
        tables_used  = tables_used  or []
        user_context = user_context or {}

        try:
            statements = sqlglot.parse(sql, dialect="postgres")
        except sqlglot.errors.ParseError:
            statements = None

        # Step 1 — Syntax
        result = self._step_syntax(sql, statements)
        if not result.passed:
            return result

        # Step 1.1 — Parameter placeholder check
        # Catches :param_name and $N before they reach EXPLAIN and cause
        # cryptic "there is no parameter $1" errors.
        result = self._step_placeholder(sql)
        if not result.passed:
            return result

        # Step 1.5 — Aggregation Checks
        result = self._step_aggregation(sql)
        if not result.passed:
            return result

        # Step 2 — Schema grounding (tables + columns)
        result = self._step_schema(sql, tables_used, statements)
        if not result.passed:
            return result

        # Step 2.5 — Join paths
        result = self._step_joins(result.sql or sql, tables_used)
        if not result.passed:
            return result

        # Step 3 — Safety
        result = self._step_safety(sql, statements)
        if not result.passed:
            return result

        # Step 4 — Security (modifies SQL — uses result.sql for subsequent steps)
        result = self._step_security(result.sql or sql, tables_used, user_context, statements)
        secured_sql = result.sql or sql
        if not result.passed:
            return result

        # Step 5 — Cost via EXPLAIN (optional — requires DB connection)
        #
        # REVIEW CLARIFICATION: this gate checks whether a connection
        # *mechanism* was configured at all (a callable or a DSN string),
        # not whether that mechanism would currently succeed. In normal
        # operation runner.py always passes get_connection=_get_connection
        # (a real function reference, never None) — so this condition is
        # effectively always True once the validator is constructed via
        # runner.py. That is intentional, not a gap: the actual blank-
        # PG_HOST short-circuit happens one level deeper, inside
        # _step_cost() below. _get_conn() is called there, and
        # runner.py's _get_connection() returns None when settings.postgres
        # .host is blank — _step_cost() then sees conn is None and returns
        # passed=True immediately (skip the cost check, do not error or
        # hang). See _step_cost()'s "if conn is None:" check.
        #
        # Net effect: for local/dry-run testing, leaving PG_HOST blank in
        # .env is sufficient to disable EXPLAIN-based cost checking with no
        # further changes needed here — this was already the case before
        # this comment was added. db_dsn (the deprecated parameter) behaves
        # the same way only if an empty string is passed; a non-empty but
        # unreachable DSN would still attempt psycopg2.connect() and raise,
        # which is why get_connection (with its own internal blank-check)
        # is the recommended path over db_dsn.
        if self._get_conn is not None or self.db_dsn:
            result = self._step_cost(secured_sql)
            if not result.passed:
                return result

        # Step 7 — Semantic heuristic checks (optional — needs original_query)
        # Lightweight string analysis to catch logical mismatches between the
        # user's question and the generated SQL. No DB or LLM calls.
        if original_query:
            result = self._step_semantic(secured_sql, original_query)
            if not result.passed:
                return result

            # Step 8 — Hardcoded literal detection
            result = self._step_hardcoded_literals(secured_sql, original_query)
            if not result.passed:
                return result

        logger.info(
            component="sql_validator",
            event="validation_passed",
            sql_preview=sql[:80],
            tables=tables_used,
        )
        return ValidationResult(passed=True, sql=secured_sql)

    # ─────────────────────────────────────────────────────────────────────
    # Individual validation steps
    # ─────────────────────────────────────────────────────────────────────

    def _step_syntax(self, sql: str, statements: list[exp.Expression] | None = None) -> ValidationResult:
        """Step 1: Parse SQL using sqlglot in PostgreSQL dialect."""
        if not sql or not sql.strip():
            return ValidationResult(
                passed=False, step="syntax",
                message="Empty SQL generated.", sql=sql,
            )

        if statements is None:
            logger.warning(
                component="sql_validator",
                event="sql_syntax_parse_error",
                error="Parse error",
                sql_preview=sql[:80],
            )
            
            # Provide explicit feedback if the LLM used a reserved keyword as a table alias.
            # We first strip strings and comments to avoid false positives.
            # M-2 fix: use module-level regex constants instead of rebuilding patterns.
            # M-8 fix: removed redundant `import re` — already imported at module level.
            sql_lower = sql.lower()
            clean_sql = re.sub(r'--.*', '', sql_lower)
            clean_sql = re.sub(r'/\*.*?\*/', '', clean_sql, flags=re.S)
            clean_sql = re.sub(r"'.*?'", "''", clean_sql)
            
            # 1. Detect alias declarations using module-level DECL_ALIAS_RE
            bad_alias_match = DECL_ALIAS_RE.search(clean_sql)
            
            # 2. Fallback: Detect toxic prefixes using module-level TOXIC_ALIAS_RE
            if not bad_alias_match:
                bad_alias_match = TOXIC_ALIAS_RE.search(clean_sql)
                
            if bad_alias_match:
                bad_alias = bad_alias_match.group("alias")
                return ValidationResult(
                    passed=False, step="syntax",
                    message=(
                        f"SQL uses '{bad_alias}' as a table alias. "
                        f"'{bad_alias}' is a SQL reserved keyword and is likely causing parsing or validation failures. "
                        f"Please rename the alias to a non-keyword identifier (e.g., if you aliased answer_script to 'as', change it to 'a' or 'ans_scr')."
                    ),
                    sql=sql,
                )

            return ValidationResult(
                passed=False, step="syntax",
                message="SQL syntax error or unparseable query structure.", sql=sql,
            )

        return ValidationResult(passed=True, step="syntax", sql=sql)

    def _step_placeholder(self, sql: str) -> ValidationResult:
        """
        Step 1.1: Reject SQL containing parameter placeholders.

        The LLM sometimes generates parameterized queries (:qp_id, :script_id,
        $1) instead of using literal values or filtering by name/title.
        These cause cryptic errors at EXPLAIN time ("there is no parameter $1").
        Catching them here gives the retry loop a clear, actionable error message.
        """
        match = _PLACEHOLDER_RE.search(sql)
        if match:
            placeholder = match.group()
            logger.warning(
                component="sql_validator",
                event="parameter_placeholder_detected",
                placeholder=placeholder,
                sql_preview=sql[:80],
            )
            return ValidationResult(
                passed=False, step="placeholder",
                message=(
                    f"SQL contains parameter placeholder '{placeholder}'. "
                    "Never use :param or $N placeholders. Use literal values "
                    "instead. When the question mentions a specific entity by "
                    "name, filter by that name: WHERE table.column = 'value'."
                ),
                sql=sql,
            )
        return ValidationResult(passed=True, step="placeholder", sql=sql)

    def _step_aggregation(self, sql: str) -> ValidationResult:
        """Step 1.5: Reject nested aggregate functions and missing GROUP BY."""
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if not ast:
                return ValidationResult(passed=True, step="aggregation", sql=sql)
            
            # Check 1: nested aggregates (existing) — e.g. AVG(COUNT(*))
            for agg in ast.find_all(exp.AggFunc):
                for child in agg.expressions:
                    if child.find(exp.AggFunc):
                        return ValidationResult(
                            passed=False, step="aggregation",
                            message="Nested aggregate functions are not allowed in PostgreSQL (e.g., AVG(COUNT(*))). Use a subquery or adjust your GROUP BY.",
                            sql=sql
                        )

            # Check 2 (Change 7): SELECT mixes aggregate and non-aggregate
            # columns without GROUP BY.  Common LLM error: SELECT col, COUNT(*)
            # without GROUP BY col — this will fail at execution, but in dry-run
            # mode there is no PostgreSQL to catch it.
            select_node = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
            if select_node and select_node.expressions:
                expressions = select_node.expressions
                has_agg = any(
                    expr.find(exp.AggFunc) for expr in expressions
                )
                if has_agg:
                    # Check if there are non-aggregate columns
                    has_non_agg = False
                    for expr in expressions:
                        # Unwrap aliases: SELECT COUNT(*) AS c is an Alias wrapping AggFunc
                        inner = expr.this if isinstance(expr, exp.Alias) else expr
                        if inner.find(exp.AggFunc):
                            continue  # this expression is aggregate — skip
                        # exp.Column or exp.Alias wrapping a Column → non-aggregate
                        if isinstance(inner, exp.Column) or (
                            isinstance(expr, exp.Alias) and isinstance(inner, exp.Column)
                        ):
                            has_non_agg = True
                            break
                        # Catch expressions that contain a Column but not inside an AggFunc
                        if inner.find(exp.Column) and not inner.find(exp.AggFunc):
                            has_non_agg = True
                            break

                    group_by = select_node.args.get("group")
                    if has_non_agg and not group_by:
                        return ValidationResult(
                            passed=False, step="aggregation",
                            message=(
                                "SELECT mixes aggregate and non-aggregate columns "
                                "without GROUP BY. Add GROUP BY for every "
                                "non-aggregated column."
                            ),
                            sql=sql,
                        )
                        
                    # Check 3: Enforce grouping by entity IDs
                    # Fixes Pattern G: Misuse of GROUP BY Granularity (e.g. Q49, Q133 grouping by fc.name only)
                    if group_by:
                        grouped_cols = list(group_by.find_all(exp.Column))
                        has_id = any(c.name.lower() in ("id", "board_id", "script_id", "course_id", "exam_id", "qp_id", "department_id") for c in grouped_cols if c.name)
                        has_name = any(c.name.lower() in ("name", "title", "code", "display_name", "description") for c in grouped_cols if c.name)
                        
                        if has_name and not has_id:
                            return ValidationResult(
                                passed=False, step="aggregation",
                                message=(
                                    "GROUP BY uses a descriptive column (like name, title, or code) "
                                    "without including the primary key (id). Names and titles may not be unique. "
                                    "Always include the entity's 'id' column in the GROUP BY clause alongside the descriptive column."
                                ),
                                sql=sql,
                            )

        except Exception:
            pass  # Syntax errors handled elsewhere
        return ValidationResult(passed=True, step="aggregation", sql=sql)

    def _step_schema(self, sql: str, tables_used: list[str], statements=None) -> ValidationResult:
        """
        Step 2: Verify tables and columns exist in the schema map.

        This method acts as the orchestrator for the schema validation step,
        delegating verification to helper methods. It is split into three main parts:
        1. Table exists check: Verify all referenced tables exist in our database schema.
        2. Column exists check: Verify all referenced columns exist (including resolving aliases).
        3. Column type / enum check: Verify literals compared with columns are type-compatible and match constraints.
        """
        # 1. Verify table names and extract the parsed AST statements and table/alias metadata
        statements, sql_tables, alias_map, cte_names, table_res = self._validate_tables(sql, tables_used)
        if table_res is not None:
            return table_res

        # 2. Verify column existence across the AST statements
        col_res = self._validate_columns(statements, sql_tables, alias_map, sql)
        if col_res is not None:
            return col_res

        # 3. Verify type-compatibility and enum constraints on comparisons, IN lists, and BETWEEN boundaries
        type_res = self._validate_column_types(statements, sql_tables, alias_map, sql)
        if type_res is not None:
            return type_res

        return ValidationResult(passed=True, step="schema", sql=sql)

    def _validate_tables(
        self, sql: str, tables_used: list[str], statements: list[exp.Expression] | None = None
    ) -> tuple[list[exp.Expression], set[str], dict[str, str], set[str], ValidationResult | None]:
        """
        Validates that all tables referenced either in the pre-extracted `tables_used` list
        or within the SQL AST itself actually exist in the database schema inventory.

        Also constructs a mapping dictionary of table aliases to actual table names.

        Args:
            sql: The generated SQL query string.
            tables_used: List of table names identified by the orchestrator/pipeline.

        Returns:
            A 5-element tuple containing:
            1. statements (list): Parsed SQL AST statement trees.
            2. sql_tables (set): Canonical table names extracted from the SQL statement.
            3. alias_map (dict): Table aliases mapped to canonical table names (plus table name mapping to itself).
            4. cte_names (set): Set of Common Table Expression (CTE) aliases to exclude from real table checks.
            5. failure_result (ValidationResult | None): A failed ValidationResult if a check fails, else None.
        """
        # Step A: Validate table names in the pre-extracted tables_used list.
        # This acts as a first line of defense against hallucinated tables.
        unknown_tables = [t for t in tables_used if t not in self.schema_map]
        if unknown_tables:
            return (
                [],
                set(),
                {},
                set(),
                ValidationResult(
                    passed=False,
                    step="schema",
                    message=f"Hallucinated table(s): {', '.join(unknown_tables)}. "
                            f"These tables do not exist in the schema.",
                    sql=sql,
                ),
            )

        # Step B: Parse the SQL string into AST statements using sqlglot.
        if statements is None:
            # If parsing fails, we skip further AST schema checks and return passed=True.
            # The syntax validator or execution step will catch any structural SQL errors.
            return [], set(), {}, set(), ValidationResult(passed=True, step="schema", sql=sql)

        sql_tables: set[str] = set()
        alias_map: dict[str, str] = {}
        cte_names: set[str] = set()

        for stmt in statements:
            if stmt is None:
                continue

            # Collect CTE alias names. CTEs are temporary named result sets (query-local views).
            # We must not treat them as physical database tables.
            for cte in stmt.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

            try:
                # Iterate over all exp.Table nodes in the AST statement.
                for tbl in stmt.find_all(exp.Table):
                    if tbl.name:
                        name = tbl.name.lower()
                        # Only add as a real table if it's not referencing a query-local CTE alias
                        if name not in cte_names:
                            sql_tables.add(name)

                        # Construct table alias mappings (e.g. SELECT * FROM user u -> alias_map["u"] = "user")
                        alias = (tbl.alias or "").lower()
                        if alias:
                            alias_map[alias] = name
                        # A table can also reference itself directly without alias, so map its canonical name to itself.
                        alias_map[name] = name
            except Exception as exc:
                logger.warning(
                    component="sql_validator",
                    event="ast_walk_error",
                    error=str(exc),
                    note="Partial schema grounding for this statement",
                )

        # Step C: Validate table names found inside the SQL AST statements.
        # This catches tables generated by the LLM that were missed by the tables_used extractor.
        unknown_in_sql = [t for t in sql_tables if t not in self.schema_map]
        if unknown_in_sql:
            return (
                statements,
                sql_tables,
                alias_map,
                cte_names,
                ValidationResult(
                    passed=False,
                    step="schema",
                    message=f"SQL references unknown table(s): {', '.join(unknown_in_sql)}. "
                            f"Use only tables that exist in the schema.",
                    sql=sql,
                ),
            )

        return statements, sql_tables, alias_map, cte_names, None

    def _validate_columns(
        self,
        statements: list[exp.Expression],
        sql_tables: set[str],
        alias_map: dict[str, str],
        sql: str,
    ) -> ValidationResult | None:
        """
        Performs column-level existence checks by walking column nodes in the AST.

        Specifically, it:
        - Resolves table qualifiers for columns via the alias_map.
        - Handles queries with a single table unambiguously (attributing columns to that table).
        - Excludes query-local columns: CTE references and projection aliases (e.g., alias in "SELECT count(*) AS alias").

        Args:
            statements: List of parsed SQL statement trees.
            sql_tables: Set of canonical table names used in the query.
            alias_map: Mapping of table aliases and names to canonical table names.
            sql: The original SQL query string.

        Returns:
            ValidationResult | None: A failed ValidationResult if hallucinated columns are detected, else None.
        """
        col_errors: list[str] = []
        try:
            for stmt in statements:
                if stmt is None:
                    continue

                # 1. Collect CTE names to exclude query-local subqueries / virtual table references.
                cte_names = set()
                for cte in stmt.find_all(exp.CTE):
                    if cte.alias:
                        cte_names.add(cte.alias.lower())

                # 1b. Collect derived table (subquery) aliases.
                # Inline subqueries like `FROM (SELECT ... GROUP BY x) AS sub`
                # create virtual column scopes that cannot be validated against
                # the DDL schema.  References like `sub.count_col` must be
                # skipped, not flagged as hallucinated.  This fixes false
                # positives on Q8, Q12, Q105 where valid subquery aliases
                # were reported as "Hallucinated column: unaliased.<col>".
                derived_table_aliases = set()
                for subq in stmt.find_all(exp.Subquery):
                    alias = subq.alias
                    if alias:
                        derived_table_aliases.add(alias.lower())

                # 2. Collect SELECT projection aliases (e.g. SELECT count(*) AS student_count).
                # These are query-local aliases. If referenced downstream (e.g., in ORDER BY student_count),
                # they will parse as Column nodes with no table qualifier. We track them here to prevent
                # looking them up in the database schema.
                projection_aliases = set()
                for a in stmt.find_all(exp.Alias):
                    if a.alias:
                        projection_aliases.add(a.alias.lower())

                # 2b. Collect Column node IDs that live INSIDE derived-table
                # subqueries.  stmt.find_all(exp.Column) traverses the
                # entire AST recursively, including columns inside
                # FROM (SELECT ... ) AS sub.  Those inner columns belong to
                # the subquery's own scope and should NOT be validated against
                # the outer query's schema.  Without this, computed aliases
                # like COUNT(*) AS section_count inside the subquery get
                # resolved to the subquery's source table and flagged as
                # hallucinated (e.g. Q105: "question_section.section_count").
                inner_subquery_col_ids: set[int] = set()
                for subq in stmt.find_all(exp.Subquery):
                    if subq.alias:  # only derived tables (aliased subqueries)
                        for inner_col in subq.find_all(exp.Column):
                            inner_subquery_col_ids.add(id(inner_col))

                # 3. Traverse and inspect every Column node in the statement.
                for col_node in stmt.find_all(exp.Column):
                    col_name = (col_node.name or "").lower()
                    tbl_part = (col_node.table or "").lower()

                    # Ignore empty column names or wildcard selectors ("*").
                    if not col_name or col_name == "*":
                        continue

                    # Skip columns that are nested inside a derived-table
                    # subquery — they belong to the subquery's own scope.
                    if id(col_node) in inner_subquery_col_ids:
                        continue


                    # If the column belongs to a CTE, skip verification as CTE schemas are generated dynamically.
                    if tbl_part in cte_names:
                        continue

                    # If the column belongs to a derived table (inline subquery),
                    # skip verification — the subquery's output columns are
                    # query-local and not in the DDL schema.
                    if tbl_part in derived_table_aliases:
                        continue

                    # If the column is an unqualified reference to a SELECT projection alias, skip verification.
                    if not tbl_part and col_name in projection_aliases:
                        continue

                    # 4. Resolve the table associated with this column.
                    resolved_table: str | None = None
                    if tbl_part:
                        # Use the alias map to resolve alias -> canonical table name (e.g., "u" -> "user")
                        resolved_table = alias_map.get(tbl_part)
                        if not resolved_table and tbl_part in sql_tables:
                            # The table might have been referenced without an alias
                            resolved_table = tbl_part
                        
                        if not resolved_table:
                            return ValidationResult(
                                passed=False, step="schema",
                                message=f"Unknown table or alias '{tbl_part}' referenced in '{tbl_part}.{col_name}'. Ensure it is declared in the FROM/JOIN clause.",
                                sql=sql
                            )
                    elif len(sql_tables - cte_names) == 1:
                        # Single-table context: if there is only one physical table in scope, attribute the column to it.
                        # We cast the set to a list and access index [0] for readability and clarity.
                        remaining_real_tables = list(sql_tables - cte_names)
                        resolved_table = remaining_real_tables[0]

                    # 4.5 Check runtime column blocklist for commonly hallucinated columns.
                    # This blocklist intercepts known phantom columns (hallucinated by the LLM
                    # due to star-schema or generic relational database assumptions) early in the
                    # validation process. The validator fails the step and feeds a tailored corrective
                    # instruction back into the generator's self-correction retry loop, prompting
                    # it to generate the correct join paths and columns.
                    # C-2 fix: blocklist hoisted to module-level _COLUMN_BLOCKLIST constant.
                    if resolved_table and (resolved_table, col_name) in _COLUMN_BLOCKLIST:
                        return ValidationResult(
                            passed=False,
                            step="schema",
                            message=f"Column validation failed: {_COLUMN_BLOCKLIST[(resolved_table, col_name)]}",
                            sql=sql,
                        )

                    # 5. Check the resolved table's schema inventory for the column.
                    if resolved_table and resolved_table in self.schema_map:
                        inv = self.schema_map[resolved_table]
                        if hasattr(inv, "columns") and col_name not in inv.columns:
                            col_errors.append(f"{resolved_table}.{col_name}")
                    elif not resolved_table:
                        real_tables = sql_tables - cte_names
                        possible_tables = []
                        for t in real_tables:
                            if t in self.schema_map and hasattr(self.schema_map[t], "columns") and col_name in self.schema_map[t].columns:
                                possible_tables.append(t)
                        if len(possible_tables) > 1:
                            return ValidationResult(
                                passed=False, step="schema",
                                message=f"Ambiguous column reference: '{col_name}'. It exists in multiple tables ({', '.join(possible_tables)}). You must qualify it with a table alias.",
                                sql=sql
                            )
                        elif len(possible_tables) == 0 and len(real_tables) > 0:
                            col_errors.append(f"unaliased.{col_name}")

        except Exception as exc:
            # Column check is a best-effort validator; do not block query execution on unexpected AST variations.
            logger.warning(
                component="sql_validator",
                event="column_check_error",
                error=str(exc),
                note="Column hallucination check skipped due to AST error",
            )

        if col_errors:
            return ValidationResult(
                passed=False,
                step="schema",
                message=f"Hallucinated column(s): {', '.join(col_errors[:5])}. "
                        f"Use only columns that exist in the schema.",
                sql=sql,
            )

        return None

    def _validate_column_types(
        self,
        statements: list[exp.Expression],
        sql_tables: set[str],
        alias_map: dict[str, str],
        sql: str,
    ) -> ValidationResult | None:
        """
        Performs best-effort column data type and CHECK constraint enum checks.

        This method verifies:
        1. Operator Comparisons (e.g., =, >, <, !=): Checks if literal types match column families (e.g. preventing id = 'string').
        2. Categorical Allowed Values (CHECK constraints): Verifies that equality/inequality values are valid member enums.
        3. IN List clauses: Verifies each element in the IN list matches the column's type.
        4. BETWEEN bounds clauses: Verifies both lower and upper bounds match the column's type.

        Args:
            statements: List of parsed SQL statement trees.
            sql_tables: Set of canonical table names used in the query.
            alias_map: Mapping of table aliases and names to canonical table names.
            sql: The original SQL query string.

        Returns:
            ValidationResult | None: A failed ValidationResult if type/enum conflicts are detected, else None.
        """
        type_errors: list[str] = []
        enum_errors: list[str] = []

        try:
            # List of comparison node classes in sqlglot
            _comparison_types = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE,
                                  exp.NEQ, exp.Is)

            for stmt in statements:
                if stmt is None:
                    continue

                # Collect CTE names to exclude query-local subqueries / virtual table references.
                # Collecting it per-statement here avoids scope leaks and ensures correct resolution context.
                cte_names = set()
                for cte in stmt.find_all(exp.CTE):
                    if cte.alias:
                        cte_names.add(cte.alias.lower())

                # ── 1. Check comparisons (e.g. column = literal or literal = column) ──
                for cmp_node in stmt.find_all(*_comparison_types):
                    left, right = cmp_node.left, cmp_node.right
                    pairs: list[tuple[exp.Expression, exp.Expression]] = []
                    
                    # We pair columns with their compared expression
                    if isinstance(left, exp.Column):
                        pairs.append((left, right))
                    if isinstance(right, exp.Column):
                        pairs.append((right, left))

                    for col_node, val_node in pairs:
                        col_name = (col_node.name or "").lower()
                        tbl_part = (col_node.table or "").lower()
                        if not col_name:
                            continue

                        # Resolve table name. If unqualified, default to single table in scope if unambiguous.
                        resolved = alias_map.get(tbl_part) if tbl_part else (
                            list(sql_tables - cte_names)[0]
                            if len(sql_tables - cte_names) == 1 else None
                        )
                        if not resolved or resolved not in self.schema_map:
                            continue

                        col_info = self.schema_map[resolved].columns.get(col_name)
                        if col_info is None:
                            continue  # Missing columns are already caught by _validate_columns

                        # CHECK constraint validation: Catches fabricated categorical/enum values.
                        # Scoped to EQ/NEQ only since inequality/equality is where fixed set member checks are relevant.
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

                        # DDL Type Family check: Verifies column datatype family (e.g. integer, numeric, string, date).
                        family = _classify_pg_type(col_info.data_type)
                        if family is None:
                            continue

                        err = _check_literal_type_compat(family, val_node)
                        if err:
                            type_errors.append(
                                f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                            )

                # ── 2. Check IN lists (e.g. column IN (val1, val2, ...)) ──
                for in_node in stmt.find_all(exp.In):
                    col_node = in_node.this
                    if not isinstance(col_node, exp.Column):
                        continue
                    col_name = (col_node.name or "").lower()
                    tbl_part = (col_node.table or "").lower()
                    if not col_name:
                        continue

                    resolved = alias_map.get(tbl_part) if tbl_part else (
                        list(sql_tables - cte_names)[0]
                        if len(sql_tables - cte_names) == 1 else None
                    )
                    if not resolved or resolved not in self.schema_map:
                        continue

                    col_info = self.schema_map[resolved].columns.get(col_name)
                    if col_info is None:
                        continue

                    family = _classify_pg_type(col_info.data_type)
                    if family is None:
                        continue

                    for val_node in in_node.expressions:
                        err = _check_literal_type_compat(family, val_node)
                        if err:
                            type_errors.append(
                                f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                            )

                # ── 3. Check BETWEEN bounds (e.g. column BETWEEN low AND high) ──
                for between_node in stmt.find_all(exp.Between):
                    col_node = between_node.this
                    if not isinstance(col_node, exp.Column):
                        continue
                    col_name = (col_node.name or "").lower()
                    tbl_part = (col_node.table or "").lower()
                    if not col_name:
                        continue

                    resolved = alias_map.get(tbl_part) if tbl_part else (
                        list(sql_tables - cte_names)[0]
                        if len(sql_tables - cte_names) == 1 else None
                    )
                    if not resolved or resolved not in self.schema_map:
                        continue

                    col_info = self.schema_map[resolved].columns.get(col_name)
                    if col_info is None:
                        continue

                    family = _classify_pg_type(col_info.data_type)
                    if family is None:
                        continue

                    for bound in (between_node.args.get("low"),
                                  between_node.args.get("high")):
                        if bound is None:
                            continue
                        err = _check_literal_type_compat(family, bound)
                        if err:
                            type_errors.append(
                                f"{resolved}.{col_name} ({col_info.data_type}) "
                                f"BETWEEN bound: {err}"
                            )

        except Exception as exc:
            # Type check is best-effort; never crash query validation on unexpected AST structures.
            logger.warning(
                component="sql_validator",
                event="type_check_error",
                error=str(exc),
                note="Type-compatibility check skipped due to AST error",
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

    def _step_safety(self, sql: str, statements: list[exp.Expression] | None = None) -> ValidationResult:
        """
        Step 3: Guard rails - Block DML/DDL statements and Cartesian joins.

        This method implements a two-layer security validation system to ensure that
        no destructive operations (like UPDATE/DELETE/DROP) or performance-killing Cartesian
        joins (cross products) are executed.

        Two-layer defense architecture:
        ================================
        Layer 1 — AST Inspection (Primary, structured validation):
          - Parses the SQL statement into an Abstract Syntax Tree (AST) using sqlglot.
          - Recursively walks the tree nodes.
          - Rejects any statement that is not a SELECT (or WITH statement encapsulating a SELECT).
          - Rejects statements containing destructive DML or DDL nodes anywhere in the tree.
          - If parsing and verification succeed, Layer 2 is skipped.

        Layer 2 — Blocked-Pattern Regex (Fallback, defensive validation):
          - Only runs if Layer 1 (AST parsing) fails due to invalid SQL syntax.
          - Searches for keywords like 'UPDATE', 'DELETE', etc.
          - By only running on parsing failures, we avoid false-positives where keywords appear
            within legitimate text literals (e.g., WHERE action_type = 'UPDATE').

        Cartesian Join Prevention:
        ==========================
        - Detects implicit Cartesian joins (e.g., joins that lack 'ON' or 'USING' conditions).
        - Prevents catastrophic query executions on large tables that would exhaust database memory.
        - Excludes intentional, explicit 'CROSS JOIN' statements.
        """
        # ── Layer 1: AST (Abstract Syntax Tree) Inspection ────────────────
        # List of dangerous database modification classes in sqlglot.
        _DML_NODES = (
            exp.Insert, exp.Update, exp.Delete,
            exp.Drop, exp.Create, exp.Command,
            exp.Grant, exp.Revoke,
        )
        ast_parse_succeeded = False
        try:
            # Parse the query string into a list of AST statements if not provided
            statements = statements or sqlglot.parse(sql, dialect="postgres")
            ast_parse_succeeded = True
            for stmt in statements:
                # Top-level query statement must be a read-only SELECT or a CTE wrapper (exp.With)
                if not isinstance(stmt, (exp.Select, exp.With)):
                    kind = type(stmt).__name__
                    return ValidationResult(
                        passed=False, step="safety",
                        message=f"Non-SELECT statement detected by AST: {kind}. "
                                f"Only SELECT queries are permitted.",
                        sql=sql,
                    )
                # Walk every subnode in the AST to look for nested modification statements (e.g., SELECT ... INTO)
                for node in stmt.walk():
                    if isinstance(node, _DML_NODES):
                        kind = type(node).__name__
                        return ValidationResult(
                            passed=False, step="safety",
                            message=f"DML/DDL node '{kind}' found in query tree. "
                                    f"Only SELECT queries are permitted.",
                            sql=sql,
                        )
        except sqlglot.errors.ParseError:
            logger.exception("step_safety_parse_error")
            # Mark parse as failed so that we fall back to the secondary Regex layer
            ast_parse_succeeded = False

        # ── Layer 2: Blocked-Pattern Regex Fallback ──────────────────────
        # Only runs if the SQL could not be parsed into a valid AST.
        if not ast_parse_succeeded:
            # Match against configured blocked patterns (configured in settings)
            match = settings.validation.blocked_pattern.search(sql)
            if match:
                return ValidationResult(
                    passed=False, step="safety",
                    message=f"Blocked keyword '{match.group(0).upper()}' detected. "
                            f"Only SELECT queries are permitted.",
                    sql=sql,
                )

        # ── Cartesian Join Check ─────────────────────────────────────────
        cartesian_detected = False
        if ast_parse_succeeded:
            try:
                # Walk the AST to find Join expressions and verify join clauses exist
                for stmt in statements:
                    if stmt is None:
                        continue
                    for join in stmt.find_all(exp.Join):
                        has_on    = join.args.get("on")    is not None
                        has_using = join.args.get("using") is not None
                        
                        # Check the join type/kind (e.g. LEFT, RIGHT, CROSS).
                        join_kind = (join.args.get("kind") or "").upper()
                        
                        # A join is a Cartesian join if it lacks ON/USING conditions,
                        # and is not explicitly labeled as an intentional 'CROSS JOIN'.
                        if not has_on and not has_using and join_kind != "CROSS":
                            cartesian_detected = True
                            break
                    if cartesian_detected:
                        break
            except sqlglot.errors.ParseError:
                logger.exception("cartesian_check_parse_error")
                # Fallback to regex checking if AST fails on Cartesian check
                cartesian_detected = bool(_CARTESIAN_PATTERN.search(sql))
        else:
            # Fall back to regex checking since AST parsing was unsuccessful
            cartesian_detected = bool(_CARTESIAN_PATTERN.search(sql))

        if cartesian_detected:
            return ValidationResult(
                passed=False, step="safety",
                message="Cartesian join detected (JOIN without ON or USING clause). "
                        "Use explicit JOIN ... ON syntax.",
                sql=sql,
            )

        return ValidationResult(passed=True, step="safety", sql=sql)

    def _step_security(
        self,
        sql:          str,
        tables_used:  list[str],
        user_context: dict,
        statements=None,
    ) -> ValidationResult:
        """
        Step 4: Tenant Isolation - Inject tenant filters idempotently.

        This method guarantees multi-tenant isolation by ensuring that tenant filters
        (e.g., board_id = 5) are present on any table touching tenant-specific data.
        It modifies the SQL statement programmatically using AST manipulation to inject
        the filters if they are missing.

        Idempotency:
        ============
        To prevent injecting duplicate filters (e.g., WHERE board_id = 5 AND board_id = 5),
        the validator walks the AST first to verify if a valid equality predicate for the
        desired tenant is already present. Simple substring searches are avoided to prevent
        skipping injection on queries that merely select the column but don't filter by it.

        Priority Ordering of Isolation Scopes:
        ======================================
        1. board_id: The most narrow/specific filter (used for academic/evaluation boards).
        2. course_id: Broader filter (used for course-level data).
        3. user_id: Connection-level Row Level Security (RLS) variable filter.
        """
        # If no tenant isolation configuration exists in settings, pass validation
        if not settings.rls_variable and not settings.tenant_column:
            return ValidationResult(passed=True, step="security", sql=sql)

        # Check if the query targets any tenant-scoped tables
        has_tenant_table = any(t in self._tenant_scoped_tables for t in tables_used)
        if not has_tenant_table:
            return ValidationResult(passed=True, step="security", sql=sql)

        # H-1 fix: Check if a SET LOCAL statement for the RLS variable exists in the SQL
        # using AST walk instead of fragile substring matching. The old substring check
        # (rls_var.split(".")[-1] in sql.lower()) was bypassed by any SQL that mentioned
        # the variable name in a column reference, alias, or string literal.
        rls_var = settings.rls_variable
        if rls_var:
            rls_key = rls_var.split(".")[-1].lower()
            try:
                # Check for SET LOCAL statement via AST — the only safe way to verify
                # that the RLS variable is being set, not just referenced.
                stmts = statements or sqlglot.parse(sql, dialect="postgres")
                for stmt in stmts:
                    if stmt is None:
                        continue
                    for node in stmt.walk():
                        if isinstance(node, exp.SetItem):
                            eq = node.find(exp.EQ)
                            if eq and rls_key in str(eq).lower():
                                return ValidationResult(passed=True, step="security", sql=sql)
            except Exception:
                # AST parse failed — fall through to check via regex as last resort
                set_local_pattern = re.compile(
                    rf"\bSET\s+LOCAL\s+{re.escape(rls_var)}\s*=", re.IGNORECASE
                )
                if set_local_pattern.search(sql):
                    return ValidationResult(passed=True, step="security", sql=sql)

        # ── Path 1: Scoping by board_id ───────────────────────────────────
        board_id = user_context.get("board_id")
        if board_id:
            try:
                safe_board_id = int(board_id)
            except (ValueError, TypeError):
                return ValidationResult(
                    passed=False, step="security",
                    message=f"Invalid board_id in user context: {board_id!r}. "
                            f"board_id must be an integer.",
                    sql=sql,
                )
            
            # If the SQL does not have a board_id = <value> constraint, inject it.
            if not self._has_eq_predicate(sql, "board_id", safe_board_id):
                injected_sql = self._inject_where(sql, "board_id", safe_board_id)
                if injected_sql:
                    logger.info(
                        component="sql_validator",
                        event="tenant_filter_injected",
                        scope="board_id",
                        value=safe_board_id,
                    )
                    return ValidationResult(passed=True, step="security", sql=injected_sql)

        # ── Path 2: Scoping by course_id ──────────────────────────────────
        course_id = user_context.get("course_id")
        if course_id:
            try:
                safe_course_id = int(course_id)
            except (ValueError, TypeError):
                return ValidationResult(
                    passed=False, step="security",
                    message=f"Invalid course_id in user context: {course_id!r}. "
                            f"course_id must be an integer.",
                    sql=sql,
                )
            
            # If the SQL does not have a course_id = <value> constraint, inject it.
            if not self._has_eq_predicate(sql, "course_id", safe_course_id):
                injected_sql = self._inject_where(sql, "course_id", safe_course_id)
                if injected_sql:
                    logger.info(
                        component="sql_validator",
                        event="tenant_filter_injected",
                        scope="course_id",
                        value=safe_course_id,
                    )
                    return ValidationResult(passed=True, step="security", sql=injected_sql)

        # ── Path 3: Scoping by user_id via Row Level Security (RLS) ────────
        user_id = user_context.get("user_id")
        if user_id and rls_var:
            logger.info(
                component="sql_validator",
                event="tenant_filter_rls",
                scope="user_id",
                rls_var=rls_var,
            )
            return ValidationResult(passed=True, step="security", sql=sql)

        # H-5 fix: Fallback log now includes SQL preview and user context keys
        # for audit trail. Previously only logged table names, making it impossible
        # to trace which query bypassed tenant filtering.
        logger.warning(
            component="sql_validator",
            event="tenant_filter_unavailable",
            tables=tables_used,
            sql_preview=sql[:120],
            user_context_keys=list(user_context.keys()),
            note="Query touches tenant-scoped tables but no board_id / course_id / "
                 "user_id found in user_context. Allowed through — verify this is "
                 "an admin query.",
        )
        return ValidationResult(passed=True, step="security", sql=sql)

    def _has_eq_predicate(self, sql: str, col_name: str, value: int) -> bool:
        """
        Check if the SQL already contains a matching equality filter.

        Walks the AST looking for `exp.EQ` nodes (representing '=' comparison predicates)
        and verifies if it checks the target column against the given tenant integer value.

        Args:
            sql: The SQL query string.
            col_name: The column name to check (e.g. 'board_id').
            value: The tenant ID value (e.g. 5).

        Returns:
            bool: True if 'column = value' is already present in the SQL, otherwise False.
        """
        try:
            stmt = sqlglot.parse_one(sql, dialect="postgres")
            for eq in stmt.find_all(exp.EQ):
                left  = eq.left
                right = eq.right
                col_node = None
                val_node = None
                
                # Check both orientations: col_name = value or value = col_name
                if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                    col_node, val_node = left, right
                elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                    col_node, val_node = right, left
                
                if col_node and val_node:
                    if col_node.name.lower() == col_name.lower():
                        try:
                            if int(val_node.this) == value:
                                return True
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass
        return False

    def _inject_where(self, sql: str, col_name: str, value: int) -> str | None:
        """
        Inject a tenant isolation filter into the query using AST manipulation.

        Rather than concatenating query strings (which is vulnerable to SQL injection),
        this parses the query, modifies the appropriate AST nodes, and renders the modified
        AST back to a SQL string.

        Qualifying Columns:
        ===================
        To avoid "ambiguous column" SQL errors in joins, this resolves table aliases
        and qualifies the injected predicate (e.g. "a.board_id = 5" instead of "board_id = 5").

        Scoping Selection:
        ==================
        1. Outer SELECT Scope: Normal queries have filters injected in the main WHERE clause.
        2. CTE Scope: If the query encapsulates data within Common Table Expressions (CTEs),
           it locates the CTE node containing the target table and injects the filter inside it.
        """
        try:
            stmt = sqlglot.parse_one(sql, dialect="postgres")

            # Step 1: Map all table names and aliases used in the query
            alias_map: dict[str, str] = {}
            for tbl in stmt.find_all(exp.Table):
                canon = tbl.name.lower()
                alias = (tbl.alias or "").lower()
                if alias:
                    alias_map[alias] = canon
                alias_map[canon] = canon

            def _tables_and_aliases_in_scope(select_node: exp.Select) -> list[tuple[str, str]]:
                """Extract all tables declared in the FROM and JOIN clauses of a SELECT node."""
                result = []
                from_node = select_node.args.get("from")
                if from_node:
                    for tbl in from_node.find_all(exp.Table):
                        canon = tbl.name.lower()
                        alias = (tbl.alias or tbl.name or "").lower()
                        result.append((alias, canon))
                for join in select_node.args.get("joins", []):
                    for tbl in join.find_all(exp.Table):
                        canon = tbl.name.lower()
                        alias = (tbl.alias or tbl.name or "").lower()
                        result.append((alias, canon))
                return result

            def _qualified_predicate(select_node: exp.Select) -> str | None:
                """Return a qualified predicate string if any table in scope owns the target column."""
                for alias, canon in _tables_and_aliases_in_scope(select_node):
                    inv = self.schema_map.get(canon)
                    if inv and hasattr(inv, "columns") and col_name in inv.columns:
                        qualifier = alias if alias else canon
                        return f"{qualifier}.{col_name} = {value}"
                return None

            # Step 2: Inject into the outermost SELECT scope if appropriate
            outer_select = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
            if outer_select:
                predicate = _qualified_predicate(outer_select)
                if predicate:
                    injected = stmt.where(predicate, dialect="postgres")
                    return injected.sql(dialect="postgres")

            # Step 3: Inject into CTE body scope if the table is located inside a CTE query block
            for cte in stmt.find_all(exp.CTE):
                cte_select = cte.find(exp.Select)
                if cte_select is None:
                    continue
                predicate = _qualified_predicate(cte_select)
                if predicate:
                    modified = cte_select.where(predicate, dialect="postgres")
                    cte.set("this", modified)
                    return stmt.sql(dialect="postgres")

            # Step 4: Fallback - inject unqualified column to the outer WHERE clause
            predicate_unqualified = f"{col_name} = {value}"
            injected = stmt.where(predicate_unqualified, dialect="postgres")
            return injected.sql(dialect="postgres")

        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="tenant_filter_ast_failed",
                col_name=col_name,
                value=value,
                error=str(exc),
                note="AST injection failed; trying next scoping path",
            )
            return None

    def _step_cost(self, sql: str) -> ValidationResult:
        """
        Step 5: Resource Cost Limit check - Run EXPLAIN and check estimated costs.

        Queries the database planner via `EXPLAIN (FORMAT JSON)` to get cost estimations.
        If the query planner's estimated total cost exceeds the configured threshold,
        validation fails. This blocks heavy or unindexed queries from crashing the database.

        Execution Options:
        ==================
        1. MCP Server path: Uses Model Context Protocol tool client calls.
        2. Direct Connection path: Uses psycopg2 database connection pool.

        Inflated Cost Prevention:
        ==========================
        To ensure accurate scanning estimates, a LIMIT clause is appended to queries
        prior to running EXPLAIN. Otherwise, Postgres assumes full-table scans when the
        client will only retrieve a limited subset, generating inflated costs and false failures.
        """
        threshold = settings.validation.explain_cost_threshold

        def check_pgcode(pgcode: str, error_msg: str) -> ValidationResult | None:
            """
            Analyze the Postgres error code. Return a failed ValidationResult if the
            query is structurally flawed, otherwise return None (so it can be ignored).
            
            Catch query-level errors exposed by the EXPLAIN planner so they get retried:
            - Class 42 (Syntax Error or Access Rule Violation): Missing commas, hallucinated columns, etc.
            - Class 22 (Data Exception): Bad casting, division by zero, JSON extraction precedence issues.
            We do NOT want to catch class 08 (Connection Exception) because those indicate DB downtime, 
            not a flaw in the generated SQL.

            2026-06-25 fix: reclassify column/table/operator errors from "cost"
            to "schema" with an actionable correction message.  The retry-loop
            uses the step label to choose what context to attach to the
            correction prompt — pointing the LLM at the right schema chunk is
            far more useful than telling it "the cost step failed".  This was
            observed in ~17 of 40 batch failures where the EXPLAIN error was
            actually a hallucinated column that slipped past _validate_columns
            (e.g. after a retry introduced a new phantom column not in alias_map).
            """
            if pgcode.startswith("42") or pgcode.startswith("22"):
                # Try to extract a more specific category and produce a
                # correction-friendly message.
                step, message = _classify_pg_error(error_msg)
                return ValidationResult(
                    passed  = False,
                    step    = step,
                    message = message,
                    sql     = sql,
                )
            return None

        # ── Case A: MCP Connection Path ───────────────────────────────────
        if settings.use_mcp_servers:
            try:
                result = call_postgres_explain(sql)
            except MCPCallError as exc:
                logger.warning(
                    component = "sql_validator",
                    event     = "explain_mcp_unavailable",
                    error     = str(exc),
                    note      = "Skipping cost check — MCP postgres server unreachable.",
                )
                return ValidationResult(passed=True, step="cost", sql=sql)

            if "error" in result:
                pgcode = result.get("pgcode", "")
                
                validation_err = check_pgcode(pgcode, result["error"])
                if validation_err:
                    return validation_err
                    
                logger.warning(component="sql_validator", event="explain_failed",
                               error=result["error"])
                return ValidationResult(passed=True, step="cost", sql=sql)

            total_cost = result.get("total_cost", 0.0)
            logger.info(component="sql_validator", event="explain_complete",
                        total_cost=total_cost, explain_ms=result.get("elapsed_ms"))

            plan = result.get("plan")
            warnings = []
            if plan and isinstance(plan, list):
                self._inspect_plan_node(plan[0].get("Plan", {}), warnings)

            if total_cost > threshold:
                msg = f"Query estimated cost {total_cost:.0f} exceeds threshold {threshold}. Add specific filters (board_id, exam_id, etc.)."
                if warnings:
                    msg += "\nPerformance issues:\n- " + "\n- ".join(warnings)
                return ValidationResult(
                    passed  = False,
                    step    = "cost",
                    message = msg,
                    sql     = sql,
                )
            return ValidationResult(passed=True, step="cost", sql=sql)

        # ── Case B: Direct psycopg2 Connection Path ───────────────────────
        conn       = None
        using_pool = False

        if self._get_conn is not None:
            # Wrap connection retrieval in try/except block to handle connection pool timeouts gracefully.
            try:
                conn = self._get_conn()
                using_pool = True
            except Exception as exc:
                logger.warning(
                    component="sql_validator",
                    event="cost_check_connection_failed",
                    error=str(exc),
                    note="Skipping cost check — could not acquire a connection "
                          "(pool timeout or other connection error).",
                )
                conn = None
                using_pool = False
        elif self.db_dsn:
            conn = psycopg2.connect(
                self.db_dsn,
                options = f"-c statement_timeout={settings.postgres.statement_timeout_ms}",
            )
            conn.set_session(readonly=True)

        if conn is None:
            # If no DB connection details are configured, skip cost check.
            return ValidationResult(passed=True, step="cost", sql=sql)

        try:
            cur = conn.cursor()

            # Append LIMIT statement to prevent cost inflated estimates
            explain_body = sql.rstrip(";")
            if not _outer_query_has_limit(explain_body):
                explain_body = f"{explain_body} LIMIT {settings.postgres.max_rows}"

            t0 = time.time()
            cur.execute(f"EXPLAIN (FORMAT JSON) {explain_body}")
            plan    = cur.fetchone()[0]
            elapsed = round((time.time() - t0) * 1000)
            cur.close()

            if not using_pool:
                conn.close()

            total_cost = 0.0
            if plan and isinstance(plan, list):
                total_cost = plan[0].get("Plan", {}).get("Total Cost", 0.0)

            logger.info(component="sql_validator", event="explain_complete",
                        total_cost=total_cost, explain_ms=elapsed)

            warnings = []
            if plan and isinstance(plan, list):
                self._inspect_plan_node(plan[0].get("Plan", {}), warnings)

            if total_cost > threshold:
                msg = f"Query estimated cost {total_cost:.0f} exceeds threshold {threshold}. Add specific filters (board_id, exam_id, etc.)."
                if warnings:
                    msg += "\nPerformance issues:\n- " + "\n- ".join(warnings)
                return ValidationResult(
                    passed  = False,
                    step    = "cost",
                    message = msg,
                    sql     = sql,
                )

        except psycopg2.Error as exc:
            logger.warning(
                component="sql_validator",
                event="explain_error",
                error=str(exc),
                sql_preview=sql[:80],
            )
            pgcode = getattr(exc, "pgcode", None) or ""
            
            validation_err = check_pgcode(pgcode, str(exc))
            if validation_err:
                return validation_err
                
            logger.warning(component="sql_validator", event="explain_failed",
                           error=str(exc))

        finally:
            # Ensure transactions are rolled back before returning connections to the pool.
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if using_pool and self._release_conn and conn is not None:
                self._release_conn(conn)

        return ValidationResult(passed=True, step="cost", sql=sql)

    def _inspect_plan_node(self, node: dict, warnings: list[str]) -> None:
        """
        Recursively traverse the EXPLAIN plan node to extract performance issues.

        Searches for sub-optimal query execution strategies (like sequential scans
        on tables exceeding 1,000 rows) and appends actionable warning hints to guide
        the model retry repair loop.
        """
        if not node:
            return
        node_type = node.get("Node Type", "")
        relation = node.get("Relation Name", "")
        plan_rows = node.get("Plan Rows", 0)
        total_cost = node.get("Total Cost", 0.0)

        # Flag Sequential Scans on large tables (e.g. above 1,000 rows)
        if node_type == "Seq Scan" and relation:
            if plan_rows > 1000:
                warnings.append(
                    f"Sequential Scan on table '{relation}' (estimated {plan_rows} rows, cost {total_cost:.1f}). "
                    f"Consider adding specific filters (e.g. board_id, course_id, exam_id, or student_id) "
                    f"to allow the query planner to use existing indexes."
                )

        # Recurse on sub-plans inside the EXPLAIN plan JSON tree
        for child in node.get("Plans", []):
            self._inspect_plan_node(child, warnings)

    # ─────────────────────────────────────────────────────────────────────
    # Step 7 — Semantic heuristic checks (delegated to semantic_checks.py)
    # ─────────────────────────────────────────────────────────────────────

    def _step_semantic(self, sql: str, original_query: str) -> ValidationResult:
        """
        Step 7: Lightweight heuristic logic checks.

        These are pure string-analysis checks (no DB or LLM calls) that catch
        common logical mismatches between the user's question and the generated
        SQL. Each check targets a failure pattern observed in batch evaluation:

        Check 1 — Anti-join mismatch:
          Question asks for "missing/without/no X" but SQL uses INNER JOIN
          instead of LEFT JOIN...IS NULL or NOT EXISTS. This causes the query
          to return the OPPOSITE of what was asked (entities WITH X, not
          entities WITHOUT X). Observed in Q33, Q60 of batch evaluation.

        Check 2 — Percentage without multiplication:
          Question asks for a "percentage" but SQL never multiplies by 100.
          The FILTER clause pattern (Rule 9) requires * 100.0. Without it,
          the result is a ratio (0.0–1.0) not a percentage (0–100).

        Check 3 — "Average per" without AVG wrapper:
          Question asks for "average X per Y" but SQL uses only GROUP BY
          without wrapping in SELECT AVG(cnt) FROM (...) sub. This returns
          a LIST of counts, not their average. Observed in Q12, Q26.

        Returns ValidationResult with passed=False and an actionable error
        message if any check fails, allowing the retry loop to self-correct.
        """
        return check_semantic(sql, original_query)

    # ─────────────────────────────────────────────────────────────────────
    # Step 8 — Hardcoded literal detection (delegated to semantic_checks.py)
    # ─────────────────────────────────────────────────────────────────────

    def _step_hardcoded_literals(self, sql: str, original_query: str) -> ValidationResult:
        """
        Step 8: Detect suspiciously hardcoded integer literal IDs via AST.
        """
        return check_hardcoded_literals(sql, original_query)

    def _step_joins(self, sql: str, tables_used: list[str]) -> ValidationResult:
        """
        Step 2.5: Validate JOIN paths against explicit FK relationships.
        
        Fixes Pattern C & F: Wrong Join Paths, Composite FKs, & Schema Misuse.
        """
        if getattr(self, 'fk_graph', None) is None:
            return ValidationResult(passed=True, step="joins", sql=sql)
            
        statements, sql_tables, alias_map, cte_names, _ = self._validate_tables(sql, tables_used)
        for stmt in statements:
            if stmt is None: continue
            for join in stmt.find_all(exp.Join):
                on_clause = join.args.get("on")
                if not on_clause: continue
                
                # Group ON conditions by table pair
                joined_cols = {}
                for eq in on_clause.find_all(exp.EQ):
                    left_col = eq.left if isinstance(eq.left, exp.Column) else eq.left.find(exp.Column)
                    right_col = eq.right if isinstance(eq.right, exp.Column) else eq.right.find(exp.Column)
                    
                    if left_col and right_col and left_col.table and right_col.table:
                        tbl1 = alias_map.get(left_col.table.lower(), left_col.table.lower())
                        tbl2 = alias_map.get(right_col.table.lower(), right_col.table.lower())
                        
                        if tbl1 != tbl2 and tbl1 not in cte_names and tbl2 not in cte_names:
                            pair = tuple(sorted([tbl1, tbl2]))
                            if pair not in joined_cols:
                                joined_cols[pair] = []
                            joined_cols[pair].append((tbl1, left_col.name.lower(), tbl2, right_col.name.lower()))

                for (tA, tB), equalities in joined_cols.items():
                    if not self.fk_graph.has_edge(tA, tB) and not self.fk_graph.has_edge(tB, tA):
                        logger.warning(
                            component="sql_validator",
                            event="invalid_join_path",
                            tbl1=tA, tbl2=tB
                        )
                        return ValidationResult(
                            passed=False, step="joins",
                            message=f"Invalid JOIN: No explicit foreign key relationship exists between '{tA}' and '{tB}'. Please review the schema for the correct join path.",
                            sql=sql
                        )
                    
                    # Check composite FKs if edge data has column_mappings
                    edge_data = self.fk_graph.get_edge_data(tA, tB) or self.fk_graph.get_edge_data(tB, tA)
                    if edge_data:
                        # Handle MultiDiGraph (dict of edges)
                        mappings = None
                        if isinstance(edge_data, dict) and 0 in edge_data:
                            # Use the first edge's mappings
                            mappings = edge_data[0].get('column_mappings')
                        elif isinstance(edge_data, dict):
                            mappings = edge_data.get('column_mappings')
                            
                        if mappings:
                            required_pairs = set()
                            for m in mappings:
                                src_col = m.get('source_column')
                                tgt_col = m.get('target_column')
                                if src_col and tgt_col:
                                    required_pairs.add((src_col, tgt_col))
                                    required_pairs.add((tgt_col, src_col))
                            
                            # Check if the ON clause has all required columns
                            matched_cols = set()
                            for tbl1, c1, tbl2, c2 in equalities:
                                matched_cols.add((c1, c2))
                                matched_cols.add((c2, c1))
                            
                            for req_c1, req_c2 in required_pairs:
                                if (req_c1, req_c2) not in matched_cols:
                                    return ValidationResult(
                                        passed=False, step="joins",
                                        message=(
                                            f"Invalid JOIN: The relationship between '{tA}' and '{tB}' uses a composite key. "
                                            f"The JOIN condition must include all columns. Missing match for column '{req_c1}' or '{req_c2}'."
                                        ),
                                        sql=sql
                                    )

        return ValidationResult(passed=True, step="joins", sql=sql)



class RetryValidator:
    """
    Wraps SQLValidator with self-correction retry / repair loop logic.

    If a validation step fails, this class orchestrates query recovery by:
      1. Formatting validation failures and original natural language questions into prompts.
      2. Querying the LLM generator to fix the error in the SQL.
      3. Running validation on the corrected SQL again.
    
    This loop continues until the query passes validation or the maximum retry limit is reached.
    """

    def __init__(
        self,
        validator:     SQLValidator,
        sql_generator,           # SQLGenerator — injected to avoid circular import
        prompt_builder,          # PromptBuilder
    ) -> None:
        self.validator      = validator
        self.sql_generator  = sql_generator
        self.prompt_builder = prompt_builder

    def validate_with_retry(
        self,
        sql:             str,
        original_query:  str,
        tables_used:     list[str]  = None,
        user_context:    dict | None = None,
        schema_context:  str        = "",
        label_filters:   list[dict] = None,
        on_retry_fallback: callable = None,  # Callback to dynamically expand context on retry
        parsed_query                = None,
        max_retries:     int        = None,
    ) -> tuple[ValidationResult, int]:
        """
        Validate SQL with up to max_retries correction attempts.

        Args:
            sql: The initial candidate SQL query string generated by the model.
            original_query: The user's natural language question.
            tables_used: List of database tables involved in the query.
            user_context: Key context mapping parameters (e.g. board_id, user_id).
            schema_context: Chunk schemas injected into the correction context.
            label_filters: RapidFuzz matching details to preserve quotes on strings.
            on_retry_fallback: Optional callback taking (attempt_num, tables_used) to fetch expanded context.

        Returns:
            A tuple of (final_ValidationResult, retry_count).
        """
        tables_used   = tables_used   or []
        user_context  = user_context  or {}
        label_filters = label_filters or []
        max_retries   = max_retries if max_retries is not None else settings.validation.max_retries

        # Perform the first check against the candidate SQL.
        # Pass original_query for Step 7 semantic heuristic checks.
        result  = self.validator.validate(sql, tables_used, user_context, original_query=original_query)
        retries = 0

        # Loop to correct/repair failed SQL candidate strings based on validation step failures.
        # This acts as a feedback loop between the validator (providing error diagnostics) and the generator (fixing errors).
        while not result.passed and retries < max_retries:
            retries += 1
            logger.info(
                component="retry_validator",
                event="retrying",
                attempt=retries,
                step=result.step,
                error=result.message[:100],
            )

            # Dynamically expand schema context if a fallback callback is provided.
            # On subsequent retries, the initial schema context might have been too restricted,
            # so we request broader database schema metadata (expanded context budget) to guide correction.
            if on_retry_fallback:
                try:
                    schema_context = on_retry_fallback(retries, tables_used)
                except Exception as exc:
                    logger.warning(
                        component="retry_validator",
                        event="retry_fallback_failed",
                        error=str(exc)
                    )

            # Build repair/correction instructions for the LLM.
            # This packages the user's query, the failing SQL, the validation error, and the schema context
            # into a structured correction prompt.
            correction_prompt = self.prompt_builder.build_correction_prompt(
                original_query = original_query,
                failed_sql     = sql,
                error_message  = result.message,
                schema_context = schema_context,
                label_filters  = label_filters,
                parsed_query   = parsed_query
            )

            # Run the SQL generator model on the correction prompt to generate a repaired candidate.
            corrected   = self.sql_generator.generate(correction_prompt)
            sql         = corrected.sql
            tables_used = corrected.tables_used or tables_used

            if not sql:
                # Break early if the generator returned empty SQL
                break

            # Validate the corrected SQL candidate again.
            # This loops back to check if the new query passes all validation steps.
            result = self.validator.validate(sql, tables_used, user_context, original_query=original_query)

        return result, retries
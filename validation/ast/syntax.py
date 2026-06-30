import re
import sqlglot.expressions as exp
from typing import Any

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

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

# Parameter placeholder regex — detects :param_name and $N patterns that
# the LLM sometimes generates instead of literal values.  These cause
# "there is no parameter $1" errors at EXPLAIN time.
# Negative lookbehind (?<!:) avoids matching PostgreSQL's :: cast operator.
# Negative lookbehind (?<!\w) avoids matching inside identifiers.
PLACEHOLDER_RE = re.compile(r"(?<!:)(?<!\w):\w+|\$\d+")


class SyntaxValidator(BaseValidationStep):
    name = "SyntaxValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """Step 1: Parse SQL logic and check for syntax errors or reserved aliases."""
        sql = ctx.working_sql or ctx.sql
        if not sql or not sql.strip():
            return ValidationResult(
                passed=False, step="syntax",
                message="Empty SQL generated.", sql=sql,
            )

        if ctx.ast is None:
            logger.warning(
                component="sql_validator",
                event="sql_syntax_parse_error",
                error="Parse error",
                sql_preview=sql[:80],
            )
            
            # Provide explicit feedback if the LLM used a reserved keyword as a table alias.
            # We first strip strings and comments to avoid false positives.
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


class PlaceholderValidator(BaseValidationStep):
    name = "PlaceholderValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 1.1: Reject SQL containing parameter placeholders.

        The LLM sometimes generates parameterized queries (:qp_id, :script_id,
        $1) instead of using literal values or filtering by name/title.
        These cause cryptic errors at EXPLAIN time ("there is no parameter $1").
        """
        sql = ctx.working_sql or ctx.sql
        match = PLACEHOLDER_RE.search(sql)
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


class AliasValidator(BaseValidationStep):
    name = "AliasValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 1.8: A4 Detector - Catch hallucinations where the LLM uses a table alias 
        (e.g., in SELECT or WHERE) but forgot to actually JOIN that table.
        """
        sql = ctx.working_sql or ctx.sql
        if ctx.ast is None:
            return ValidationResult(passed=True, step="schema", sql=sql)
            
        for stmt in ctx.ast:
            declared_aliases = set()
            
            # 1. Find all CTEs (WITH clauses) as they act like declared tables.
            cte_names = set(cte.alias.lower() for cte in stmt.find_all(exp.CTE) if cte.alias)
            
            # 2. Find all tables properly declared in FROM or JOIN clauses.
            for tbl in stmt.find_all(exp.Table):
                if tbl.name and tbl.name.lower() not in cte_names:
                    declared_aliases.add((tbl.alias or tbl.name).lower())
                    
            # 3. Check every column reference in the query (e.g., 'table_alias.column_name').
            for col in stmt.find_all(exp.Column):
                tbl_ref = (col.table or "").lower()
                # If the column has a prefix, it MUST match a CTE or a declared table alias.
                if tbl_ref and tbl_ref not in declared_aliases and tbl_ref not in cte_names:
                    logger.warning(
                        component="sql_validator",
                        event="undeclared_alias_detected",
                        alias=tbl_ref,
                        column=col.name
                    )
                    return ValidationResult(
                        passed=False, step="schema",
                        message=(
                            f"Table or alias '{tbl_ref}' is referenced (e.g. '{tbl_ref}.{col.name}') "
                            f"but is not declared in the FROM/JOIN clauses. Add the missing table "
                            f"to the FROM list, or check JOIN ordering."
                        ),
                        sql=sql,
                    )
        
        # Populate context state for future steps to reuse
        for stmt in ctx.ast:
            for cte in stmt.find_all(exp.CTE):
                if cte.alias:
                    ctx.cte_names.add(cte.alias.lower())

        return ValidationResult(passed=True, step="schema", sql=sql)

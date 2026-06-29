"""
validation/semantic_checks.py
──────────────────────────────
Semantic heuristic checks and hardcoded-literal detection for SQL validation.

Contains Step 7 (12 semantic checks) and Step 8 (hardcoded literal IDs).
These are pure analysis — no DB or LLM calls — and form the fastest-changing
part of the validator.  New checks are added with each batch evaluation audit.

Separated from sql_validator.py so that new checks can be added without
touching the core structural validation pipeline.
"""

from __future__ import annotations

import re

import sqlglot
import sqlglot.expressions as exp

from models.schema import ValidationResult
from utils.logging_config import get_logger
from utils.heuristics import HEURISTICS

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Ordinal/sequential columns that should not be averaged directly.
# AVG(page_number) averages page *numbers* (meaningless), not page *count*.
_ORDINAL_COLUMNS = frozenset(HEURISTICS.get('ordinal_columns', []))

# Known phantom string values the LLM invents as placeholder IDs.
_PHANTOM_STRING_PATTERNS = frozenset(HEURISTICS.get('phantom_patterns', []))

# Known safe integer literals that appear frequently in valid SQL.
SAFE_LITERALS = frozenset(str(x) for x in HEURISTICS.get('safe_literals', []))


# ── Step 7: Semantic heuristic checks ─────────────────────────────────────────

def check_semantic(sql: str, original_query: str, schema_map: dict | None = None) -> ValidationResult:
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

    PHASE-1 FIX (Check 18b):
      schema_map (optional) enables a SCHEMA-DRIVEN defensive-filter
      check that supplements the original hard-coded keyword-list
      Check 18.  When provided, the new check fires for ANY column whose
      DDL has a CHECK (col IN (...)) constraint (i.e. an enum), if the
      SQL filters that column to a value not present in the NL question.
      This catches the Q67/Q91/Q162 class of hidden bugs where the LLM
      added an unprompted enum filter on a column the original Check 18
      didn't know to look at (e.g. user_type, rule_type).
    """
    sql_lower = sql.lower()
    query_lower = original_query.lower()

    # ── Check 1: Anti-join mismatch ────────────────────────────────────
    # Detect: question has negation words AND SQL uses only INNER JOINs
    # (no LEFT JOIN, NOT EXISTS, or IS NULL pattern).
    negation_phrases = HEURISTICS.get('anti_join_negation_phrases', [])
    has_negation = any(phrase in query_lower for phrase in negation_phrases)
    has_anti_pattern = (
        'not exists' in sql_lower
        or 'is null' in sql_lower
        or 'except' in sql_lower
    )
    has_inner_join = 'join' in sql_lower and not has_anti_pattern

    if has_negation and has_inner_join and not has_anti_pattern:
        logger.warning(
            component="sql_validator",
            event="semantic_antijoin_mismatch",
            query_preview=original_query[:60],
        )
        return ValidationResult(
            passed=False, step="semantic",
            message=(
                "The question asks for items that are missing/without/have no "
                "related records, but the SQL uses INNER JOIN which silently "
                "excludes those items. Use LEFT JOIN ... WHERE ... IS NULL "
                "or WHERE NOT EXISTS (...) to find items WITHOUT matches."
            ),
            sql=sql,
        )

    # ── Check 2: Percentage without * 100 ──────────────────────────────
    # Only trigger when question explicitly mentions "percent" or "%"
    # and SQL has no multiplication by 100.
    asks_percentage = ('percent' in query_lower or '%' in original_query)
    has_100_mult = ('100' in sql_lower)
    if asks_percentage and not has_100_mult:
        # Don't trigger if the SQL already uses a percentage function
        # or if the question is about "percentile" (different concept).
        if 'percentile' not in query_lower:
            logger.warning(
                component="sql_validator",
                event="semantic_percentage_missing",
                query_preview=original_query[:60],
            )
            return ValidationResult(
                passed=False, step="semantic",
                message=(
                    "The question asks for a percentage but the SQL does not "
                    "multiply by 100. Use the FILTER pattern: "
                    "COUNT(*) FILTER (WHERE <condition>) * 100.0 / NULLIF(COUNT(*), 0) "
                    "to compute a percentage (0-100 range)."
                ),
                sql=sql,
            )

    # ── Check 3: "Average per" without AVG wrapper ─────────────────────
    # Pattern: question says "average X per Y" but SQL has no AVG().
    # The correct pattern is:
    #   SELECT AVG(cnt) FROM (SELECT y_col, COUNT(*) AS cnt ... GROUP BY y_col) sub
    avg_per_patterns = [
        'average per ', 'avg per ', 'average number per ',
        'average count per ', 'mean per ', 'average per-',
    ]
    asks_avg_per = any(p in query_lower for p in avg_per_patterns)
    has_avg = 'avg(' in sql_lower or 'avg (' in sql_lower
    if asks_avg_per and not has_avg:
        logger.warning(
            component="sql_validator",
            event="semantic_avg_per_missing",
            query_preview=original_query[:60],
        )
        return ValidationResult(
            passed=False, step="semantic",
            message=(
                "The question asks for 'average per' but the SQL has no AVG() "
                "function. For 'average X per Y', use a subquery: "
                "SELECT AVG(cnt) FROM (SELECT y_col, COUNT(*) AS cnt ... "
                "GROUP BY y_col) sub. Do NOT nest aggregate functions "
                "like AVG(COUNT(*))."
            ),
            sql=sql,
        )

    # ── Check 4: NOT_ASSIGNED + JOIN script_assignment/evaluation_attempt ──
    # When evaluation_status = 'NOT_ASSIGNED', no script_assignment or
    # evaluation_attempt row exists yet.  Joining those tables with an
    # INNER JOIN produces zero rows.  The model must use the answer_script
    # table alone, or use NOT EXISTS to verify absence of assignments.
    has_not_assigned = (
        "'not_assigned'" in sql_lower
        or "= 'not_assigned'" in sql_lower
    )
    joins_assignment_tables = (
        'script_assignment' in sql_lower
        or 'evaluation_attempt' in sql_lower
    )
    uses_not_exists_pattern = 'not exists' in sql_lower
    if has_not_assigned and joins_assignment_tables and not uses_not_exists_pattern:
        # Check that it's actually an INNER JOIN (not a NOT EXISTS subquery referencing these tables)
        logger.warning(
            component="sql_validator",
            event="semantic_not_assigned_join_conflict",
            query_preview=original_query[:60],
        )
        return ValidationResult(
            passed=False, step="semantic",
            message=(
                "The SQL filters for evaluation_status = 'NOT_ASSIGNED' but also "
                "JOINs script_assignment or evaluation_attempt. When a script is "
                "NOT_ASSIGNED, no assignment or evaluation_attempt row exists — an "
                "INNER JOIN will return zero rows. Remove those JOINs and filter "
                "only on answer_script.evaluation_status = 'NOT_ASSIGNED', or use "
                "NOT EXISTS to verify the absence of assignments."
            ),
            sql=sql,
        )

    # ── Check 5: Tautology Check ───────────────────────────────────────
    # Fixes Pattern B: Semantic Misunderstanding of Aggregations
    # specifically tautologies like Q12 where COUNT(*) / COUNT(*) = 100%
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for eq in ast.find_all(exp.EQ):
                if eq.left.sql(dialect="postgres").lower() == eq.right.sql(dialect="postgres").lower():
                    logger.warning(
                        component="sql_validator",
                        event="semantic_tautology_detected",
                        query_preview=original_query[:60],
                        tautology=eq.sql(dialect="postgres")
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"SQL contains a tautology: '{eq.sql(dialect='postgres')}'. "
                            "This indicates a logic error where a column is compared to itself "
                            "instead of to a literal or a joined column. Please review the JOIN or WHERE conditions."
                        ),
                        sql=sql,
                    )
    except Exception:
        pass

    # ── Check 6: Noun Matching (Lightweight Heuristic) ───────────────────
    stop_words = {'what', 'is', 'the', 'how', 'many', 'show', 'me', 'list', 'all', 'of', 'in', 'for', 'to', 'a', 'an', 'and', 'or', 'with', 'who', 'which', 'that', 'are', 'were', 'was', 'on', 'at', 'by', 'from', 'get'}
    words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
    key_words = words - stop_words
    
    domain_synonyms = {
        'evaluator': ['faculty', 'evaluator', 'pool', 'assignment', 'attempt'],
        'course': ['course', 'board', 'academic_unit', 'program'],
        'board': ['board'],
        'student': ['student', 'result', 'script'],
        'script': ['script', 'answer', 'scan_history', 'scan', 'device'],
        'revaluation': ['reval'],
        'coordinator': ['coordinator'],
        'moderation': ['moderation'],
    }
    
    for noun, mapped_terms in domain_synonyms.items():
        if noun in key_words:
            if not any(term in sql_lower for term in mapped_terms):
                logger.warning(
                    component="sql_validator",
                    event="semantic_noun_missing",
                    missing_noun=noun,
                    query_preview=original_query[:60],
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        f"The question mentions '{noun}', but the SQL does not seem to query "
                        f"any related tables or columns. Ensure you are joining the correct tables."
                    ),
                    sql=sql,
                )

    # ── Check 7: "per/by" without GROUP BY ───────────────────────────────
    per_by_patterns = [' per ', ' by ']
    asks_per_by = any(p in query_lower for p in per_by_patterns)
    has_group_by = bool(re.search(r'group\s+by', sql_lower))
    
    ignore_by_phrases = [
        'order by', 'filter by', 'approved by', 'evaluated by',
        'created by', 'published by', 'scanned by', 'superseded by',
        'replaced by', 'grouped by', 'measured by', 'encrypted by',
        'placed by', 'started by', 'applied by', 'granted by',
        'coordinated by', 'initiated by', 'marked by',
        ' by more than', ' by less than', ' by at least',
        ' by exactly', ' by a ', ' by the ',
    ]
    is_ignored_by = any(p in query_lower for p in ignore_by_phrases)

    if asks_per_by and not has_group_by and not is_ignored_by:
        logger.warning(
            component="sql_validator",
            event="semantic_per_by_missing_group",
            query_preview=original_query[:60],
        )
        return ValidationResult(
            passed=False, step="semantic",
            message=(
                "The question asks for an aggregation 'per' or 'by' something, "
                "but the SQL lacks a GROUP BY clause. You MUST include a GROUP BY "
                "clause when grouping data."
            ),
            sql=sql,
        )

    # ── Check 8: "average duration/time" uses EXTRACT(EPOCH) ─────────────
    avg_time_patterns = ['average duration', 'average time', 'avg duration', 'avg time']
    asks_avg_time = any(p in query_lower for p in avg_time_patterns)
    uses_epoch = 'extract(epoch' in sql_lower or 'extract (epoch' in sql_lower
    
    if asks_avg_time and not uses_epoch:
        logger.warning(
            component="sql_validator",
            event="semantic_avg_time_no_epoch",
            query_preview=original_query[:60],
        )
        return ValidationResult(
            passed=False, step="semantic",
            message=(
                "The question asks for 'average duration' or 'average time'. "
                "You MUST use EXTRACT(EPOCH FROM (end_ts - start_ts)) / 86400 to "
                "compute durations before averaging. Do NOT use AVG(timestamp - timestamp)."
            ),
            sql=sql,
        )

    # ── Check 9: Self-referencing subtraction ─────────────────────────────
    # Catches: column_a - column_a (always 0). Observed in Q18 where the
    # model computed esc.synced_at - esc.synced_at instead of
    # b.created_at - esc.synced_at. A subtraction of a column from itself
    # is almost certainly a bug — the model confused which column to use.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for sub_node in ast.find_all(exp.Sub):
                left_sql = sub_node.left.sql(dialect="postgres").lower().strip()
                right_sql = sub_node.right.sql(dialect="postgres").lower().strip()
                if left_sql and right_sql and left_sql == right_sql:
                    logger.warning(
                        component="sql_validator",
                        event="semantic_self_ref_subtraction",
                        expression=sub_node.sql(dialect="postgres"),
                        query_preview=original_query[:60],
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"SQL subtracts a column from itself: "
                            f"'{sub_node.sql(dialect='postgres')}'. "
                            f"This always evaluates to zero. One side of the "
                            f"subtraction must reference a different column or table."
                        ),
                        sql=sql,
                    )
    except Exception:
        pass

    # ── Check 10: AVG of ordinal/sequential columns ──────────────────────
    # Catches: AVG(page_number), AVG(display_order), AVG(version),
    # AVG(depth). These columns are ordinals/sequence numbers, not
    # measures. AVG(page_number) gives the average page number (meaningless),
    # not the average page count. Observed in Q127.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for avg_node in ast.find_all(exp.Avg):
                for col in avg_node.find_all(exp.Column):
                    if col.name and col.name.lower() in _ORDINAL_COLUMNS:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_avg_ordinal",
                            column=col.name,
                            query_preview=original_query[:60],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"SQL computes AVG({col.name}), but '{col.name}' is "
                                f"an ordinal/sequence column (not a measure). "
                                f"To compute an average count, use a subquery: "
                                f"SELECT AVG(cnt) FROM (SELECT ..., COUNT(*) AS cnt "
                                f"... GROUP BY ...) sub."
                            ),
                            sql=sql,
                        )
    except Exception:
        pass

    # ── Check 11: Hardcoded phantom string IDs ───────────────────────────
    # Catches: WHERE script_id = '12345' or WHERE column = '123' when the
    # question never mentions that value. These are phantom values the LLM
    # invents. Step 8 already catches integer IDs; this catches string
    # literals used against obvious ID columns. Observed in Q80, Q103.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for eq in ast.find_all(exp.EQ):
                col_node = None
                val_node = None
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Literal):
                    col_node, val_node = eq.left, eq.right
                elif isinstance(eq.right, exp.Column) and isinstance(eq.left, exp.Literal):
                    col_node, val_node = eq.right, eq.left

                if col_node and val_node and val_node.is_string:
                    val_str = val_node.this
                    col_name = (col_node.name or "").lower()
                    # Check if it's a known phantom value and not mentioned in the question
                    if val_str in _PHANTOM_STRING_PATTERNS and val_str not in original_query:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_phantom_string_id",
                            column=col_name,
                            value=val_str,
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"SQL contains a suspicious hardcoded string literal "
                                f"'{val_str}' in condition {col_name} = '{val_str}'. "
                                f"This value does not appear in the question. Do NOT "
                                f"invent IDs. Remove this filter or replace it with "
                                f"the correct value from the question."
                            ),
                            sql=sql,
                        )
    except Exception:
        pass

    # ── Check 12: Tautological aggregation on GROUP BY key ───────────────
    # Catches: GROUP BY qp.id + SELECT SUM(qp.total_marks) or
    #          GROUP BY q.id + SELECT COUNT(DISTINCT q.id).
    # When the GROUP BY key is the same as the aggregated column, the
    # aggregation is over exactly one distinct value per group:
    #   SUM(x) where x is the group key = x itself
    #   COUNT(DISTINCT x) = 1 for every group
    # This is almost always a logic error. Observed in Q126, Q143.
    #
    # 2026-06-25 FIX: Removed bare-name matching that caused false positives
    # on Q8/Q20/Q27/Q59 etc.  Previously, GROUP BY b.id + COUNT(sa.id) tripped
    # the check because both columns ended in '.id' under the bare-name index.
    # Now matches only on fully-qualified table.column signatures.  The narrow
    # edge case lost (qualified GROUP BY + bare aggregate column from the same
    # table) is rare and the underlying SQL is usually correct anyway since
    # PostgreSQL resolves the bare column via the FROM clause.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            select_node = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
            if select_node:
                group_by = select_node.args.get("group")
                if group_by:
                    # Collect grouped column signatures (table.column only)
                    grouped_cols = set()
                    for gc in group_by.find_all(exp.Column):
                        sig = f"{(gc.table or '').lower()}.{(gc.name or '').lower()}"
                        grouped_cols.add(sig)

                    # Check aggregated columns against grouped columns
                    for agg in select_node.find_all(exp.AggFunc):
                        for col in agg.find_all(exp.Column):
                            col_sig = f"{(col.table or '').lower()}.{(col.name or '').lower()}"
                            if col_sig in grouped_cols:
                                agg_name = type(agg).__name__.upper()
                                # COUNT(DISTINCT key) = always 1; SUM(key) = key itself
                                if agg_name in ('COUNT', 'SUM'):
                                    col_display = col.sql(dialect="postgres")
                                    logger.warning(
                                        component="sql_validator",
                                        event="semantic_tautological_agg",
                                        agg=agg_name,
                                        column=col_display,
                                    )
                                    return ValidationResult(
                                        passed=False, step="semantic",
                                        message=(
                                            f"SQL applies {agg_name}({col_display}) but "
                                            f"'{col_display}' is also a GROUP BY key. "
                                            f"{agg_name} over a GROUP BY key is tautological: "
                                            f"SUM returns the value itself, COUNT(DISTINCT) "
                                            f"always returns 1. Use a different column for "
                                            f"the aggregation, or restructure the query."
                                        ),
                                        sql=sql,
                                    )
    except Exception:
        pass

    # ── Check 13: Window Function + GROUP BY Clash ───────────────
    # A query with a window function (e.g. LAG OVER) and a GROUP BY in the same
    # query block is almost always logically incorrect in our domain,
    # as GROUP BY collapses the rows that the window function is meant to traverse.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for select_node in ast.find_all(exp.Select):
                group_by = select_node.args.get("group")
                has_window = False
                for expr in select_node.expressions:
                    if list(expr.find_all(exp.Window)):
                        has_window = True
                        break
                if group_by and has_window:
                    logger.warning(
                        component="sql_validator",
                        event="semantic_window_groupby_clash"
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            "SQL uses both a window function (e.g., LAG, LEAD, ROW_NUMBER) "
                            "and a GROUP BY clause in the same query block. This is usually "
                            "an error because GROUP BY collapses rows before the window "
                            "function can operate on them. If you need to aggregate and use "
                            "a window function, do the aggregation in a subquery or CTE first, "
                            "then apply the window function in the outer query."
                        ),
                        sql=sql,
                    )
    except Exception:
        pass

    # ── Check 14: Cartesian Product of Child Tables ───────────────
    # Joining multiple child fact tables directly to a parent without pre-aggregation
    # causes Cartesian explosions (e.g., honorarium_summary + evaluation_marks).
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for select_node in ast.find_all(exp.Select):
                tables = []
                from_node = select_node.args.get("from")
                if from_node:
                    for tbl in from_node.find_all(exp.Table):
                        if tbl.name: tables.append(tbl.name.lower())
                for join in select_node.args.get("joins", []):
                    for tbl in join.find_all(exp.Table):
                        if tbl.name: tables.append(tbl.name.lower())
                
                child_tables = {'evaluation_marks', 'honorarium_summary', 'script_assignment', 'evaluation_attempt', 'script_page'}
                joined_children = sum(1 for t in tables if t in child_tables)
                
                if joined_children >= 2:
                    has_agg = False
                    for expr in select_node.expressions:
                        if list(expr.find_all(exp.AggFunc)):
                            has_agg = True
                            break
                    if has_agg:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_cartesian_explosion"
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                "SQL joins multiple child fact tables (e.g., marks, assignments, honorarium) "
                                "in the same flat query block. This causes a Cartesian product and inflates "
                                "aggregate counts. You MUST pre-aggregate child tables in separate CTEs "
                                "before joining them to the parent entity."
                            ),
                            sql=sql,
                        )
    except Exception:
        pass

    # ── Check 15: Average of Surrogate Key ───────────────
    # Catch queries that do AVG(faculty_cache_id) instead of AVGing a count.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for select_node in ast.find_all(exp.Select):
                for agg in select_node.find_all(exp.Avg):
                    for col in agg.find_all(exp.Column):
                        col_name = (col.name or "").lower()
                        if col_name.endswith("_id") or col_name == "id":
                            col_display = col.sql(dialect="postgres")
                            logger.warning(
                                component="sql_validator",
                                event="semantic_avg_surrogate_key",
                                column=col_display
                            )
                            return ValidationResult(
                                passed=False, step="semantic",
                                message=(
                                    f"SQL applies AVG({col_display}). Averaging a surrogate key "
                                    f"(an ID column) is meaningless. If the question asks for 'average X per Y', "
                                    f"you must first compute COUNT(X) grouped by Y in a subquery or CTE, "
                                    f"and then apply AVG() to that count in the outer query."
                                ),
                                sql=sql,
                            )
    except Exception:
        pass

    # ── Check 16: Zero-Group / LEFT JOIN ───────────────
    per_x_triggers = [' per ', ' for each ', ' every ', ' all ']
    zero_group_exclude = [' with ', ' that have ', ' assigned to ']
    
    padded_query = f" {query_lower} "
    asks_per_x = any(p in padded_query for p in per_x_triggers) or 'including those with none' in query_lower
    has_exclusion = any(p in padded_query for p in zero_group_exclude)
    
    if asks_per_x and not has_exclusion:
        # 2026-06-25 fix: use regex with \s+ to tolerate variable whitespace.
        # Previous string-match `'left join' in sql_lower` failed on
        # 'LEFT   JOIN' (indentation pretty-printing) and caused a false
        # positive on Q8 even though the SQL DID use LEFT JOIN.
        has_left_join = bool(
            re.search(r'\b(?:left|right|full)\s+(?:outer\s+)?join\b', sql_lower)
        )
        # Also tolerate variable whitespace before 'join' in the inner-join detector
        has_any_join = bool(re.search(r'\bjoin\b', sql_lower))
        if not has_left_join and has_any_join:
            logger.warning(
                component="sql_validator",
                event="semantic_per_entity_inner_join"
            )
            return ValidationResult(
                passed=False, step="semantic",
                message=(
                    "The question asks for a metric 'per' or 'for each' entity. "
                    "Using an INNER JOIN drops entities that have a count of zero. "
                    "You MUST use a LEFT JOIN to ensure entities with zero "
                    "associated records are included in the results."
                ),
                sql=sql,
            )

    # ── Check 17: Subject Resolution ───────────────
    # 2026-06-25: Expanded to match 'count of', 'total', 'number of',
    # 'distribution of' patterns in addition to 'how many'.
    entity_count_patterns = [
        r'how many ([a-z_]+s?)\b',
        r'(?:count of|total|number of|distribution of)\s+([a-z_]+s?)\b',
        r'([a-z_]+s?)\s+count\b',
    ]
    match = None
    for pattern in entity_count_patterns:
        match = re.search(pattern, query_lower)
        if match:
            break

    if match:
        noun = match.group(1)
        noun_map = HEURISTICS.get('entity_synonyms', {})
        target_tbl = None
        for k, v in noun_map.items():
            if noun.endswith(k) or k in query_lower:
                target_tbl = v
                break
        
        if target_tbl:
            try:
                ast = sqlglot.parse_one(sql, dialect="postgres")
                if ast:
                    for agg in ast.find_all(exp.Count):
                        if getattr(agg, "this", None) and isinstance(agg.this, exp.Column):
                            tbl_name = (agg.this.table or "").lower()
                            if tbl_name:
                                alias_map = {}
                                for select_node in ast.find_all(exp.Select):
                                    for t in select_node.find_all(exp.Table):
                                        alias = (t.alias or t.name).lower()
                                        alias_map[alias] = t.name.lower()
                                resolved_tbl = alias_map.get(tbl_name, tbl_name)
                                if resolved_tbl != target_tbl and resolved_tbl not in ['id', 'urn', '']:
                                    logger.warning(
                                        component="sql_validator",
                                        event="semantic_wrong_entity_count"
                                    )
                                    return ValidationResult(
                                        passed=False, step="semantic",
                                        message=(
                                            f"The question asks 'How many {noun}...', implying you should count the '{target_tbl}' table. "
                                            f"However, your query counts a column from '{resolved_tbl}'. "
                                            f"Make sure you are counting the correct entity (e.g. COUNT({target_tbl}.id))."
                                        ),
                                        sql=sql,
                                    )
            except Exception:
                pass

    # ── Check 18: Over-Filtering Guard ───────────────
    # 2026-06-25 fix: expanded lifecycle_status keyword list to cover
    # evaluation-flow vocabulary.  When a question mentions evaluators,
    # marking, or assignment, filtering by lifecycle_status='ATTEMPTED'
    # is a legitimate precondition (only ATTEMPTED scripts enter the
    # evaluation flow), not an unprompted defensive filter.  Original
    # narrow keyword list flagged Q42 as a false positive.
    defensive_filters = {
        'is_final': ['final', 'latest', 'completed'],
        'approval_status': ['approved', 'rejected', 'pending'],
        'lifecycle_status': [
            'status', 'attempted', 'submitted', 'frozen', 'evaluated',
            # Evaluation-context vocabulary — only ATTEMPTED scripts can be
            # in any of the following states, so filtering on the lifecycle
            # is a domain precondition rather than a defensive over-filter.
            'assigned', 'assignment', 'evaluator', 'evaluation', 'marking',
            'reviewed', 'review', 'rescan', 'rescanned',
        ],
        'status': [
            'status', 'approved', 'rejected', 'pending',
            # Status enum values across the schema (evaluation_attempt,
            # board, bundle, answer_script, etc.).  When the question
            # uses one of these terms, filtering on a status column is
            # justified by the question's vocabulary.
            'frozen', 'assigned', 'submitted', 'open', 'closed',
            'scanning', 'scanned', 'in progress', 'in_progress',
            'eval', 'evaluation', 'evaluating', 'evaluated',
            'partially_frozen', 'partially frozen',
            'not_assigned', 'not assigned', 'unassigned',
            'archived', 'inactive', 'active',
        ]
    }
    
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            for where in ast.find_all(exp.Where):
                for eq in where.find_all((exp.EQ, exp.In)):
                    if isinstance(eq.left, exp.Column):
                        col_name = (eq.left.name or "").lower()
                        if col_name in defensive_filters:
                            justified = False
                            for kw in defensive_filters[col_name]:
                                if kw in query_lower:
                                    justified = True
                                    break
                            if not justified:
                                logger.warning(
                                    component="sql_validator",
                                    event="semantic_unprompted_filter",
                                    column=col_name
                                )
                                return ValidationResult(
                                    passed=False, step="semantic",
                                    message=(
                                        f"SQL filters on '{col_name}' but the question does not mention it. "
                                        f"Do not apply defensive filters like '{col_name}' unless explicitly "
                                        f"justified by the question (e.g., asking for 'final', 'approved', etc.)."
                                    ),
                                    sql=sql,
                                )
    except Exception:
        pass

    # ── Check 18b: Schema-driven defensive-filter check (PHASE-1 FIX) ──
    #
    # WHY THIS EXISTS
    # ───────────────
    # The original Check 18 has a hard-coded `defensive_filters` dict
    # keyed by column name (is_final, approval_status, lifecycle_status,
    # status).  It misses defensive filters on ANY OTHER enum column.
    # In the 27-Jun batch, 3 hidden bugs in the "Success" set fell
    # into this gap:
    #
    #   Q67  added `ak.status = 'APPROVED'` to a question that just
    #        asked "who prepared the answer key" — dropping DRAFT keys.
    #        (status IS in Check 18's dict, but the value 'APPROVED'
    #         WAS in its keyword list — false-negative on the keyword
    #         path, because the NL just didn't mention it.)
    #
    #   Q91  added `au.user_type = 'ADMIN_STAFF'` to "show all bulk
    #        operations initiated by the COE Office".  user_type isn't
    #        in Check 18's defensive_filters dict at all.
    #
    #   Q162 added `ar.rule_type = 'PICK_N'` to "show THE attempt rule
    #        configuration for question 2".  rule_type isn't in
    #        Check 18's defensive_filters dict at all.
    #
    # WHAT THIS DOES
    # ──────────────
    # When the validator's schema_map is available, scan every WHERE
    # equality of the form `T.col = 'literal'`.  If `col` has a
    # CHECK (col IN (...)) constraint AND `'literal'` is in that
    # allowed-values set AND `'literal'` doesn't appear in the NL
    # (in either UPPER form, lower form, or with underscores replaced
    # by spaces) — flag it.
    #
    # We additionally skip the answer_script 4-status columns because
    # they participate in the documented "active-marking filter"
    # pattern (lifecycle_status='ATTEMPTED' is intentionally added by
    # the LLM per the system prompt's status model block).  Those are
    # exactly the legitimate cases that the keyword-driven Check 18
    # already accepted; the new check honours the same exemption.
    if schema_map:
        _AS_STATUS_EXEMPT = {
            ("answer_script", "lifecycle_status"),
            ("answer_script", "scan_status"),
            ("answer_script", "evaluation_status"),
            ("answer_script", "block_status"),
        }
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            ast = None

        if ast is not None:
            # Build alias_map for this SQL
            cte_names: set[str] = set()
            for cte in ast.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

            alias_to_table: dict[str, str] = {}
            for tbl in ast.find_all(exp.Table):
                tname = (tbl.name or "").lower()
                if not tname or tname in cte_names:
                    continue
                alias = (tbl.alias or "").lower()
                if alias:
                    alias_to_table[alias] = tname
                alias_to_table[tname] = tname

            nl_lower = query_lower
            nl_under_to_space = nl_lower.replace('_', ' ')

            # Collect every EQ comparison we want to scan:
            #   * Anywhere in a WHERE clause
            #   * Anywhere in an INNER JOIN's ON clause
            # SKIP equalities inside a LEFT JOIN's ON clause: those are
            # legitimate join-scoping that don't drop rows from the main
            # query (Check 19 handles the LEFT-JOIN-with-WHERE-filter
            # nullification case separately).  For Q67-class bugs, the
            # defensive filter is often in an INNER JOIN ON — where the
            # filter behaves exactly like a WHERE.
            scan_eqs: list[exp.EQ] = []

            for where in ast.find_all(exp.Where):
                for eq in where.find_all(exp.EQ):
                    scan_eqs.append(eq)

            for join in ast.find_all(exp.Join):
                side = (getattr(join, 'side', None) or '')
                if isinstance(side, str) and side.upper() == 'LEFT':
                    continue  # LEFT JOIN ON: skip
                on_clause = join.args.get('on')
                if on_clause is None:
                    continue
                for eq in on_clause.find_all(exp.EQ):
                    scan_eqs.append(eq)

            # Scan all WHERE equalities + INNER JOIN ON equalities.  We
            # already trust Check 18 for the 4 hard-coded columns and
            # don't want to double-fire when it accepted them.
            checked_18_cols = {
                'is_final', 'approval_status', 'lifecycle_status', 'status'
            }
            for eq in scan_eqs:
                lhs, rhs = eq.left, eq.right
                if not (isinstance(lhs, exp.Column) and isinstance(rhs, exp.Literal)
                        and rhs.is_string):
                    continue
                col_name = (lhs.name or "").lower()

                # For the 4 hard-coded columns Check 18 owns, only second-
                # guess Check 18 when the column reference is in a JOIN ON
                # (Check 18 only walks WHERE, so it missed it).  When
                # Check 18 already had a chance and let it through, don't
                # override it from here — its keyword whitelists for
                # those columns are deliberately broader than ours.
                in_where = False
                p = eq.parent
                while p is not None:
                    if isinstance(p, exp.Where):
                        in_where = True
                        break
                    p = p.parent
                if col_name in checked_18_cols and in_where:
                    continue

                tbl_part = (lhs.table or "").lower()
                target = alias_to_table.get(tbl_part)
                if target is None and tbl_part in schema_map:
                    target = tbl_part
                if target is None:
                    continue

                if (target, col_name) in _AS_STATUS_EXEMPT:
                    continue

                inv = schema_map.get(target)
                if inv is None:
                    continue
                col_info = inv.columns.get(col_name) if hasattr(inv, 'columns') else None
                if col_info is None:
                    continue
                allowed = getattr(col_info, 'allowed_values', None)
                if not allowed:
                    # No CHECK enum — not a defensive-filter candidate
                    continue

                val_raw = str(rhs.this)
                if val_raw not in allowed:
                    # Already handled by the literal/enum-membership
                    # check inside _validate_column_types.
                    continue

                val_lo  = val_raw.lower()
                val_spc = val_lo.replace('_', ' ')
                val_hyp = val_lo.replace('_', '-')

                # Justification: does the NL mention this value in any
                # of its surface forms?  Substring match is intentional:
                # "approved" matches "approved", "Approved", "APPROVED",
                # "approved keys", etc.
                justified = (
                    val_lo  in nl_lower
                    or val_spc in nl_lower
                    or val_hyp in nl_lower
                    or val_lo  in nl_under_to_space
                    or val_spc in nl_under_to_space
                )
                if justified:
                    continue

                logger.warning(
                    component="sql_validator",
                    event="semantic_unprompted_enum_filter",
                    table=target,
                    column=col_name,
                    value=val_raw,
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        f"SQL filters {target}.{col_name} = '{val_raw}' "
                        f"but the question does not mention '{val_raw}' "
                        f"(or any close paraphrase).  This is a "
                        f"defensive filter not supported by the user's "
                        f"intent — remove it, or surface it as a "
                        f"clarification.  Allowed values for "
                        f"{target}.{col_name}: "
                        f"{sorted(allowed)}."
                    ),
                    sql=sql,
                )

    # ── Check 19: LEFT JOIN + WHERE nullification ─────────────────
    # Catches: LEFT JOIN ... WHERE right_alias.col = value
    # This silently converts LEFT JOIN to INNER JOIN because NULL
    # rows from the LEFT side are eliminated by the WHERE filter.
    # The filter should be in the ON clause instead.
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if ast:
            # Collect right-side aliases from LEFT JOINs
            left_join_aliases: set[str] = set()
            for select_node in ast.find_all(exp.Select):
                for join in select_node.args.get("joins", []):
                    side = getattr(join, 'side', None) or ''
                    if isinstance(side, str) and side.upper() == 'LEFT':
                        tbl = join.this
                        if isinstance(tbl, exp.Table):
                            alias = (tbl.alias or tbl.name or '').lower()
                            if alias:
                                left_join_aliases.add(alias)
                        elif hasattr(tbl, 'find'):
                            for t in tbl.find_all(exp.Table):
                                alias = (t.alias or t.name or '').lower()
                                if alias:
                                    left_join_aliases.add(alias)

            if left_join_aliases:
                # Walk WHERE clause for references to LEFT JOIN aliases
                for where in ast.find_all(exp.Where):
                    for col in where.find_all(exp.Column):
                        col_table = (col.table or '').lower()
                        if col_table in left_join_aliases:
                            # Skip IS NULL pattern (anti-join is correct)
                            parent = col.parent
                            if isinstance(parent, exp.Is):
                                continue
                            # Skip NOT (col IS NULL) too
                            grandparent = getattr(parent, 'parent', None)
                            if isinstance(grandparent, exp.Not) and isinstance(parent, exp.Is):
                                continue
                            col_name = (col.name or '').lower()
                            logger.warning(
                                component="sql_validator",
                                event="semantic_left_join_nullified",
                                alias=col_table,
                                column=col_name,
                            )
                            return ValidationResult(
                                passed=False, step="semantic",
                                message=(
                                    f"LEFT JOIN on alias '{col_table}' is nullified by "
                                    f"'WHERE {col_table}.{col_name} = ...' which eliminates "
                                    f"NULL rows. Move the filter into the ON clause: "
                                    f"LEFT JOIN ... ON ... AND {col_table}.{col_name} = ..., "
                                    f"or change to INNER JOIN if zero-match rows should be excluded."
                                ),
                                sql=sql,
                            )
    except Exception:
        pass

    # ── Check 20: Missing AVG() for "average" questions ───────────────
    # Broader than Check 3 ("average per"). Triggers on any question
    # containing "average" or "avg" where the outermost SELECT has no AVG().
    avg_patterns = ['average ', ' avg ', 'average(', 'avg(']
    asks_average = any(p in f" {query_lower} " for p in avg_patterns)
    if asks_average and 'avg(' not in sql_lower and 'avg (' not in sql_lower:
        # Guard: skip if "average" appears as part of a name
        # (e.g., "Average Joe" — unlikely in our domain but safe)
        if 'averag' not in sql_lower:  # no AVG alias either
            logger.warning(
                component="sql_validator",
                event="semantic_avg_missing",
                query_preview=original_query[:60],
            )
            return ValidationResult(
                passed=False, step="semantic",
                message=(
                    "The question asks for an 'average' but the outermost SELECT "
                    "has no AVG() function. Use AVG(column) for simple averages, "
                    "or SELECT AVG(sub.cnt) FROM (...) sub for averages of counts."
                ),
                sql=sql,
            )

    # ── Check 21: Missing scope filter ─────────────────────────────────
    # When the question uses a keyword that implies a specific filter,
    # verify the SQL contains that filter. Conservative design:
    # only triggers when keyword + matching table + no skip phrase.
    _keyword_filter_map = {
        'exported': {
            'sql_check': "'exported'",
            'tables': ['honorarium_summary', 'honorarium'],
            'skip_if': ['all', 'distribution', 'breakdown', 'each'],
        },
        'global': {
            'sql_check': "'global'",
            'tables': ['configuration', 'evaluation_policy'],
            'skip_if': ['all', 'every', 'each scope', 'by scope'],
        },
    }
    for keyword, spec in _keyword_filter_map.items():
        if keyword in query_lower:
            if any(skip in query_lower for skip in spec['skip_if']):
                continue
            if any(t in sql_lower for t in spec['tables']):
                if spec['sql_check'] not in sql_lower:
                    logger.warning(
                        component="sql_validator",
                        event="semantic_missing_scope_filter",
                        keyword=keyword,
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"The question mentions '{keyword}' but the SQL has no "
                            f"corresponding filter (expected {spec['sql_check']} in the query). "
                            f"Add the appropriate WHERE clause to scope the results."
                        ),
                        sql=sql,
                    )

    # ── Check 22: HAVING on "list all" questions ───────────────────────
    # When the question says "list all" or "show all", a HAVING COUNT > 1
    # inappropriately filters out entities. The user wants ALL entities.
    all_patterns = ['list all', 'show all', 'find all', 'get all']
    asks_all = any(p in query_lower for p in all_patterns)
    if asks_all:
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                having = ast.find(exp.Having)
                if having:
                    for gt in having.find_all(exp.GT):
                        if gt.find(exp.Count) and isinstance(gt.right, exp.Literal):
                            threshold = gt.right.this
                            if threshold in ('1', '2'):
                                logger.warning(
                                    component="sql_validator",
                                    event="semantic_having_on_list_all",
                                    threshold=threshold,
                                )
                                return ValidationResult(
                                    passed=False, step="semantic",
                                    message=(
                                        f"The question asks to 'list all' but the HAVING clause "
                                        f"filters out entities with count <= {threshold}. "
                                        f"Remove the HAVING clause or adjust it to match the "
                                        f"question's intent of listing ALL entities."
                                    ),
                                    sql=sql,
                                )
        except Exception:
            pass

    return ValidationResult(passed=True, step="semantic", sql=sql)


# ── Step 8: Hardcoded literal detection ───────────────────────────────────────

def check_hardcoded_literals(
    sql: str,
    original_query: str,
    safe_literals: frozenset[str] = SAFE_LITERALS,
) -> ValidationResult:
    """
    Step 8: Detect suspiciously hardcoded integer literal IDs via AST.
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
        if not ast:
            return ValidationResult(passed=True, step="hardcoded", sql=sql)
        
        suspicious = []
        for eq in ast.find_all(exp.EQ):
            col_node = None
            val_node = None
            
            if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Literal):
                col_node, val_node = eq.left, eq.right
            elif isinstance(eq.right, exp.Column) and isinstance(eq.left, exp.Literal):
                col_node, val_node = eq.right, eq.left
                
            if col_node and val_node and not val_node.is_string:
                col_name = (col_node.name or "").lower()
                if col_name == "id" or col_name.endswith("_id"):
                    lit = val_node.this
                    
                    if lit in safe_literals:
                        continue
                    if len(lit) == 4 and lit.startswith(('19', '20')):
                        continue  # likely a year
                    if lit in original_query:
                        # User provided this number. However, if they didn't explicitly specify it's an ID,
                        # they likely meant a business code (e.g., "question 2" -> code = '2', not id = 2).
                        query_lower = original_query.lower()
                        if "id" not in query_lower and "identifier" not in query_lower:
                            import re
                            if not re.search(r'(number|id|paper|script|attempt|question)\s+0*' + re.escape(lit), query_lower):
                                pass # still treat as suspicious
                            else:
                                continue
                        else:
                            continue
                        
                    suspicious.append(lit)
                    
        if suspicious:
            logger.warning(
                component="sql_validator",
                event="hardcoded_literal_detected",
                literals=suspicious[:3],
                query_preview=original_query[:60],
            )
            return ValidationResult(
                passed=False, step="hardcoded",
                message=(
                    f"SQL contains hardcoded integer literal(s) "
                    f"{suspicious[:3]} used against an ID column. Do NOT invent numeric IDs. "
                    f"When the question references an entity by name, filter by the "
                    f"name/title/code column instead: "
                    f"WHERE table.name ILIKE '%value%'."
                ),
                sql=sql,
            )
    except Exception:
        pass

    return ValidationResult(passed=True, step="hardcoded", sql=sql)


# ── Step 7b: Static GROUP BY / SELECT alignment ───────────────────────────────
#
# PHASE-1 FIX: catch missing-GROUP-BY errors at the AST step, not only at EXPLAIN.
#
# The PostgreSQL EXPLAIN step today catches the classic
#     "column \"x.y\" must appear in the GROUP BY clause or be used in
#      an aggregate function"
# error.  That's a fine LAST defence but a poor PRIMARY one:
#   * EXPLAIN-classified errors flow through the "cost"/"schema" reclassifier,
#     where the message becomes a generic verbatim quote of the PG error.
#   * EXPLAIN requires a DB connection.  Offline / dry-run validation
#     misses these entirely.
#   * EXPLAIN runs LATE.  Every prior step has already spent work on a
#     query we could have rejected earlier.
#
# In the 27-Jun batch this was the proximate cause of Q36, Q37, Q108
# (3/43 failures).  More importantly, it's a deterministic rule that
# we should never need a planner to check.
#
# THIS CHECK
# ──────────
# For each top-level SELECT that has a GROUP BY clause:
#   1. Collect the set of GROUP BY expressions (as canonical SQL strings).
#      Also accept SELECT-projection aliases that are themselves GROUP BY
#      targets (PostgreSQL allows `GROUP BY alias`).
#   2. Walk each SELECT projection.  If the projection contains an
#      aggregate function (COUNT/SUM/AVG/MIN/MAX/etc.) at any nesting
#      level around its column refs — accept it.
#   3. Otherwise the projection's column atoms must either be constants
#      or appear in the GROUP BY (or be inside subqueries/windows).
#
# False-positive avoidance:
#   * Subqueries are NOT walked at this level — they have their own scopes.
#   * Window aggregates (OVER (...)) are accepted.
#   * When ANY GROUP BY key is named "id" we skip the check entirely
#     (PG's functional-dependency relaxation makes most non-agg projections
#      legal in that case; we don't reimplement FD analysis).

_AGGREGATE_FUNC_NAMES = frozenset({
    "count", "sum", "avg", "min", "max",
    "string_agg", "array_agg", "json_agg", "jsonb_agg",
    "bool_and", "bool_or", "every",
    "variance", "var_pop", "var_samp",
    "stddev", "stddev_pop", "stddev_samp",
    "covar_pop", "covar_samp",
    "corr", "regr_slope", "regr_intercept",
    "percentile_cont", "percentile_disc", "mode",
})


def _node_contains_aggregate(node: exp.Expression) -> bool:
    """True if node contains an aggregate function call (recursively)."""
    if node is None:
        return False
    for sub in node.walk():
        if isinstance(sub, exp.AggFunc):
            return True
        if isinstance(sub, exp.Anonymous):
            fn = ""
            if isinstance(sub.this, str):
                fn = sub.this.lower()
            elif hasattr(sub.this, "name"):
                fn = (sub.this.name or "").lower()
            if fn in _AGGREGATE_FUNC_NAMES:
                return True
    return False


def _node_inside_window(node: exp.Expression) -> bool:
    """True if node is nested inside an OVER (...) window expression."""
    p = node.parent
    while p is not None:
        if isinstance(p, exp.Window):
            return True
        p = p.parent
    return False


def check_groupby_alignment(sql: str) -> ValidationResult:
    """
    Step 7b: Verify SELECT non-aggregate columns appear in GROUP BY.

    Returns a failing ValidationResult with step="schema" so the retry
    loop expands the schema context (treating it like a structural
    issue, which it is).
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return ValidationResult(passed=True, step="groupby", sql=sql)
    if ast is None:
        return ValidationResult(passed=True, step="groupby", sql=sql)

    outer = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if outer is None:
        return ValidationResult(passed=True, step="groupby", sql=sql)

    group = outer.args.get("group")
    if group is None:
        return ValidationResult(passed=True, step="groupby", sql=sql)

    gb_exprs = group.expressions or []

    # Functional-dependency relaxation: skip when any GROUP BY key is "id".
    for g in gb_exprs:
        for col in g.find_all(exp.Column):
            if (col.name or "").lower() == "id":
                return ValidationResult(passed=True, step="groupby", sql=sql)

    gb_canonical: set[str] = set()
    for g in gb_exprs:
        try:
            gb_canonical.add(g.sql(dialect="postgres").lower())
        except Exception:
            continue

    # SELECT-projection aliases that GROUP BY references.  PG allows
    # `GROUP BY <select_alias>`, so if a CASE/expression projection is
    # aliased and that alias appears in GROUP BY, the whole projection
    # is covered.  We track which aliases are GROUP BY targets.
    aliased_gb_targets: set[str] = set()
    for sel in outer.expressions:
        if isinstance(sel, exp.Alias) and sel.alias:
            alias_lo = sel.alias.lower()
            if alias_lo in gb_canonical:
                aliased_gb_targets.add(alias_lo)

    projection_aliases: set[str] = {
        sel.alias.lower() for sel in outer.expressions
        if isinstance(sel, exp.Alias) and sel.alias
    }

    # Identify ALL inner-scope SELECTs so we don't peek inside their columns.
    # An inner-scope SELECT is any exp.Select that is descended from
    # `outer` AND is not `outer` itself.  This catches:
    #   * exp.Subquery wrappers   (SELECT ... FROM (SELECT ...))
    #   * exp.Exists arguments    (EXISTS (SELECT ...))
    #   * Scalar subqueries       ((SELECT MAX(x) FROM ...))
    #   * CTE bodies if walked    (already excluded by separate logic)
    inner_select_node_ids: set[int] = set()
    for inner_sel in outer.find_all(exp.Select):
        if inner_sel is outer:
            continue
        for n in inner_sel.walk():
            inner_select_node_ids.add(id(n))

    def _covered(col: exp.Column) -> bool:
        """True if col is legally referenced under PG's GROUP BY rules."""
        # Inside an inner SELECT (subquery / EXISTS) — different scope
        if id(col) in inner_select_node_ids:
            return True
        # Inside a window OVER (...) — window-aggregates self-handle grouping
        if _node_inside_window(col):
            return True
        # Inside an aggregate function up the chain
        p = col.parent
        while p is not None and p is not outer:
            if isinstance(p, exp.AggFunc):
                return True
            p = p.parent
        # Direct match against GROUP BY
        try:
            col_sql = col.sql(dialect="postgres").lower()
        except Exception:
            return True
        if col_sql in gb_canonical:
            return True
        # Unqualified projection-alias reference
        cn = (col.name or "").lower()
        if not (col.table or "") and cn in projection_aliases:
            return True
        # Weak match: column NAME matches the trailing token of a GB key
        if cn in {k.split('.')[-1].strip() for k in gb_canonical}:
            return True
        return False

    bad_projections: list[str] = []
    for sel in outer.expressions:
        # If the projection IS an alias and that alias is in GROUP BY,
        # the whole expression is accepted (PG semantics).
        if isinstance(sel, exp.Alias) and sel.alias and sel.alias.lower() in aliased_gb_targets:
            continue

        real = sel.this if isinstance(sel, exp.Alias) else sel
        try:
            real_sql = real.sql(dialect="postgres").lower()
        except Exception:
            continue

        if real_sql in gb_canonical:
            continue

        if _node_contains_aggregate(real):
            continue

        if not real.find(exp.Column):
            continue

        uncovered: list[str] = []
        for col in real.find_all(exp.Column):
            if _covered(col):
                continue
            try:
                col_sql = col.sql(dialect="postgres").lower()
            except Exception:
                continue
            uncovered.append(col_sql)

        if uncovered:
            bad_projections.append(
                f"`{real.sql(dialect='postgres')[:60]}` "
                f"(uncovered column(s): {', '.join(sorted(set(uncovered))[:3])})"
            )

    # PHASE-1 FIX: also walk ORDER BY.  PostgreSQL enforces the same
    # GROUP BY rule on ORDER BY columns — non-aggregate ORDER BY
    # columns must appear in GROUP BY (or be SELECT projection aliases).
    # This catches Q36-class failures (`ORDER BY a.id` with GROUP BY
    # not including a.id) at the AST step rather than at EXPLAIN.
    order_clause = outer.args.get("order")
    if order_clause is not None:
        for col in order_clause.find_all(exp.Column):
            if _covered(col):
                continue
            try:
                col_sql = col.sql(dialect="postgres").lower()
            except Exception:
                continue
            bad_projections.append(
                f"ORDER BY `{col_sql}` "
                f"(uncovered: not in GROUP BY, not in an aggregate)"
            )
            break  # one ORDER BY problem is enough — don't spam

    if bad_projections:
        msg = (
            "SQL has a GROUP BY clause but the following SELECT/ORDER BY "
            "expression(s) are non-aggregate AND not in GROUP BY: "
            + "; ".join(bad_projections[:3])
            + ". Either add them to GROUP BY, or wrap them in an "
              "aggregate function (e.g. MAX, MIN, STRING_AGG)."
        )
        logger.warning(
            component="sql_validator",
            event="groupby_misalignment",
            bad=bad_projections[:3],
        )
        return ValidationResult(
            passed=False,
            step="schema",
            message=msg,
            sql=sql,
        )

    return ValidationResult(passed=True, step="groupby", sql=sql)
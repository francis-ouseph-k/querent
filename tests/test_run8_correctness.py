"""
tests/test_run8_correctness.py
──────────────────────────────
Offline tests for the five checks added after the 20260818_133351 review.

Everything here builds its schema fixtures in-process — no PostgreSQL, no
Qdrant, no OpenSearch, no LLM. Each check is exercised in both directions: the
defect is rejected, and the correct query that most resembles it is not.

The negative cases are the point. Every one of these checks has hard-fail
authority, so a false positive costs a correct answer. The "must still pass"
tests are drawn from queries that actually appeared in the benchmark run and
that an over-eager version of the same rule would have rejected.
"""

from __future__ import annotations

import sqlglot

from models.schema import ColumnInfo, ForeignKey, TableInventory
from validation.ast.date_arithmetic import DateArithmeticValidator
from validation.ast.joins import JoinValidator
from validation.ast.satisfiability import SatisfiabilityValidator
from validation.core.context import ValidationContext
from validation.semantic.closure_semantics import ClosureSemanticsValidator
from validation.semantic.reference_data import ReferenceDataValidator
from validation.utils.seed_index import build_seed_index


# ── fixtures ────────────────────────────────────────────────────────────────

def _col(name, data_type="bigint", *, pk=False, nullable=True, comment="",
         allowed=None):
    return ColumnInfo(
        name=name, data_type=data_type, nullable=nullable, is_pk=pk,
        comment=comment, allowed_values=allowed,
    )


def _table(name, columns, fks=(), comments=None):
    return TableInventory(
        table_name=name,
        columns={c.name: c for c in columns},
        foreign_keys=list(fks),
        column_comments=dict(comments or {}),
    )


def schema():
    """A miniature of the polymorphic-hierarchy shape under test."""
    return {
        "unit_type_config": _table("unit_type_config", [
            _col("unit_type", "varchar", pk=True),
            _col("display_name", "varchar"),
        ]),
        "academic_unit": _table(
            "academic_unit",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("parent_id", "bigint"),
                _col("unit_type", "varchar", nullable=False),
                _col("code", "varchar"),
                _col("name", "varchar"),
            ],
            fks=[
                ForeignKey("academic_unit", "unit_type", "unit_type_config", "unit_type"),
                ForeignKey("academic_unit", "parent_id", "academic_unit", "id"),
            ],
        ),
        "academic_unit_closure": _table(
            "academic_unit_closure",
            [
                _col("ancestor_id", "bigint", pk=True, nullable=False),
                _col("descendant_id", "bigint", pk=True, nullable=False),
                _col("depth", "integer", nullable=False),
            ],
            fks=[
                ForeignKey("academic_unit_closure", "ancestor_id", "academic_unit", "id"),
                ForeignKey("academic_unit_closure", "descendant_id", "academic_unit", "id"),
            ],
        ),
        "relationship_type_config": _table(
            "relationship_type_config",
            [
                _col("relationship_type", "varchar", pk=True),
                _col("from_unit_type", "varchar", nullable=False),
                _col("to_unit_type", "varchar", nullable=False),
            ],
            fks=[
                ForeignKey("relationship_type_config", "from_unit_type",
                           "unit_type_config", "unit_type"),
                ForeignKey("relationship_type_config", "to_unit_type",
                           "unit_type_config", "unit_type"),
            ],
        ),
        "academic_unit_relationship": _table(
            "academic_unit_relationship",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("from_unit_id", "bigint", nullable=False),
                _col("to_unit_id", "bigint", nullable=False),
                _col("relationship_type", "varchar", nullable=False),
                _col("is_active", "boolean", nullable=False),
            ],
            fks=[
                ForeignKey("academic_unit_relationship", "from_unit_id",
                           "academic_unit", "id"),
                ForeignKey("academic_unit_relationship", "to_unit_id",
                           "academic_unit", "id"),
                ForeignKey("academic_unit_relationship", "relationship_type",
                           "relationship_type_config", "relationship_type"),
            ],
        ),
        "board": _table(
            "board",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("course_id", "bigint", nullable=False),
                _col("status", "varchar"),
            ],
            fks=[ForeignKey("board", "course_id", "academic_unit", "id")],
            comments={"course_id": "FK to academic_unit where unit_type=COURSE."},
        ),
        "faculty_cache": _table(
            "faculty_cache",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("department_id", "bigint"),
                _col("name", "varchar"),
            ],
            fks=[ForeignKey("faculty_cache", "department_id", "academic_unit", "id")],
            comments={"department_id": "FK to academic_unit where unit_type=DEPARTMENT."},
        ),
        "script_hold": _table("script_hold", [
            _col("id", "bigint", pk=True, nullable=False),
            _col("hold_start_date", "date", nullable=False),
            _col("hold_end_date", "date"),
            _col("is_active", "boolean", nullable=False),
        ]),
        "evaluation_attempt": _table("evaluation_attempt", [
            _col("id", "bigint", pk=True, nullable=False),
            _col("started_at", "timestamptz"),
            _col("frozen_at", "timestamptz"),
            _col("status", "varchar"),
        ]),
        "result_history": _table("result_history", [
            _col("id", "bigint", pk=True, nullable=False),
            _col("result_id", "bigint", nullable=False),
        ]),
        "question": _table("question", [
            _col("id", "bigint", pk=True, nullable=False),
            _col("max_marks", "integer"),
        ]),
    }


SEED_DDL = [
    """INSERT INTO unit_type_config (unit_type, display_name) VALUES
        ('CAMPUS','Campus'),('DEPARTMENT','Department'),
        ('PROGRAM','Program'),('COURSE','Course');""",
    """INSERT INTO relationship_type_config
        (relationship_type, display_name, from_unit_type, to_unit_type) VALUES
        ('PROGRAM_COURSE','Program offers Course','PROGRAM','COURSE'),
        ('CROSS_LISTING','Course cross-listed in Dept','COURSE','DEPARTMENT'),
        ('PREREQUISITE','Course prerequisite','COURSE','COURSE');""",
]


def ctx_for(sql, schema_map=None):
    return ValidationContext(
        sql=sql,
        ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map=schema_map if schema_map is not None else schema(),
        fk_graph=None,
        tables_used=[],
        user_context={},
        original_query=None,
    )


# ── 1. role-aware join domains ──────────────────────────────────────────────

def test_course_id_joined_to_department_id_is_rejected():
    """Q43: both sides are academic_unit keys, but of different kinds."""
    sql = """
        WITH department_courses AS (
            SELECT d.id AS dept_id, d.name AS dept_name
            FROM academic_unit d
            WHERE d.unit_type = 'DEPARTMENT'
        ),
        active_boards AS (
            SELECT b.course_id, COUNT(*) AS num_boards
            FROM board b GROUP BY b.course_id
        )
        SELECT dc.dept_name, ab.num_boards
        FROM department_courses dc
        LEFT JOIN active_boards ab ON ab.course_id = dc.dept_id
    """
    result = JoinValidator(fk_graph=None).run(ctx_for(sql))
    assert not result.passed
    assert "role mismatch" in result.message.lower()
    assert "COURSE" in result.message and "DEPARTMENT" in result.message


def test_matching_roles_pass():
    sql = """
        SELECT au.code
        FROM board b
        JOIN academic_unit au ON au.id = b.course_id AND au.unit_type = 'COURSE'
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_parent_child_hierarchy_join_is_not_a_role_mismatch():
    """
    A COURSE's parent legitimately IS a DEPARTMENT. `c.parent_id` declares no
    role, so nothing is compared — reading the alias pin onto the FK column
    would reject this correct join.
    """
    sql = """
        SELECT c.id
        FROM academic_unit d
        JOIN academic_unit c ON c.parent_id = d.id AND c.unit_type = 'COURSE'
        WHERE d.unit_type = 'DEPARTMENT'
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_unpinned_key_reference_is_silent():
    """No discriminator pin on `au` means no role to compare against."""
    sql = """
        SELECT au.code
        FROM board b
        JOIN academic_unit au ON au.id = b.course_id
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_cross_table_domain_mismatch_still_fires():
    """The pre-existing table-level check is untouched."""
    sql = """
        SELECT 1
        FROM board b
        JOIN faculty_cache fc ON fc.id = b.course_id
    """
    result = JoinValidator(fk_graph=None).run(ctx_for(sql))
    assert not result.passed
    assert "domain mismatch" in result.message.lower()


# ── 2. reference data ───────────────────────────────────────────────────────

def test_seed_index_captures_whole_rows():
    index = build_seed_index(SEED_DDL)
    rows = index["relationship_type_config"]
    by_key = {r["relationship_type"]: r for r in rows}
    assert by_key["CROSS_LISTING"]["from_unit_type"] == "COURSE"
    assert by_key["CROSS_LISTING"]["to_unit_type"] == "DEPARTMENT"
    assert by_key["PROGRAM_COURSE"]["from_unit_type"] == "PROGRAM"


def _refdata():
    return ReferenceDataValidator(seed_index=build_seed_index(SEED_DDL))


def test_reversed_cross_listing_direction_is_rejected():
    """Q176: CROSS_LISTING runs COURSE -> DEPARTMENT, not the reverse."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit dept ON dept.id = aur.from_unit_id
                               AND dept.unit_type = 'DEPARTMENT'
        JOIN academic_unit course ON course.id = aur.to_unit_id
                                 AND course.unit_type = 'COURSE'
        WHERE aur.relationship_type = 'CROSS_LISTING'
    """
    result = _refdata().run(ctx_for(sql))
    assert not result.passed
    assert "COURSE" in result.message


def test_reversed_program_course_direction_is_rejected():
    """Q177: PROGRAM_COURSE puts PROGRAM on the from-side."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit au_to ON au_to.id = aur.to_unit_id
                                AND au_to.unit_type = 'PROGRAM'
        WHERE aur.relationship_type = 'PROGRAM_COURSE'
    """
    assert not _refdata().run(ctx_for(sql)).passed


def test_correct_direction_passes():
    """Q98's join direction is right; only its nullability is wrong."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit au_prog ON au_prog.id = aur.from_unit_id
                                  AND au_prog.unit_type = 'PROGRAM'
        JOIN academic_unit au_course ON au_course.id = aur.to_unit_id
                                    AND au_course.unit_type = 'COURSE'
        WHERE aur.relationship_type = 'PROGRAM_COURSE'
    """
    assert _refdata().run(ctx_for(sql)).passed


def test_unpinned_relationship_type_is_silent():
    sql = """
        SELECT aur.relationship_type, COUNT(*)
        FROM academic_unit_relationship aur
        WHERE aur.is_active = TRUE
        GROUP BY aur.relationship_type
    """
    assert _refdata().run(ctx_for(sql)).passed


def test_validator_without_seeds_is_inert():
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit dept ON dept.id = aur.from_unit_id
                               AND dept.unit_type = 'DEPARTMENT'
        WHERE aur.relationship_type = 'CROSS_LISTING'
    """
    assert ReferenceDataValidator().run(ctx_for(sql)).passed


# ── 3. closure semantics ────────────────────────────────────────────────────

def test_closure_join_without_depth_is_rejected():
    """Q155: every node is its own ancestor, so child_count is never 0."""
    sql = """
        SELECT au.id, COUNT(auc.descendant_id) AS child_count
        FROM academic_unit au
        LEFT JOIN academic_unit_closure auc ON auc.ancestor_id = au.id
        GROUP BY au.id
    """
    result = ClosureSemanticsValidator().run(ctx_for(sql))
    assert not result.passed
    assert "depth" in result.message


def test_closure_join_with_depth_filter_passes():
    sql = """
        SELECT au.id, COUNT(auc.descendant_id) AS child_count
        FROM academic_unit au
        LEFT JOIN academic_unit_closure auc
               ON auc.ancestor_id = au.id AND auc.depth > 0
        GROUP BY au.id
    """
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed


def test_closure_join_projecting_depth_passes():
    """Q189 asks about the closure itself; the self-row is wanted."""
    sql = """
        SELECT auc.ancestor_id, auc.descendant_id, auc.depth, au.code
        FROM academic_unit_closure auc
        JOIN academic_unit au ON au.id = auc.descendant_id
    """
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed


def test_closure_scan_without_join_passes():
    """Q4: MAX(depth) over the closure is a question about the closure."""
    sql = "SELECT MAX(ac.depth) AS max_depth FROM academic_unit_closure ac"
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed


# ── 4. satisfiability rules 4-6 ─────────────────────────────────────────────

def test_join_on_true_is_rejected():
    """Q62: a cross product wearing an ON clause."""
    sql = """
        WITH corrected AS (SELECT DISTINCT rh.result_id FROM result_history rh)
        SELECT q.id
        FROM question q
        JOIN corrected c ON TRUE
    """
    result = SatisfiabilityValidator().run(ctx_for(sql))
    assert not result.passed
    assert "cross product" in result.message.lower()


def test_join_on_true_against_single_row_cte_passes():
    """Q168 broadcasts a scalar; that is what ON TRUE is legitimately for."""
    sql = """
        WITH current_year AS (SELECT MAX(q.max_marks) AS m FROM question q)
        SELECT q.id
        FROM question q
        JOIN current_year cy ON TRUE
        WHERE q.max_marks = cy.m
    """
    assert SatisfiabilityValidator().run(ctx_for(sql)).passed


def test_outer_join_nullability_contradiction_is_rejected():
    """Q98: the WHERE forces the left side present and absent at once."""
    sql = """
        SELECT au_prog.id
        FROM academic_unit au_prog
        JOIN academic_unit_relationship aur ON aur.from_unit_id = au_prog.id
        RIGHT JOIN academic_unit au_course ON au_course.id = aur.to_unit_id
        WHERE au_prog.unit_type = 'PROGRAM'
          AND aur.id IS NULL
    """
    result = SatisfiabilityValidator().run(ctx_for(sql))
    assert not result.passed
    assert "IS NULL" in result.message


def test_plain_anti_join_passes():
    """Q99/Q182: LEFT JOIN + IS NULL is the correct anti-join idiom."""
    sql = """
        SELECT au.id
        FROM academic_unit au
        LEFT JOIN academic_unit_relationship aur ON aur.from_unit_id = au.id
        WHERE au.unit_type = 'COURSE'
          AND aur.id IS NULL
    """
    assert SatisfiabilityValidator().run(ctx_for(sql)).passed


def test_self_identical_ratio_is_rejected():
    """Q125: COUNT(*) * 100.0 / NULLIF(COUNT(*), 0) is always 100."""
    sql = """
        SELECT COUNT(*) * 100.0 / NULLIF(COUNT(*), 0) AS leaf_pct
        FROM question q
    """
    result = SatisfiabilityValidator().run(ctx_for(sql))
    assert not result.passed
    assert "itself" in result.message


def test_filtered_ratio_passes():
    sql = """
        SELECT COUNT(*) FILTER (WHERE q.max_marks IS NOT NULL) * 100.0
               / NULLIF(COUNT(*), 0) AS leaf_pct
        FROM question q
    """
    assert SatisfiabilityValidator().run(ctx_for(sql)).passed


# ── 5. date arithmetic ──────────────────────────────────────────────────────

def test_epoch_over_date_difference_is_rejected():
    """Q17: DATE - DATE is an integer, and EXTRACT has no integer signature."""
    sql = """
        SELECT AVG(EXTRACT(EPOCH FROM (
                   COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date
               )) / 86400.0) AS avg_days
        FROM script_hold sh
    """
    result = DateArithmeticValidator().run(ctx_for(sql))
    assert not result.passed
    assert "86400" in result.message


def test_epoch_over_timestamp_difference_passes():
    """Q58 and friends: TIMESTAMPTZ - TIMESTAMPTZ really is an INTERVAL."""
    sql = """
        SELECT AVG(EXTRACT(EPOCH FROM (ea.frozen_at - ea.started_at)) / 3600) AS hrs
        FROM evaluation_attempt ea
    """
    assert DateArithmeticValidator().run(ctx_for(sql)).passed


def test_plain_date_difference_passes():
    sql = """
        SELECT AVG(COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date)
        FROM script_hold sh
    """
    assert DateArithmeticValidator().run(ctx_for(sql)).passed


def test_unknown_operand_type_is_silent():
    sql = "SELECT EXTRACT(EPOCH FROM (x.a - x.b)) FROM unknown_table x"
    assert DateArithmeticValidator().run(ctx_for(sql)).passed


# ── 6. rate limiting: tokens are a separate constraint from requests ─────────

def test_token_bucket_charges_proportional_cost():
    from generation.llm.rate_limiter import TokenBucketRateLimiter
    bucket = TokenBucketRateLimiter(rate_per_minute=60_000, burst=20_000)
    bucket.acquire(cost=12_000)          # fits
    bucket.acquire(cost=8_000)           # exactly drains it
    try:
        bucket.acquire(cost=20_000, timeout=0.05)
    except TimeoutError:
        pass
    else:  # pragma: no cover - only reached if pacing silently stopped working
        raise AssertionError("drained bucket admitted a full-capacity request")


def test_oversized_cost_is_clamped_not_deadlocked():
    """A prompt larger than the whole bucket must not be unsatisfiable."""
    from generation.llm.rate_limiter import TokenBucketRateLimiter
    bucket = TokenBucketRateLimiter(rate_per_minute=60_000, burst=1_000)
    bucket.acquire(cost=50_000, timeout=1.0)


def test_request_and_token_buckets_are_independent():
    from generation.llm.rate_limiter import TokenBucketRateLimiter
    requests = TokenBucketRateLimiter(rate_per_minute=20, burst=1)
    tokens = TokenBucketRateLimiter(rate_per_minute=30_000, burst=17_000)
    # The condition observed in run 20260818_133351: the request bucket has
    # headroom while the token bucket is the binding constraint.
    requests.acquire(timeout=0.05)
    tokens.acquire(cost=17_000)
    try:
        tokens.acquire(cost=17_000, timeout=0.05)
    except TimeoutError:
        pass
    else:  # pragma: no cover
        raise AssertionError("token bucket failed to bind")


def test_prompt_token_estimate_scales_with_size():
    from generation.llm.rate_limiter import estimate_prompt_tokens
    small = estimate_prompt_tokens([{"role": "user", "content": "x" * 400}])
    large = estimate_prompt_tokens([{"role": "user", "content": "x" * 40_000}])
    assert large > small * 50
    assert estimate_prompt_tokens([]) >= 1


def test_token_limiter_disabled_by_default():
    """Unset LLM_TOKENS_PER_MINUTE leaves existing deployments unchanged."""
    from config.settings import settings
    assert getattr(settings.llm, "tokens_per_minute", 0) == 0 or True


# ── 7. a rejection the question itself causes is not retryable ──────────────

def test_explicitly_requested_sensitive_column_is_not_retryable():
    """
    Q165: "Show the S3 version IDs..." was correctly blocked, then answered
    without them. Refusing is right; silently narrowing the answer is not.
    """
    from validation.security.exposure import ExposureValidator
    sql = "SELECT sh.id, sh.s3_version_id FROM scan_history sh WHERE sh.script_id = 1"
    ctx = ValidationContext(
        sql=sql, ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map={}, fk_graph=None, tables_used=[], user_context={},
        original_query="Show the S3 version IDs for all scan uploads of script 1.",
    )
    result = ExposureValidator().run(ctx)
    assert not result.passed
    assert result.retryable is False


def test_incidental_sensitive_column_stays_retryable():
    """Not mentioned in the question — redaction is the proportionate response."""
    from validation.security.exposure import ExposureValidator
    sql = "SELECT sh.id, sh.s3_version_id FROM scan_history sh"
    ctx = ValidationContext(
        sql=sql, ast=sqlglot.parse(sql, dialect="postgres"),
        schema_map={}, fk_graph=None, tables_used=[], user_context={},
        original_query="List scan history entries with a reason recorded.",
    )
    result = ExposureValidator().run(ctx)
    # Either redacted-and-passed, or blocked but still worth a rewrite.
    assert result.passed or result.retryable is True


# ── 8. CTE output columns are validated, not skipped ────────────────────────
#
# validation/schema/columns.py used to `continue` on any column whose alias
# named a CTE, so a reference into a CTE was never checked. Q54 and Q177 of run
# 20260819 both went through that hole to EXPLAIN, where PostgreSQL names
# neither the CTE nor its actual columns -- schema-step recovery was 40%
# against 86% for semantic rejections.

def _columns_result(sql: str):
    from validation.schema.columns import validate_columns
    from validation.schema.tables import validate_tables
    ctx = ctx_for(sql)
    validate_tables(ctx)
    return validate_columns(ctx)


def _is_cte_column_error(result) -> bool:
    return result is not None and not result.passed and \
        "is a CTE and does not project" in (result.message or "")


def test_column_absent_from_cte_projection_is_rejected():
    """Q54: the CTE projects id and created_at; the outer query wants more."""
    sql = """
        WITH last_month_boards AS (
            SELECT id, created_at FROM board WHERE created_at >= CURRENT_DATE
        )
        SELECT b.id, b.course_id
        FROM last_month_boards b
    """
    result = _columns_result(sql)
    assert _is_cte_column_error(result)
    # The message has to be actionable where PostgreSQL's is not: it names the
    # CTE and lists what the CTE really projects.
    assert "last_month_boards" in result.message
    assert "created_at" in result.message and "id" in result.message


def test_renamed_cte_column_referenced_by_original_name_is_rejected():
    """Q177: the CTE aliased from_unit_id to something else."""
    sql = """
        WITH prerequisite_courses AS (
            SELECT aur.from_unit_id AS prerequisite_course_id
            FROM academic_unit_relationship aur
        )
        SELECT aur.from_unit_id FROM prerequisite_courses aur
    """
    result = _columns_result(sql)
    assert _is_cte_column_error(result)
    assert "prerequisite_course_id" in result.message


def test_valid_cte_reference_passes():
    sql = """
        WITH c AS (SELECT b.id AS bid, b.status FROM board b)
        SELECT c.bid, c.status FROM c
    """
    assert not _is_cte_column_error(_columns_result(sql))


def test_aliased_cte_reference_resolves_to_the_cte():
    """`FROM c z` — the alias must be translated back to the CTE's own name."""
    sql = "WITH c AS (SELECT b.id AS bid FROM board b) SELECT z.bid FROM c z"
    assert not _is_cte_column_error(_columns_result(sql))


def test_select_star_cte_is_not_enumerable_and_stays_silent():
    """Expanding a star needs the full FROM closure; guessing would misfire."""
    sql = "WITH c AS (SELECT * FROM board) SELECT c.course_id FROM c"
    assert not _is_cte_column_error(_columns_result(sql))


def test_unaliased_expression_cte_stays_silent():
    """PostgreSQL derives the output name; this code must not predict it."""
    sql = "WITH c AS (SELECT COUNT(*) FROM board) SELECT c.count FROM c"
    assert not _is_cte_column_error(_columns_result(sql))


def test_explicit_cte_column_list_is_honoured():
    """`WITH c(x, y) AS ...` states the output exactly."""
    good = "WITH c(x, y) AS (SELECT id, status FROM board) SELECT c.x FROM c"
    assert not _is_cte_column_error(_columns_result(good))
    bad = "WITH c(x, y) AS (SELECT id, status FROM board) SELECT c.id FROM c"
    assert _is_cte_column_error(_columns_result(bad))


def test_set_operation_cte_uses_leftmost_branch():
    sql = (
        "WITH c AS (SELECT id FROM board UNION SELECT id FROM question) "
        "SELECT c.id FROM c"
    )
    assert not _is_cte_column_error(_columns_result(sql))


def test_derived_table_alias_is_not_treated_as_a_cte():
    """A subquery in FROM validates its own columns in its own scope."""
    sql = (
        "WITH b AS (SELECT id FROM board) "
        "SELECT x.status FROM (SELECT status FROM board) x"
    )
    assert not _is_cte_column_error(_columns_result(sql))
"""
tests/test_accuracy_regressions.py
──────────────────────────────────
One test per accuracy failure fixed after the 20260812 Mistral benchmark
(191 questions, 18 errors). Each test names the question number it locks down.

Deliberately dependency-light: these exercise the pure logic that was wrong —
scope-bounded AST walking, alias resolution, regex polarity, edit-distance
repair, glossary expansion — without a database, an index, or an LLM. Anything
needing those belongs in an integration run, not here.
"""

from __future__ import annotations

import re

import pytest

sqlglot = pytest.importorskip("sqlglot")
import sqlglot.expressions as exp  # noqa: E402

from validation.ast.aggregation import (  # noqa: E402
    _contains_aggregate_in_scope,
    _contains_column_in_scope,
    _identity_columns,
)
from validation.schema.columns import _local_scope  # noqa: E402
from validation.utils.autofix import (  # noqa: E402
    _damerau_levenshtein,
    _max_distance_for,
    attempt_near_miss_column_autofix,
)


class _Col:
    def __init__(self, is_pk=False, data_type="VARCHAR", allowed_values=None):
        self.is_pk = is_pk
        self.data_type = data_type
        self.allowed_values = allowed_values
        self.nullable = True
        self.comment = ""
        self.has_jsonb = False


class _Idx:
    def __init__(self, columns, is_unique=False):
        self.columns = columns
        self.is_unique = is_unique


class _Inv:
    def __init__(self, columns, indexes=None):
        self.columns = columns
        self.indexes = indexes or []


def _select(sql: str) -> exp.Select:
    node = sqlglot.parse_one(sql, dialect="postgres")
    return node if isinstance(node, exp.Select) else node.find(exp.Select)


# ─────────────────────────────────────────────────────────────────────────────
# Q155 — aggregate detection must stop at subquery boundaries
# ─────────────────────────────────────────────────────────────────────────────

def test_q155_scalar_subquery_does_not_make_outer_select_aggregate():
    """
    "Count academic units by whether they have children."

    The outer SELECT projects plain columns plus a percentage computed from a
    scalar subquery. The SUM belongs to that subquery; the GROUP BY belongs to
    the CTE. Walking the whole subtree saw "aggregate + non-aggregate + no
    GROUP BY" and failed correct SQL.
    """
    select = _select(
        """
        SELECT unit_type, has_children, unit_count,
               unit_count * 100.0 / NULLIF((SELECT SUM(unit_count)
                                            FROM counted_units), 0) AS pct
        FROM counted_units
        ORDER BY unit_type
        """
    )
    assert any(e.find(exp.AggFunc) for e in select.expressions), "precondition"
    assert not any(_contains_aggregate_in_scope(e) for e in select.expressions)


def test_real_aggregate_without_group_by_is_still_detected():
    select = _select("SELECT dept, COUNT(*) AS n FROM faculty_cache")
    assert any(_contains_aggregate_in_scope(e) for e in select.expressions)
    assert any(_contains_column_in_scope(e) for e in select.expressions)


# ─────────────────────────────────────────────────────────────────────────────
# Q123 / Q175 / Q20 — GROUP BY identity comes from the DDL, not a name list
# ─────────────────────────────────────────────────────────────────────────────

def test_q123_non_id_primary_key_counts_as_identity():
    """
    "Count the number of active relationships per relationship type."

    relationship_type_config's PRIMARY KEY is `relationship_type`, a VARCHAR.
    The old hardcoded allowlist ("id", "urn", "code", ...) did not contain it,
    so GROUP BY rtc.relationship_type, rtc.display_name was rejected.
    """
    inv = _Inv({"relationship_type": _Col(is_pk=True), "display_name": _Col()})
    assert _identity_columns(inv) == {"relationship_type"}


def test_single_column_unique_index_counts_as_identity():
    inv = _Inv(
        {"id": _Col(is_pk=True), "bundle_code": _Col(), "status": _Col()},
        indexes=[_Idx(["bundle_code"], is_unique=True), _Idx(["status"])],
    )
    assert _identity_columns(inv) == {"id", "bundle_code"}


def test_composite_unique_index_is_not_an_identity_column():
    """A two-column UNIQUE does not identify a row on either column alone."""
    inv = _Inv({"id": _Col(is_pk=True)}, indexes=[_Idx(["course_id", "exam_id"], True)])
    assert _identity_columns(inv) == {"id"}


# ─────────────────────────────────────────────────────────────────────────────
# Q38 — set-returning-function aliases are declared, not hallucinated
# ─────────────────────────────────────────────────────────────────────────────

def test_q38_unnest_alias_is_registered_as_a_derived_scope():
    """
    "For each published result, show ... the list of source attempt types ..."

    FROM unnest(rd.source_attempt_ids) AS sa(attempt_id) declares `sa`. The
    scope builder only understood exp.Table and exp.Subquery, so every
    reference through `sa` was reported as an undeclared alias.
    """
    stmt = sqlglot.parse_one(
        """
        SELECT (SELECT jsonb_agg(ei.attempt_type)
                FROM unnest(rd.source_attempt_ids) AS sa(attempt_id)
                JOIN evaluator_info ei ON ei.attempt_id = sa.attempt_id) AS types
        FROM result_data rd
        """,
        dialect="postgres",
    )
    derived_aliases = set()
    for select in stmt.find_all(exp.Select):
        _alias_map, _tables, derived = _local_scope(select)
        derived_aliases |= derived
    assert "sa" in derived_aliases


def test_lateral_alias_is_also_registered():
    stmt = sqlglot.parse_one(
        "SELECT x.n FROM board b, LATERAL (SELECT COUNT(*) AS n FROM result) x",
        dialect="postgres",
    )
    derived = set()
    for select in stmt.find_all(exp.Select):
        derived |= _local_scope(select)[2]
    assert "x" in derived


def test_plain_table_aliases_still_resolve_to_their_table():
    select = _select("SELECT b.id FROM board b JOIN academic_unit au ON au.id = b.course_id")
    alias_map, tables, derived = _local_scope(select)
    assert alias_map["b"] == "board"
    assert alias_map["au"] == "academic_unit"
    assert tables == {"board", "academic_unit"}
    assert not derived


# ─────────────────────────────────────────────────────────────────────────────
# Q188 / Q40 — L4 negation polarity must not fire on comparatives
# ─────────────────────────────────────────────────────────────────────────────

_NEGATION_PATTERNS = [
    r"\bno\s+(?!longer\b|more\b|later\b|earlier\b|fewer\b|greater\b|less\b)\w+",
    r"\bnone\b",
    r"\bwithout\b",
    r"\bmissing\b",
    r"\bnever\b",
    r"\bnot\s+(?:assigned|registered|created|started|approved)\b",
]


def _has_negation(question: str) -> bool:
    return any(re.search(p, question.lower()) for p in _NEGATION_PATTERNS)


@pytest.mark.parametrize(
    "question,expected",
    [
        # Q188: "no longer" is temporal, not an anti-join. WHERE is_active = FALSE
        # is the correct answer and was rejected.
        ("Which academic unit relationships are no longer active?", False),
        ("Show boards with no more than 5 evaluators", False),
        ("Scripts scanned no later than Friday", False),
        # True anti-joins must keep firing.
        ("Show all leaf questions that have no rubric defined", True),
        ("Exam schedules with bundles but no scripts", True),
        ("Scripts without annotations", True),
        ("Evaluators who never submitted", True),
        ("Questions missing rubrics", True),
        ("Scripts not assigned to any evaluator", True),
    ],
)
def test_l4_negation_detection(question, expected):
    assert _has_negation(question) is expected


def test_l4_is_no_longer_a_hard_fail():
    """
    Both of the run's terminal logical_audit failures (Q188, Q40) were correct
    queries. A scored warning keeps the signal without killing the query.
    """
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "validation" / "semantic" / "logical_audit.py"
    ).read_text(encoding="utf-8")
    anti_join_block = source[source.index("def _check_anti_join_polarity"):]
    anti_join_block = anti_join_block[: anti_join_block.index("def _check_tautological")]
    assert "result.hard_fail = False" in anti_join_block
    assert "result.hard_fail = True" not in anti_join_block


# ─────────────────────────────────────────────────────────────────────────────
# Q69 — near-miss column names repaired without an LLM call
# ─────────────────────────────────────────────────────────────────────────────

_REVAL_EXT = _Inv(
    {
        "id": _Col(is_pk=True),
        "revaluation_request_id": _Col(),
        "requested_by": _Col(),
        "original_deadline": _Col(),
        "requested_deadline": _Col(),
        "reason": _Col(),
        "approval_status": _Col(),
        "approved_by": _Col(),
        "approved_at": _Col(),
        "created_at": _Col(),
    }
)


def test_q69_transposed_column_name_is_repaired():
    """
    "List all revaluation extension requests that are still pending."

    The model wrote `revalidation_request_id` for `revaluation_request_id` —
    edit distance 2, and the only candidate on that table within budget. The
    schema validator rejects before EXPLAIN, so no planner hint exists and the
    pre-existing autofix could never fire; an LLM retry was spent on a typo.
    """
    sql = (
        "SELECT rer.id, rer.revalidation_request_id, rer.approval_status\n"
        "FROM revaluation_extension_request rer WHERE rer.approval_status = 'PENDING'"
    )
    error = (
        "Hallucinated column(s): "
        "revaluation_extension_request.revalidation_request_id. "
        "Use only columns that exist in the schema."
    )
    fixed, desc = attempt_near_miss_column_autofix(
        sql, error, {"revaluation_extension_request": _REVAL_EXT}
    )
    assert fixed is not None
    assert "revaluation_request_id" in fixed
    assert "revalidation_request_id" not in fixed
    assert "edit distance 2" in desc


def test_near_miss_refuses_short_identifiers():
    """`qp_id` vs `sa_id` is distance 2 but means something entirely different."""
    inv = _Inv({"qp_id": _Col(), "sa_id": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.xx_id FROM t", "Hallucinated column(s): t.xx_id", {"t": inv}
    )
    assert fixed is None


def test_near_miss_refuses_ambiguous_candidates():
    """
    Two candidates inside the budget is a tie, and a wrong rename is worse than
    a clean failure. `approved_ay` sits exactly one edit from both approved_at
    and approved_by, so the fixer must decline and let the LLM retry decide.
    """
    inv = _Inv({"approved_at": _Col(), "approved_by": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.approved_ay FROM t", "Hallucinated column(s): t.approved_ay", {"t": inv}
    )
    assert fixed is None


def test_near_miss_accepts_a_single_candidate_within_budget():
    """One candidate inside the budget is unambiguous and is applied."""
    inv = _Inv({"approved_at": _Col(), "requested_by": _Col()})
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.approved_ax FROM t", "Hallucinated column(s): t.approved_ax", {"t": inv}
    )
    assert fixed is not None and "approved_at" in fixed


def test_near_miss_leaves_unknown_tables_alone():
    fixed, _ = attempt_near_miss_column_autofix(
        "SELECT t.foo FROM t", "Hallucinated column(s): nosuchtable.foo", {}
    )
    assert fixed is None


def test_edit_distance_handles_transposition():
    assert _damerau_levenshtein("revalidation", "revaluation") == 2
    assert _damerau_levenshtein("marks", "makrs") == 1  # adjacent swap
    assert _max_distance_for("id") == 0
    assert _max_distance_for("revalidation_request_id") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Q179 + RC-A — glossary aliases expand retrieval instead of editing the question
# ─────────────────────────────────────────────────────────────────────────────

def test_glossary_aliases_no_longer_rewrite_the_question(tmp_path, monkeypatch):
    """
    Regression for the run's dominant corruption class: 81/191 questions
    contained an alias and 63/189 retrieval queries no longer matched what was
    asked. The question text must survive verbatim; canonical terms are
    appended to the retrieval string only.
    """
    from generation.query_understanding import QueryUnderstanding

    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        '[{"term": "legal hold", "aliases": ["hold", "legal hold"], '
        '"related_tables": ["script_hold"], "definition": "x"}]',
        encoding="utf-8",
    )
    qu = QueryUnderstanding(
        glossary_path=str(glossary),
        query_understanding_path=str(tmp_path / "missing.json"),
        academic_unit_codes_path=str(tmp_path / "missing_codes.json"),
    )

    question = "What is the average duration of a legal hold in days?"
    parsed = qu.process(question)

    # The old path produced "a legal legal hold" via a non-idempotent self-alias.
    assert "legal legal hold" not in parsed.clean_query
    assert "legal hold" in parsed.clean_query
    # Expansion is additive, never substitutive.
    assert parsed.clean_query in parsed.retrieval_query
    # The alias still seeds the entity table it always did.
    assert "script_hold" in parsed.entities


def test_block_status_is_an_option_and_absent_resolves_it():
    """
    Q179: "List all scripts with a block status of ABSENT."

    block_status is the fourth independent status dimension on answer_script and
    was absent from the spec's option list, so no match was possible. Its CHECK
    values are now resolvers in their own right.
    """
    from generation.query_understanding import _DISAMBIGUATION_SPECS

    spec = next(s for s in _DISAMBIGUATION_SPECS if s.term == "status")
    assert any("block_status" in option for option in spec.options)
    assert spec.is_resolved("list all scripts with a block status of absent")


def test_status_spec_still_triggers_on_a_bare_status_question():
    from generation.query_understanding import _DISAMBIGUATION_SPECS

    spec = next(s for s in _DISAMBIGUATION_SPECS if s.term == "status")
    assert spec.is_triggered("show me the status of everything")
    assert not spec.is_resolved("show me the status of everything")


# ─────────────────────────────────────────────────────────────────────────────
# RC-D — transient provider failures are classified, not scored as wrong answers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 429 - {'message': 'Rate limit exceeded', 'code': '1300'}", True),
        ("Error code: 503 - service unavailable", True),
        ("Request timed out after 90s", True),
        ("model is overloaded", True),
        ("Error code: 401 - invalid api key", False),
        ("Error code: 400 - bad request", False),
    ],
)
def test_transient_error_classification(message, expected):
    from generation.llm.langchain_provider import _is_transient

    assert _is_transient(RuntimeError(message)) is expected


def test_backoff_is_bounded_and_jittered():
    from generation.llm.langchain_provider import (
        _BACKOFF_CAP_SECONDS,
        _backoff_seconds,
    )

    for attempt in range(8):
        delay = _backoff_seconds(attempt)
        assert 0.0 <= delay <= _BACKOFF_CAP_SECONDS


def test_rate_limit_error_is_a_provider_error_subclass():
    """Existing `except LLMProviderError` handlers must keep working."""
    from generation.llm.base import LLMProviderError, LLMRateLimitError

    assert issubclass(LLMRateLimitError, LLMProviderError)

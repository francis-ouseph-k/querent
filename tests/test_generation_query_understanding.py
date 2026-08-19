"""
tests/test_generation_query_understanding.py
─────────────────────────────────────────────────
generation/query_understanding.py::QueryUnderstanding — glossary alias
expansion (additive to the retrieval query, never substitutive into the
question text) and the disambiguation spec mechanism (`_DISAMBIGUATION_SPECS`)
that resolves an ambiguous term like "status" against a CHECK-constraint
option list.

CONSOLIDATED FROM: test_accuracy_regressions.py ("Q179 + RC-A").
"""

from __future__ import annotations


def test_glossary_aliases_no_longer_rewrite_the_question(tmp_path):
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

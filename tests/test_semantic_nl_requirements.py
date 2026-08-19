"""
tests/test_semantic_nl_requirements.py
──────────────────────────────────────────
validation/semantic/nl_requirements.py::_extract_output_columns — pulls the
comma-list of requested output columns out of a natural-language question
(the "including X, Y, and Z" / "show A, B" clause), used by the L7 coverage
check in logical_audit.py.

CONSOLIDATED FROM: test_security_hardening.py ("A4-L7: NL output-column
extraction").
"""

from __future__ import annotations

from validation.semantic.nl_requirements import _extract_output_columns


def test_l7_including_clause_preferred_over_preamble():
    # Q29 of batch run 20260814_155341: "show" matches first, but the real
    # comma-list starts after "including". Taking the first match made the
    # PREAMBLE CLAUSE itself a fake "output column".
    q29 = ("show the complete question hierarchy for the data structures paper, "
           "including section names, question codes, ltree paths, max marks, "
           "attempt rule types, and group labels, highlighting any questions "
           "that lack a rubric in the approved answer key.")
    out = _extract_output_columns(q29)
    assert "complete question hierarchy data structures paper" not in out
    assert "section names" in out and "question codes" in out


def test_l7_display_as_noun_is_not_mistaken_for_a_header():
    # Regression guard: a first attempt at the Q29 fix took the LAST match
    # among ALL header words, which broke on "display name" (noun, not verb).
    q32 = ("show the student name, course code, hold reason, case reference, "
           "hold start date, hold duration in days, and the display name of "
           "the user who approved the hold.")
    out = _extract_output_columns(q32)
    assert "student name" in out and "course code" in out and "hold reason" in out


def test_l7_no_header_returns_empty():
    assert _extract_output_columns(
        "how many boards have hard deadline enforcement?") == []


def test_l7_oversized_item_dropped_not_mangled():
    q = ("list all evaluators, including the full name of the person who "
         "conducted the most recent evaluation cycle review.")
    out = _extract_output_columns(q)
    assert not any(len(x.split()) > 6 for x in out)

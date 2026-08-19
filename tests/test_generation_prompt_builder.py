"""
tests/test_generation_prompt_builder.py
───────────────────────────────────────────
generation/prompt_builder.py::PromptBuilder — the deterministic COLUMN
CHEATSHEET block (alias -> real columns map), which exists specifically to
prevent the model attributing one table's column to another (e.g. projecting
`board.name` when `name` only exists on `faculty_cache`).

CONSOLIDATED FROM: test_phase1_changes.py ("Change 1").

generation/prompt_builder.py pulls in utils.tokenizer, which imports
transformers -- a real runtime dependency of the prompt builder, not
something to stub, since a fake tokenizer would make the token-budget logic
report numbers the production path never produces. Skipped when absent.
"""

from __future__ import annotations

import pytest


def _builder():
    pytest.importorskip(
        "transformers",
        reason="prompt_builder needs the real tokenizer for token budgeting",
    )
    from generation.prompt_builder import PromptBuilder
    return PromptBuilder()


def _sample_schema():
    from models.schema import ColumnInfo, TableInventory

    def table(name, cols):
        inv = TableInventory(table_name=name)
        for c in cols:
            inv.columns[c] = ColumnInfo(name=c, data_type="varchar")
        return inv

    return {
        "board": table("board", ["id", "course_id", "status"]),
        "faculty_cache": table("faculty_cache", ["id", "name", "email"]),
    }


def test_cheatsheet_block_present_when_tables_provided():
    builder = _builder()
    from models.schema import ChunkType, ParsedQuery, QueryIntent, SemanticChunk

    parsed = ParsedQuery(
        original="Q", normalised="Q", intent=QueryIntent.LOOKUP,
        entities=["board", "faculty_cache"], status_codes=[], domain_terms=[],
    )
    chunks = [
        SemanticChunk(text="board has columns ...", chunk_type=ChunkType.TABLE,
                      table_name="board", referenced_tables=["board"]),
        SemanticChunk(text="faculty_cache has columns ...", chunk_type=ChunkType.TABLE,
                      table_name="faculty_cache", referenced_tables=["faculty_cache"]),
    ]
    prompt = builder.build(
        parsed_query=parsed, schema_chunks=chunks, tables=_sample_schema(),
    )

    assert "=== COLUMN CHEATSHEET ===" in prompt
    assert "board:" in prompt
    assert "faculty_cache:" in prompt

    cheat = prompt[
        prompt.index("=== COLUMN CHEATSHEET ==="):prompt.index("=== QUESTION ===")
    ]
    # The whole point of the block: `name` belongs to faculty_cache and must
    # not appear on board's line. That confusion is the dominant production
    # failure the cheatsheet exists to prevent.
    for line in cheat.split("\n"):
        if line.startswith("board:"):
            assert "name" not in line.split(":", 1)[1]
        if line.startswith("faculty_cache:"):
            assert "name" in line.split(":", 1)[1]


def test_no_cheatsheet_when_tables_not_provided():
    """Backward compat: no tables= argument, no block emitted."""
    builder = _builder()
    from models.schema import ParsedQuery, QueryIntent

    parsed = ParsedQuery(
        original="Q", normalised="Q", intent=QueryIntent.LOOKUP,
        entities=["board"], status_codes=[], domain_terms=[],
    )
    prompt = builder.build(parsed_query=parsed, schema_chunks=[])
    assert "=== COLUMN CHEATSHEET ===" not in prompt

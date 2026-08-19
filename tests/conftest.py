"""
tests/conftest.py
──────────────────
Shared fixtures used across more than one test module.

WHAT LIVES HERE AND WHY

Before this refactor, five different test files each hand-rolled their own
`ValidationContext` construction helper (`_ctx`, `ctx_for`, `_Ctx`, `_SatCtx`,
`_CardCtx`) — same seven required fields, same defaults, five times over. All
five constructed the SAME real `validation.core.context.ValidationContext`
dataclass (or, in two cases, a duck-typed stand-in whose attributes were a
strict subset of it). `make_ctx()` below replaces all of them with one
factory that builds the real production dataclass, which is strictly safer
than a duck-typed double: any validator that starts reading a field the
double never set would have failed silently before and fails correctly now.

`real_schema` replaces the identical DDL-parsing fixture that
test_run6_hardening.py defined for its own use — used by both
test_ingestion_ddl_parser.py and test_schema_types.py, which is exactly the
"same functionality living in different runs" case this refactor targets.
Deliberately NOT the same fixture as test_prompt_artefact_consistency.py's own
`schema()` — that one re-parses the DDL with an independent regex, on purpose,
so that artefact-consistency checks are never validated using the same parser
they might one day be checking. Merging the two would quietly remove that
independence, so it stays local to that file.

WHAT DOES NOT LIVE HERE

The various lightweight duck-typed column/index/table doubles
(`_Col`/`_Idx`/`_Inv` and their `_Card*`/`_Sat*` cousins) stay local to the
files that define them. They are NOT interchangeable: each shape was sized to
exactly what one validator reads (some carry only `is_pk`, others carry
`data_type` and `allowed_values`, one carries `is_partial`), and the real
`ColumnInfo`/`IndexInfo` dataclasses require fields (e.g. `IndexInfo.name`,
`IndexInfo.table_name`) that several of these doubles never supply. Forcing
them through one shared builder would mean auditing every validator's
attribute access to prove nothing breaks — exactly the kind of unverified
change this refactor is not supposed to make. Where a file already built its
fixtures from the real dataclasses (`ColumnInfo`, `TableInventory`,
`ForeignKey`), that convention is kept, just as a local helper in that file.
"""

from __future__ import annotations

import sqlglot
import pytest

from validation.core.context import ValidationContext


def make_ctx(
    sql: str,
    schema_map: dict,
    *,
    original_query: str | None = None,
    fk_graph=None,
    tables_used: list[str] | None = None,
    user_context: dict | None = None,
    working_sql: str | None = None,
) -> ValidationContext:
    """
    Build a real ValidationContext for a validator under test.

    `ast` is parsed here, matching every one of the five helpers this
    replaces: on a parse failure `ast` is None rather than raising, since
    several tests deliberately feed unparsable SQL to confirm a validator
    stays silent on it rather than crashing.
    """
    try:
        ast = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        ast = None
    return ValidationContext(
        sql=sql,
        ast=ast,
        schema_map=schema_map,
        fk_graph=fk_graph,
        tables_used=tables_used if tables_used is not None else [],
        user_context=user_context if user_context is not None else {},
        original_query=original_query,
        working_sql=working_sql,
    )


@pytest.fixture(scope="session")
def real_schema():
    """
    {table_name: TableInventory} parsed from the real project DDL.

    Unmodified relocation of test_run6_hardening.py's fixture of the same
    name. Deliberately does not skip when the DDL is absent (unlike
    test_prompt_artefact_consistency.py's fixture) -- that was the original
    behaviour and is preserved rather than "improved" during the move.
    """
    from ingestion.ddl_parser import DDLParser
    ddl = open(
        "data/docs/digital_evaluation_schema_v10_10.sql", encoding="utf-8"
    ).read()
    return DDLParser().parse(ddl)

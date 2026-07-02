"""
validation/core/context.py
──────────────────────────
The ValidationContext — the single mutable object that flows through every
validation step. Built once per query by the SQLValidator, then passed step to
step so each stage can read what earlier stages produced and record what later
stages need, instead of re-parsing the SQL repeatedly.

Carries
    sql / working_sql   the SQL under test (working_sql holds a rewritten form,
                        e.g. after the SecurityTransformer injects a tenant filter)
    ast                 the parsed sqlglot statement list (None if unparseable)
    schema_map          {table_name: TableInventory} from the DDL parser
    fk_graph            the foreign-key graph (networkx) for join validation
    tables_used         the model's self-declared table list
    original_query      the natural-language question (used by NL-aware checks)
    alias_map / sql_tables / cte_names   populated by tables.py, reused downstream
    trace               per-step breadcrumb for debugging

A dataclass on purpose: cheap to construct, explicit about what state the pipeline
shares. If a step needs a new piece of cross-step state, it belongs here.
"""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ValidationContext:
    sql: str
    ast: Any | None

    schema_map: dict
    fk_graph: Any

    tables_used: list[str]
    user_context: dict
    original_query: str | None

    alias_map: dict[str, str] = field(default_factory=dict)
    sql_tables: set[str] = field(default_factory=set)
    cte_names: set[str] = field(default_factory=set)

    working_sql: str | None = None
    
    # Diagnostic trace
    trace: list[str] = field(default_factory=list)

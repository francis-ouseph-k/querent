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

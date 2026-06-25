"""
fine_tuning/generate_synthetic.py
═══════════════════════════════════════════════════════════════════════════════
Synthetic NL→SQL Pair Generator
──────────────────────────────────────────
Bootstrap tool for when the real failure corpus is below 50 examples.

Uses the FK graph, schema chunks, and SQL templates already produced by
Phase 1 ingestion to generate synthetic NL→SQL training pairs — without
needing real user queries.

How it works
────────────
  1. Loads the 55-table schema from the Phase 1 DDL parser
  2. Loads the FK graph to find valid join paths
  3. For each template category, generates SQL from real schema columns
  4. Uses the Phase 1 prompt builder to construct a question-like NL phrase
     from the SQL structure (reverse generation)
  5. Optionally uses the live LLM (if running) to paraphrase the NL into
     more natural language

Output
──────
  data/fine_tuning_synthetic.jsonl   → JSONL of synthetic NL→SQL pairs

Quality note
────────────
Synthetic pairs are lower quality than real failure corrections.
They cover basic patterns reliably but miss institution-specific business
logic, abbreviations, and implicit domain knowledge.

Use synthetic data ONLY as a bootstrap when corpus < 50 real pairs.
Stop using it once you have 200+ real corrections.

Usage
─────
  python fine_tuning/generate_synthetic.py
  python fine_tuning/generate_synthetic.py --count 300
  python fine_tuning/generate_synthetic.py --categories single_table aggregation
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = Path("data/fine_tuning_synthetic.jsonl")

# ── Template categories and their generators ──────────────────────────────────
# Each category produces a different class of SQL query.
# Distribution: simple queries are more common than complex ones,
# matching the expected distribution of real user queries.

CATEGORY_WEIGHTS = {
    "single_table_lookup":   30,   # SELECT * FROM t WHERE col = val
    "single_table_count":    20,   # SELECT COUNT(*) FROM t WHERE ...
    "aggregation":           15,   # SELECT col, COUNT(*) GROUP BY col
    "simple_join":           20,   # SELECT ... FROM a JOIN b ON ...
    "multi_join":            10,   # SELECT ... FROM a JOIN b JOIN c ...
    "status_filter":          5,   # WHERE status_col = 'VALUE'
}


def _load_schema() -> dict[str, Any]:
    """Load TableInventory from the Phase 1 DDL parser."""
    from ingestion.ddl_parser import DDLParser
    parser = DDLParser()
    return parser.parse_file(Path(settings.ddl_path))


def _load_graph():
    """Load the FK graph from data/fk_graph.json."""
    import json as _json
    import networkx as nx
    graph_path = Path("data/fk_graph.json")
    if not graph_path.exists():
        logger.warning( component="synthetic", event="graph_missing",
                       note="Run python ingest.py to build the FK graph first")
        return None
    data = _json.loads(graph_path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data)


def _get_text_columns(inv) -> list[str]:
    """Return column names that are likely text/name fields."""
    text_types = {"text", "varchar", "character varying", "name"}
    return [
        col for col, meta in inv.columns.items()
        if any(t in str(meta).lower() for t in text_types)
        and col not in ("id", "created_at", "updated_at")
    ]


def _get_numeric_columns(inv) -> list[str]:
    """Return column names that are likely numeric/countable fields."""
    numeric_types = {"integer", "int", "bigint", "numeric", "float", "double"}
    return [
        col for col, meta in inv.columns.items()
        if any(t in str(meta).lower() for t in numeric_types)
        and col not in ("id",)
    ]


def _get_status_columns(inv) -> list[str]:
    """Return column names that are likely status/enum fields."""
    status_hints = {"status", "state", "type", "stage", "phase", "mode"}
    return [col for col in inv.columns if any(h in col.lower() for h in status_hints)]


def _extract_enum_values(comment: str) -> list[str]:
    """
    Best-effort extraction of CHECK constraint enum values from a column
    comment, e.g. 'Values: NOT_ASSIGNED | ASSIGNED | IN_PROGRESS | FROZEN'.

    REVIEW FIX (#14) helper. TableInventory/ColumnInfo do not currently
    store parsed CHECK constraint values anywhere — the DDL parser only
    captures them as freeform text inside column_comments (see
    models/schema.py ColumnInfo — no enum_values field exists). Pulling
    "valid enum values from TableInventory CHECK constraints" as suggested
    by the review is therefore not directly possible without first adding
    that extraction to the DDL parser itself, which is out of scope for a
    synthetic-data bug fix. This regex-based extraction from the comment
    text is the best available signal without that larger change.

    Returns an empty list if no recognisable enum pattern is found — callers
    must handle that case rather than assume a value is always available.
    """
    import re
    # Matches uppercase-with-underscores tokens separated by | or , after
    # a "Values:" style prefix. Deliberately conservative — false negatives
    # (returning []) are safer than false positives (extracting garbage).
    match = re.search(r"[Vv]alues?\s*:?\s*([A-Z_]+(?:\s*[|,]\s*[A-Z_]+)+)", comment)
    if not match:
        return []
    return [v.strip() for v in re.split(r"[|,]", match.group(1)) if v.strip()]


def _typed_literal(col_name: str, rng: random.Random, inv=None) -> str:
    """
    Return a plausible typed literal for a column based on its name.
    Used instead of $1 placeholders so generated SQL is directly executable
    and the model does not learn prepared-statement syntax at inference.
    rng is the seeded instance from generate() for full reproducibility.

    REVIEW FIX (#14): status/state/type columns previously always returned
    the hardcoded literal 'ACTIVE'. This schema's actual enum values are
    things like NOT_ASSIGNED, FROZEN, SUBMITTED, ATTEMPTED — 'ACTIVE' does
    not exist anywhere in the schema. Training the model on synthetic SQL
    containing a non-existent status value teaches it a plausible-looking
    hallucination, which is exactly the failure mode Phase 2 evaluator.py's
    hallucination check exists to catch — except CHECK constraint values
    aren't table/column names, so that check wouldn't catch this either.

    `inv` (the TableInventory for this column's table) is now optional but
    should be passed whenever available, so real enum values can be pulled
    from column_comments. When inv is None or no comment exists or no enum
    pattern is found in the comment text, falls back to a value that is
    obviously synthetic ('SYNTHETIC_PLACEHOLDER') rather than a plausible-
    looking fake — this is deliberately ugly so a reviewer scanning
    generated SQL notices it immediately and knows to either add a comment
    to that column or treat that pair with suspicion, rather than silently
    training on a wrong-but-plausible value.
    """
    col = col_name.lower()
    if "id" in col:
        return str(rng.randint(1, 9999))
    if "board" in col:
        return str(rng.randint(1, 20))
    if "status" in col or "state" in col or "type" in col:
        if inv is not None:
            comment = inv.column_comments.get(col_name, "")
            enum_values = _extract_enum_values(comment)
            if enum_values:
                return f"'{rng.choice(enum_values)}'"
        return "'SYNTHETIC_PLACEHOLDER'"
    if "year" in col:
        return str(rng.randint(2020, 2025))
    if "count" in col or "score" in col or "mark" in col or "total" in col:
        return str(rng.randint(1, 100))
    if "name" in col:
        return "'Sample Name'"
    if "date" in col or "time" in col or "created" in col:
        return "'2024-01-01'"
    return str(rng.randint(1, 100))


# ── NL phrase generators ──────────────────────────────────────────────────────
# H3 fix: all helpers now accept rng (seeded random.Random instance) and call
# rng.choice() instead of the module-level random.choice(). Without this, NL
# phrasing was non-deterministic even when --seed was set, making the synthetic
# JSONL non-reproducible across runs.

def _nl_for_single_lookup(table: str, col: str, rng: random.Random) -> str:
    table_clean = table.replace("_", " ")
    col_clean   = col.replace("_", " ")
    return rng.choice([
        f"Show all {table_clean} records where {col_clean} matches",
        f"List {table_clean} entries filtered by {col_clean}",
        f"Get {table_clean} where {col_clean} is specified",
        f"Find {table_clean} with a given {col_clean}",
    ])


def _nl_for_count(table: str, col: str, rng: random.Random) -> str:
    table_clean = table.replace("_", " ")
    col_clean   = col.replace("_", " ")
    return rng.choice([
        f"Count {table_clean} records grouped by {col_clean}",
        f"How many {table_clean} entries exist for each {col_clean}",
        f"Show the number of {table_clean} per {col_clean}",
    ])


def _nl_for_aggregation(table: str, col: str, rng: random.Random) -> str:
    table_clean = table.replace("_", " ")
    col_clean   = col.replace("_", " ")
    return rng.choice([
        f"Show total {col_clean} from {table_clean}",
        f"What is the sum of {col_clean} in {table_clean}",
        f"Get average {col_clean} across all {table_clean}",
        f"Find maximum {col_clean} in {table_clean}",
    ])


def _nl_for_join(table_a: str, table_b: str, join_col: str, rng: random.Random) -> str:
    a_clean = table_a.replace("_", " ")
    b_clean = table_b.replace("_", " ")
    return rng.choice([
        f"Show {a_clean} with their related {b_clean} information",
        f"List {a_clean} joined with {b_clean}",
        f"Get {a_clean} and corresponding {b_clean} details",
        f"Combine {a_clean} and {b_clean} data",
    ])


def _generate_single_table_lookup(
    tables: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT col1, col2 FROM t WHERE filter_col = <typed_literal>"""
    table_name = rng.choice(list(tables.keys()))
    inv        = tables[table_name]
    cols       = list(inv.columns.keys())

    if len(cols) < 2:
        return None

    select_cols = rng.sample(cols, min(3, len(cols)))
    filter_col  = rng.choice([c for c in cols if c not in select_cols] or cols)

    # Use a typed literal instead of $1 so the SQL is directly executable and
    # the model does not learn prepared-statement syntax at inference time.
    # REVIEW FIX (#14): pass inv so status/state/type columns can pull real
    # enum values from column_comments instead of a hardcoded fake value.
    literal    = _typed_literal(filter_col, rng, inv=inv)
    select_str = ", ".join(f"t.{c}" for c in select_cols)
    sql = (
        f"SELECT {select_str}\n"
        f"FROM {table_name} t\n"
        f"WHERE t.{filter_col} = {literal}\n"
        f"ORDER BY t.id\n"
        f"LIMIT 100;"
    )

    nl = _nl_for_single_lookup(table_name, filter_col, rng)   # H3: pass rng
    reasoning = (
        f"Single table lookup on {table_name}. "
        f"Filter: {filter_col} = {literal}. "
        f"Selected columns: {', '.join(select_cols)}."
    )

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "single_table_lookup", "tables": [table_name]}


def _generate_single_table_count(
    tables: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT group_col, COUNT(*) FROM t GROUP BY group_col"""
    table_name = rng.choice(list(tables.keys()))
    inv        = tables[table_name]
    cols       = list(inv.columns.keys())

    group_candidates = [c for c in cols if c not in ("id", "created_at", "updated_at")]
    if not group_candidates:
        return None

    group_col = rng.choice(group_candidates)
    sql = (
        f"SELECT t.{group_col}, COUNT(*) AS record_count\n"
        f"FROM {table_name} t\n"
        f"GROUP BY t.{group_col}\n"
        f"ORDER BY record_count DESC\n"
        f"LIMIT 50;"
    )

    nl        = _nl_for_count(table_name, group_col, rng)   # H3: pass rng
    reasoning = f"Aggregation on {table_name} grouped by {group_col}."

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "single_table_count", "tables": [table_name]}


def _generate_aggregation(
    tables: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT agg_func(numeric_col) FROM t GROUP BY group_col"""
    table_name   = rng.choice(list(tables.keys()))
    inv          = tables[table_name]
    numeric_cols = _get_numeric_columns(inv)
    group_cols   = [c for c in inv.columns if c not in ("id", "created_at", "updated_at")]

    if not numeric_cols or not group_cols:
        return None

    num_col   = rng.choice(numeric_cols)
    group_col = rng.choice([c for c in group_cols if c != num_col] or group_cols)
    agg_func  = rng.choice(["SUM", "AVG", "MAX", "MIN", "COUNT"])

    sql = (
        f"SELECT t.{group_col},\n"
        f"       {agg_func}(t.{num_col}) AS {agg_func.lower()}_{num_col}\n"
        f"FROM {table_name} t\n"
        f"GROUP BY t.{group_col}\n"
        f"ORDER BY {agg_func.lower()}_{num_col} DESC\n"
        f"LIMIT 100;"
    )

    nl        = _nl_for_aggregation(table_name, num_col, rng)   # H3: pass rng
    reasoning = (
        f"{agg_func} aggregation on {table_name}.{num_col} "
        f"grouped by {group_col}."
    )

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "aggregation", "tables": [table_name]}


def _generate_simple_join(
    tables: dict[str, Any],
    graph,
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT ... FROM a JOIN b ON a.fk = b.id"""
    if graph is None:
        return None

    # Use undirected edges so both FK traversal directions are considered.
    # graph.edges() on a DiGraph returns only forward (child → parent) edges;
    # converting to undirected makes parent → child joins available too.
    undirected = graph.to_undirected()
    edges = list(undirected.edges(data=True))
    if not edges:
        return None

    src, dst, data = rng.choice(edges)
    if src not in tables or dst not in tables:
        return None

    from_col = data.get("from_col", "id")
    to_col   = data.get("to_col", "id")

    src_inv  = tables[src]
    dst_inv  = tables[dst]

    src_cols = rng.sample(list(src_inv.columns.keys()), min(2, len(src_inv.columns)))
    dst_cols = rng.sample(list(dst_inv.columns.keys()), min(2, len(dst_inv.columns)))

    src_select = ", ".join(f"a.{c}" for c in src_cols)
    dst_select = ", ".join(f"b.{c}" for c in dst_cols)

    sql = (
        f"SELECT {src_select},\n"
        f"       {dst_select}\n"
        f"FROM {src} a\n"
        f"JOIN {dst} b ON a.{from_col} = b.{to_col}\n"
        f"ORDER BY a.id\n"
        f"LIMIT 100;"
    )

    nl        = _nl_for_join(src, dst, from_col, rng)   # H3: pass rng
    reasoning = (
        f"Join {src} → {dst} via {src}.{from_col} = {dst}.{to_col}. "
        f"Selected from {src}: {', '.join(src_cols)}. "
        f"Selected from {dst}: {', '.join(dst_cols)}."
    )

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "simple_join", "tables": [src, dst]}


def _generate_multi_join(
    tables: dict[str, Any],
    graph,
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT ... FROM a JOIN b ON ... JOIN c ON ..."""
    if graph is None:
        return None

    # Use undirected graph so both FK traversal directions are available.
    # graph.out_degree / graph.successors only finds child → parent edges;
    # undirected degree ≥ 2 finds any node with 2+ FK connections regardless
    # of direction.  The redundant `import networkx as nx` is also removed —
    # nx is already imported at module level via _load_graph().
    undirected = graph.to_undirected()

    candidates = [n for n in undirected.nodes() if undirected.degree(n) >= 2 and n in tables]
    if not candidates:
        return None

    root      = rng.choice(candidates)
    neighbors = [n for n in undirected.neighbors(root) if n in tables]
    if len(neighbors) < 2:
        return None

    b, c     = neighbors[0], neighbors[1]
    root_inv = tables[root]
    b_inv    = tables[b]
    c_inv    = tables[c]

    edge_ab = undirected.get_edge_data(root, b) or {}
    edge_ac = undirected.get_edge_data(root, c) or {}

    ab_from = edge_ab.get("from_col", "id")
    ab_to   = edge_ab.get("to_col", "id")
    ac_from = edge_ac.get("from_col", "id")
    ac_to   = edge_ac.get("to_col", "id")

    root_cols = rng.sample(list(root_inv.columns.keys()), min(2, len(root_inv.columns)))
    b_cols    = rng.sample(list(b_inv.columns.keys()), min(1, len(b_inv.columns)))
    c_cols    = rng.sample(list(c_inv.columns.keys()), min(1, len(c_inv.columns)))

    root_select = ", ".join(f"a.{col}" for col in root_cols)
    b_select    = ", ".join(f"b.{col}" for col in b_cols)
    c_select    = ", ".join(f"c.{col}" for col in c_cols)

    sql = (
        f"SELECT {root_select},\n"
        f"       {b_select},\n"
        f"       {c_select}\n"
        f"FROM {root} a\n"
        f"JOIN {b} b ON a.{ab_from} = b.{ab_to}\n"
        f"JOIN {c} c ON a.{ac_from} = c.{ac_to}\n"
        f"ORDER BY a.id\n"
        f"LIMIT 100;"
    )

    nl = (
        f"Show {root.replace('_', ' ')} with related "
        f"{b.replace('_', ' ')} and {c.replace('_', ' ')} details"
    )
    reasoning = (
        f"Three-table join: {root} → {b} via {ab_from}/{ab_to}, "
        f"{root} → {c} via {ac_from}/{ac_to}."
    )

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "multi_join", "tables": [root, b, c]}


def _generate_status_filter(
    tables: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    """SELECT ... FROM t WHERE status_col = 'VALUE'"""
    # Find tables with status-like columns
    status_tables = [
        (name, inv) for name, inv in tables.items()
        if _get_status_columns(inv)
    ]
    if not status_tables:
        return None

    table_name, inv = rng.choice(status_tables)
    status_col      = rng.choice(_get_status_columns(inv))
    cols            = rng.sample(list(inv.columns.keys()), min(3, len(inv.columns)))

    # Use a typed literal instead of $1 — same rationale as _generate_single_table_lookup.
    # REVIEW FIX (#14): pass inv — this is the status-filter generator, the
    # exact path that was always emitting 'ACTIVE' regardless of schema.
    literal = _typed_literal(status_col, rng, inv=inv)
    sql = (
        f"SELECT {', '.join(f't.{c}' for c in cols)}\n"
        f"FROM {table_name} t\n"
        f"WHERE t.{status_col} = {literal}\n"
        f"ORDER BY t.id\n"
        f"LIMIT 100;"
    )

    nl = (
        f"Show {table_name.replace('_', ' ')} records with a specific "
        f"{status_col.replace('_', ' ')} value"
    )
    reasoning = f"Status filter on {table_name}.{status_col}."

    return {"nl_query": nl, "sql": sql, "reasoning": reasoning,
            "category": "status_filter", "tables": [table_name]}


_GENERATORS = {
    "single_table_lookup": _generate_single_table_lookup,
    "single_table_count":  _generate_single_table_count,
    "aggregation":         _generate_aggregation,
    "simple_join":         _generate_simple_join,
    "multi_join":          _generate_multi_join,
    "status_filter":       _generate_status_filter,
}


def generate(
    count:      int       = 200,
    categories: list[str] | None = None,
    seed:       int       = 42,
) -> None:
    """
    Generate synthetic NL→SQL pairs and write to data/fine_tuning_synthetic.jsonl.

    Args:
        count:      Number of pairs to generate
        categories: Which categories to include (default: all)
        seed:       Random seed for reproducibility
    """
    rng = random.Random(seed)

    # Load schema and graph
    print("\nLoading schema and FK graph…")
    tables = _load_schema()
    graph  = _load_graph()
    print(f"  {len(tables)} tables loaded")
    if graph:
        print(f"  FK graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Build weighted category list
    active_categories = categories or list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS.get(c, 10) for c in active_categories]

    pairs:    list[dict[str, Any]] = []
    attempts: int = 0
    max_attempts = count * 10   # avoid infinite loop on small schemas

    print(f"\nGenerating {count} synthetic pairs…")

    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        category = rng.choices(active_categories, weights=weights, k=1)[0]
        generator = _GENERATORS.get(category)
        if not generator:
            continue

        # Status-filter and single-table generators don't need graph
        if category in ("simple_join", "multi_join"):
            pair = generator(tables, graph, rng)
        else:
            pair = generator(tables, rng)

        if pair and pair.get("nl_query") and pair.get("sql"):
            pair["source"] = "synthetic"
            pairs.append(pair)

    if len(pairs) < count:
        print(
            f"⚠  Only generated {len(pairs)}/{count} pairs after {attempts} attempts.\n"
            "   Schema may be too small for some categories.\n"
        )

    # Deduplicate by NL query
    seen: set[str] = set()
    unique_pairs = []
    for p in pairs:
        key = p["nl_query"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    # Write atomically
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    lines = "\n".join(json.dumps(p, ensure_ascii=False) for p in unique_pairs)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(lines, encoding="utf-8")
    os.replace(tmp, OUTPUT_PATH)

    logger.info(
        component="synthetic",
        event="generation_complete",
        requested=count,
        generated=len(unique_pairs),
        output=str(OUTPUT_PATH),
    )

    # Category breakdown
    from collections import Counter
    by_cat = Counter(p["category"] for p in unique_pairs)
    print(f"\n✓  Generated {len(unique_pairs)} synthetic pairs → {OUTPUT_PATH}")
    print("\n  Category breakdown:")
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat:<30} {n:>4}")
    print(
        f"\n⚠  Reminder: synthetic pairs are a bootstrap tool.\n"
        "   Real failure corrections are always higher quality.\n"
        "   Stop using synthetic data once you have 200+ real pairs.\n"
        "\nNext step: python fine_tuning/data_pipeline.py --include-synthetic\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--count",      type=int,    default=200,
                        help="Number of synthetic pairs to generate (default: 200)")
    parser.add_argument("--categories", nargs="+",
                        choices=list(CATEGORY_WEIGHTS.keys()),
                        help="Categories to generate (default: all)")
    parser.add_argument("--seed",       type=int,    default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate(
        count      = args.count,
        categories = args.categories,
        seed       = args.seed,
    )
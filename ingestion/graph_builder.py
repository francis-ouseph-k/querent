"""
ingestion/graph_builder.py
───────────────────────────
Builds a NetworkX DiGraph from parsed TableInventory objects.

Graph semantics:
  Node  = table name (str)
  Edge  = FK relationship  source_table → target_table
  Edge attributes: from_col, to_col, constraint_name

Self-referential FKs (table → itself) are EXCLUDED from the graph.
They are preserved in the TableInventory for semantic chunk documentation
but carry no join-path information and would create trivial self-loops.

Cross-table cycles (rare in well-designed schemas but possible) are handled
gracefully by the BFS traversal via a visited set — not treated as errors.
After filtering self-refs, nx.is_directed_acyclic_graph() is called as a
diagnostic. If it returns False (cross-table cycles exist), a warning is
logged but the graph is fully usable for bidirectional BFS.

Public API:
    builder = GraphBuilder()
    G       = builder.build(tables)
    paths   = builder.find_join_paths(G, seed_tables=["evaluation_marks", "board"])
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import networkx as nx

from models.schema import TableInventory
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class GraphBuilder:
    """Builds and queries the FK relationship graph."""

    def build(self, tables: dict[str, TableInventory]) -> nx.DiGraph:
        """
        Build a directed graph from TableInventory FK relationships.

        Returns a nx.DiGraph where:
        G.nodes[table_name]  — node exists for every table
        G.edges[(src, dst)]  — one edge per FK (self-refs excluded)
        G.edges[(src, dst)]["from_col"]  — FK column on source table
        G.edges[(src, dst)]["to_col"]    — referenced column on target table
        """
        G: nx.DiGraph = nx.DiGraph()

        # ── Add all tables as nodes ───────────────────────────────────────
        for table_name, inv in tables.items():
            G.add_node(table_name, comment=inv.comment, is_view=inv.is_view)

        # ── Add FK edges (excluding self-referential FKs) ─────────────────
        self_ref_count  = 0
        edge_count      = 0
        for table_name, inv in tables.items():
            for fk in inv.foreign_keys:
                if fk.is_self_referential:
                    self_ref_count += 1
                    logger.debug(
                        component="graph_builder",
                        event="self_ref_excluded",
                        table=table_name,
                        column=fk.from_col,
                    )
                    continue

                # Add target node if not already present (e.g. ERP cache tables)
                if fk.to_table not in G:
                    G.add_node(fk.to_table, comment="", is_view=False)

                # Allow multiple FK edges between the same pair of tables
                # (e.g. answer_script has multiple FKs to academic_unit)
                # Use a unique key per edge
                edge_key = f"{fk.from_col}__{fk.to_col}"
                G.add_edge(
                    fk.from_table,
                    fk.to_table,
                    key=edge_key,
                    from_col=fk.from_col,
                    to_col=fk.to_col,
                    constraint_name=fk.constraint_name,
                )
                edge_count += 1

        # ── Load derived FK edges (virtual) ───────────────────────────────
        derived_path = Path(__file__).parents[1] / "config" / "derived_fks.yaml"
        if derived_path.exists():
            try:
                import yaml
                with derived_path.open(encoding="utf-8") as f:
                    derived_payload = yaml.safe_load(f)
                for d in derived_payload.get("derived_fks", []):
                    src = d["source_table"]
                    tgt = d["target_table"]
                    
                    mappings = d.get("column_mappings", [])
                    from_col = mappings[0]["source_column"] if mappings else None
                    to_col = mappings[0]["target_column"] if mappings else None
                    
                    G.add_edge(
                        src,
                        tgt,
                        key=f"derived_{src}_{tgt}",
                        from_col=from_col,
                        to_col=to_col,
                        column_mappings=mappings,
                        join_type=d.get("join_type", "INNER"),
                        condition=d.get("condition"),
                        derived=True,
                        comment=d.get("comment", ""),
                    )
                if settings.debug_mode:
                    logger.debug(
                        component="graph_builder",
                        event="derived_fks_loaded",
                        count=len(derived_payload.get("derived_fks", [])),
                        details=derived_payload.get("derived_fks", []),
                    )
            except Exception as e:
                logger.error(
                    component="graph_builder",
                    event="derived_fks_load_error",
                    error=str(e),
                )

        # ── DAG diagnostic ────────────────────────────────────────────────
        is_dag = nx.is_directed_acyclic_graph(G)
        if is_dag:
            logger.info(
                component="graph_builder",
                event="graph_is_dag",
                nodes=G.number_of_nodes(),
                edges=G.number_of_edges(),
                self_refs_excluded=self_ref_count,
            )
        else:
            # FIX-NEW-H6: limit simple_cycles materialisation with islice.
            # nx.simple_cycles() is a generator — calling list() on it fully
            # materialises all cycles, which is polynomial in the number of edges.
            # We only log the first 5 cycles and a total count; there is no
            # reason to materialise more than a small cap for diagnostics.
            import itertools
            _MAX_CYCLES = 20
            cycles = list(itertools.islice(nx.simple_cycles(G), _MAX_CYCLES))
            logger.warning(
                component="graph_builder",
                event="cross_table_cycles_detected",
                cycles_shown=len(cycles),
                cycles_capped_at=_MAX_CYCLES,
                cycles=[" → ".join(c) for c in cycles[:5]],  # log first 5
                note="BFS traversal handles cycles safely via visited set",
            )

        logger.info(
            component="graph_builder",
            event="build_complete",
            nodes=G.number_of_nodes(),
            edges=edge_count,
            self_refs_excluded=self_ref_count,
        )
        return G

    # ─────────────────────────────────────────────────────────────────────
    # Graph traversal — called at query time by the retrieval orchestrator
    # ─────────────────────────────────────────────────────────────────────

    def find_join_paths(
        self,
        G:            nx.DiGraph,
        seed_tables:  list[str],
    ) -> dict[str, Any]:
        """
        Find the minimal connecting subgraph between seed tables using
        the Steiner Tree spanning tree approximation.

        Steiner Tree algorithm connections:
        ===================================
        - Connecting a subset of target nodes (seeds) globally via a Minimum Spanning Tree (MST).
        - Discovers intermediate bridging tables (Steiner nodes) only as needed.
        - Avoids the combinatorial complexity and redundant paths generated by BFS pairwise union paths.
        - Prevents context-bloat (redundant JOIN clauses) in the generated SQL queries.

        Args:
            G (nx.DiGraph): The schema graph where nodes are tables and edges are foreign keys.
            seed_tables (list): The list of tables explicitly extracted from the user query.

        Returns:
            dict: A dictionary containing:
                - "connecting_tables" (set): All seed tables plus bridging tables.
                - "join_paths" (list): Edge tuples representing resolved joins.
                - "path_descriptions" (list): Explicit, human-readable SQL "JOIN table ON ..." clauses.
        """
        # Step 1: Return empty lists if no seed tables are provided.
        if not seed_tables:
            return {"connecting_tables": set(), "join_paths": [], "path_descriptions": []}

        # Step 2: Ensure that we only check tables that actually exist in our schema graph.
        valid_seeds = [t for t in seed_tables if t in G]
        if not valid_seeds:
            return {"connecting_tables": set(), "join_paths": [], "path_descriptions": []}

        # Step 3: If there is only a single table, no join paths are needed.
        if len(valid_seeds) <= 1:
            return {
                "connecting_tables": set(valid_seeds),
                "join_paths": [],
                "path_descriptions": []
            }

        # Step 4: Convert directed schema graph to an undirected graph for spanning tree approximations.
        U = G.to_undirected()

        # Step 5: Approximate the Steiner Tree connecting all valid seed tables.
        # This will discover the minimal set of connections between the table nodes.
        # If the schema has disconnected components (e.g. isolated tables like academic_calendar),
        # we must run steiner_tree independently on each connected component to prevent KeyErrors.
        from networkx.algorithms.approximation import steiner_tree
        
        connecting_tables = set(valid_seeds)
        T_edges = []
        
        try:
            for comp in nx.connected_components(U):
                comp_seeds = [s for s in valid_seeds if s in comp]
                if len(comp_seeds) >= 2:
                    subgraph = U.subgraph(comp)
                    T = steiner_tree(subgraph, comp_seeds)
                    if T.number_of_nodes() > 0:
                        connecting_tables.update(T.nodes())
                        T_edges.extend(T.edges())
        except Exception as exc:
            logger.warning(
                component="graph_builder",
                event="steiner_tree_failed",
                error=str(exc),
                note="Falling back to original seeds without join paths."
            )
            return {
                "connecting_tables": set(valid_seeds),
                "join_paths": [],
                "path_descriptions": []
            }

        join_paths = []
        path_descriptions = []
        seen_edges = set()

        # Step 6: Loop through the edges of the spanning tree to generate SQL JOIN statements.
        for u, v in T_edges:
            # Keep order independent seen set to avoid duplicate traversal
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Retrieve the specific foreign key columns and format the JOIN clause.
            # We must check the correct FK direction in our directed graph G.
            if G.has_edge(u, v):
                edge_data = self._get_edge_data(G, u, v)
                if edge_data:
                    from_col = edge_data.get("from_col")
                    to_col = edge_data.get("to_col")
                    
                    if edge_data.get("derived", False):
                        mappings = edge_data.get("column_mappings", [])
                        cond_parts = []
                        for m in mappings:
                            src_col = m["source_column"]
                            tgt_col = m["target_column"]
                            cond_parts.append(f"{v}.{tgt_col} = {u}.{src_col}")
                        
                        extra_cond = edge_data.get("condition")
                        if extra_cond:
                            if "scope_type" in extra_cond and not any(t + "." in extra_cond for t in [u, v]):
                                extra_cond = extra_cond.replace("scope_type", f"{u}.scope_type")
                            cond_parts.append(extra_cond)
                            
                        join_cond = " AND ".join(cond_parts)
                        join_type = edge_data.get("join_type", "INNER")
                        join_prefix = "JOIN" if join_type == "INNER" else f"{join_type} JOIN"
                        join_clause = f"{join_prefix} {v} ON {join_cond}"
                        
                        join_paths.append((u, v, from_col, to_col))
                        path_descriptions.append(join_clause)
                        
                        if settings.debug_mode:
                            print(f"[DEBUG] Using derived FK from '{u}' to '{v}': {join_clause}")
                    else:
                        join_paths.append((u, v, from_col, to_col))
                        path_descriptions.append(
                            f"JOIN {v} ON {v}.{to_col} = {u}.{from_col}"
                        )
            elif G.has_edge(v, u):
                edge_data = self._get_edge_data(G, v, u)
                if edge_data:
                    from_col = edge_data.get("from_col")
                    to_col = edge_data.get("to_col")
                    
                    if edge_data.get("derived", False):
                        mappings = edge_data.get("column_mappings", [])
                        cond_parts = []
                        for m in mappings:
                            src_col = m["source_column"]
                            tgt_col = m["target_column"]
                            cond_parts.append(f"{v}.{src_col} = {u}.{tgt_col}")
                        
                        extra_cond = edge_data.get("condition")
                        if extra_cond:
                            if "scope_type" in extra_cond and not any(t + "." in extra_cond for t in [u, v]):
                                extra_cond = extra_cond.replace("scope_type", f"{v}.scope_type")
                            cond_parts.append(extra_cond)
                            
                        join_cond = " AND ".join(cond_parts)
                        join_type = edge_data.get("join_type", "INNER")
                        join_prefix = "JOIN" if join_type == "INNER" else f"{join_type} JOIN"
                        join_clause = f"{join_prefix} {v} ON {join_cond}"
                        
                        join_paths.append((v, u, from_col, to_col))
                        path_descriptions.append(join_clause)
                        
                        if settings.debug_mode:
                            print(f"[DEBUG] Using derived FK from '{v}' to '{u}': {join_clause}")
                    else:
                        join_paths.append((v, u, from_col, to_col))
                        path_descriptions.append(
                            f"JOIN {v} ON {v}.{from_col} = {u}.{to_col}"
                        )

        return {
            "connecting_tables": connecting_tables,
            "join_paths":        join_paths,
            "path_descriptions": path_descriptions,
        }

    def _get_edge_data(
        self,
        G:   nx.DiGraph,
        src: str,
        dst: str,
    ) -> dict[str, str] | None:
        """
        Safely retrieve edge data (FK column metadata) for a directed edge.

        Handles both standard directed graphs (where edge data is a flat dictionary)
        and multi-directed graphs (where there could be multiple foreign keys between
        the same pair of tables, returning a dictionary-of-dictionaries).

        Args:
            G (nx.DiGraph): The schema graph.
            src (str): Source table name.
            dst (str): Destination table name.

        Returns:
            dict: Dictionary with 'from_col' and 'to_col' keys, or None.
        """
        # Check standard direction src -> dst
        if G.has_edge(src, dst):
            data = G.get_edge_data(src, dst)
            if isinstance(data, dict):
                # If there are multiple edges (MultiDiGraph), data is a dict of dicts.
                # We fetch the first edge relation.
                vals = list(data.values())
                first_val = vals[0] if len(vals) > 0 else None
                if isinstance(first_val, dict):
                    return first_val
                return data
        
        # Check reverse direction dst -> src
        if G.has_edge(dst, src):
            data = G.get_edge_data(dst, src)
            if isinstance(data, dict):
                vals = list(data.values())
                first_val = vals[0] if len(vals) > 0 else None
                if isinstance(first_val, dict):
                    return first_val
                return data
        return None
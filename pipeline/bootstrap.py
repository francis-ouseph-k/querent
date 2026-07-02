"""
pipeline/bootstrap.py
─────────────────────
Pipeline assembly and the schema-version gate.

    check_schema_version(strict)  compares the current DDL hash against the hash
                                  the vector index / FK graph were built from. On
                                  drift it warns, or aborts when strict — a
                                  benchmark run on a stale index is invalid.
    load_tables()                 parse the DDL into the {table: TableInventory} map
    load_graph()                  load the persisted FK graph (networkx)
    create_runner(strict)         wire tables + graph + the 12-step validation
                                  pipeline into a ready PipelineRunner

This is the composition root: the place where the concrete objects are built and
handed to the runner. Everything downstream receives them by injection.
"""

import json
import sys
from pathlib import Path

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def check_schema_version(strict: bool = False) -> None:
    """
    Compare the current DDL hash against the hash stored after
    the last successful ingest run.
    """
    from utils.schema_versioning import compute_ddl_hash, load_stored_state

    ddl_path = Path(settings.ddl_path)
    if not ddl_path.exists():
        return

    ddl_text = ddl_path.read_text(encoding="utf-8")
    current_hash = compute_ddl_hash(ddl_text)
    stored = load_stored_state(settings.schema_hash_path)

    if not stored:
        return

    stored_hash = stored.get("ddl_hash", "")
    if stored_hash and stored_hash != current_hash:
        if strict:
            print(
                "\n"
                "❌ FATAL ERROR: DDL has changed since the last ingestion run.\n"
                "   The vector index and FK graph are stale.\n"
                "   New columns / tables will not appear in retrieved context.\n"
                "   Run:  python ingest.py\n"
                "   to re-index the schema before querying.\n"
            )
            logger.critical(
                component="bootstrap",
                event="schema_drift_fatal",
                stored_hash=stored_hash[:12],
                current_hash=current_hash[:12],
                note="Ingestion required before execution under strict version check",
            )
            sys.exit(1)

        print(
            "\n"
            "⚠  WARNING: DDL has changed since the last ingestion run.\n"
            "   The vector index and FK graph may be stale.\n"
            "   New columns / tables will not appear in retrieved context.\n"
            "   Run:  python ingest.py\n"
            "   to re-index the schema before querying.\n"
        )
        logger.warning(
            component="bootstrap",
            event="schema_drift_detected",
            stored_hash=stored_hash[:12],
            current_hash=current_hash[:12],
            note="Run ingest.py to re-index the updated schema",
        )


def load_graph():
    """Load the pre-built FK graph from disk. Fail fast if not found."""
    import networkx as nx

    graph_path_json = Path("data/fk_graph.json")
    graph_path_pkl = Path("data/fk_graph.pkl")

    if graph_path_json.exists():
        graph_data = json.loads(graph_path_json.read_text(encoding="utf-8"))
        return nx.node_link_graph(graph_data)

    if graph_path_pkl.exists():
        print(
            "ERROR: Only the legacy data/fk_graph.pkl found.\n"
            "The .pkl format has been replaced with .json for security.\n"
            "Run:  python ingest.py --full\n"
            "to regenerate the graph in the safe JSON format."
        )
        sys.exit(1)

    print(
        "ERROR: FK graph not found at data/fk_graph.json\n"
        "Run:  python ingest.py\n"
        "to build the graph and index the schema first."
    )
    sys.exit(1)


def load_tables():
    """Parse the DDL to get TableInventory objects for the validator."""
    from ingestion.ddl_parser import DDLParser
    ddl_path = Path(settings.ddl_path)
    if not ddl_path.exists():
        print(f"ERROR: DDL file not found: {ddl_path}")
        sys.exit(1)
    parser = DDLParser()
    return parser.parse_file(ddl_path)


def create_runner(strict_version_check: bool = False):
    """
    Initialize and return a ready-to-use PipelineRunner.
    This handles version checking and loading graph/tables.
    """
    check_schema_version(strict_version_check)
    
    print("Loading schema…")
    tables = load_tables()
    graph = load_graph()
    print(f"  {len(tables)} tables, {graph.number_of_nodes()} graph nodes ready.")

    from pipeline.runner import PipelineRunner
    return PipelineRunner(tables=tables, fk_graph=graph)

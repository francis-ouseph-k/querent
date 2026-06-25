"""
utils/schema_versioning.py
──────────────────────────
DDL change detection via SHA-256 hash.
When the DDL changes, only the chunks referencing changed tables
are invalidated and re-embedded — not the full index.

FIX C3 (Critical) — original compute_table_hashes() used a regex with
non-greedy .*? to extract CREATE TABLE blocks. This had the same nested-
parentheses failure mode as the original DDL parser:
  CHECK (status IN ('A', 'B')) — first ')' terminates the match early
  PARTITION OF parent — no column list, pattern never matches
  DDL without trailing semicolons — pattern misses the block entirely
If a table block was missed, its hash was never computed. A real schema
change could go undetected (stale chunks persist) or a formatting change
could trigger false-positive re-indexing of the wrong tables.

FIX: compute_table_hashes() now accepts a pre-parsed tables dict from
DDLParser instead of re-parsing the raw DDL text. The DDLParser already
uses sqlglot AST and handles all these edge cases correctly. We extract
the per-table DDL by normalising via sqlglot's own SQL generation rather
than trying to find block boundaries in the raw text.

This eliminates the regex entirely. The DDL text hash (top-level) is
still computed directly from the raw text — that is stable and correct
because it is a whole-file hash, not a structural parse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _hash_text(text: str) -> str:
    """Return hex SHA-256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_ddl_hash(ddl_text: str) -> str:
    """Top-level hash of the full DDL file. Stable — hashes raw text."""
    return _hash_text(ddl_text)


def compute_table_hashes(
    tables: dict,        # dict[str, TableInventory] from DDLParser.parse()
) -> dict[str, str]:
    """
    Compute a per-table hash from the parsed TableInventory objects.

    FIX C3: replaces the fragile CREATE TABLE regex with a deterministic
    hash derived from the parsed structure. Each table's hash covers:
      - Column names and types
      - Foreign key relationships
      - Index definitions
      - Table and column comments

    Two tables with identical structure but different whitespace / comment
    formatting will produce the same hash — this is desirable because
    cosmetic DDL reformatting should not trigger re-indexing.

    Two tables where a column was added, a FK changed, or a comment was
    updated will produce different hashes — triggering targeted re-indexing
    of only the affected chunks.
    """
    table_hashes: dict[str, str] = {}

    for table_name, inv in tables.items():
        # Build a canonical string representation of this table's structure.
        # Order each section to ensure determinism regardless of dict insertion order.
        parts = [f"table:{table_name}"]

        # Comment
        if inv.comment:
            parts.append(f"comment:{inv.comment}")

        # Columns — sorted by name for determinism
        for col_name in sorted(inv.columns):
            col = inv.columns[col_name]
            parts.append(
                f"col:{col_name}:{col.data_type}:"
                f"{'nn' if not col.nullable else 'null'}:"
                f"{'pk' if col.is_pk else ''}:"
                f"{'jsonb' if col.has_jsonb else ''}:"
                f"{col.comment or ''}"
            )

        # Foreign keys — sorted by (from_col, to_table) for determinism
        for fk in sorted(inv.foreign_keys, key=lambda f: (f.from_col, f.to_table)):
            parts.append(f"fk:{fk.from_col}->{fk.to_table}.{fk.to_col}")

        # Indexes — sorted by name
        for idx in sorted(inv.indexes, key=lambda i: i.name):
            parts.append(
                f"idx:{idx.name}:{idx.method}:"
                f"{'u' if idx.is_unique else ''}:"
                f"{'p' if idx.is_partial else ''}:"
                f"{','.join(idx.columns)}"
            )

        canonical = "\n".join(parts)
        table_hashes[table_name] = _hash_text(canonical)

    return table_hashes


def load_stored_state(hash_path: str) -> dict:
    """Load persisted hash state from disk. Returns empty dict if absent."""
    p = Path(hash_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(event="hash_state_load_failed", path=hash_path, error=str(exc))
        return {}


def save_stored_state(hash_path: str, state: dict) -> None:
    """Persist hash state to disk."""
    p = Path(hash_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def detect_changed_tables(
    ddl_text: str,
    hash_path: str,
    tables:    dict | None = None,   # dict[str, TableInventory] — required for per-table hashing
) -> tuple[set[str], str, bool]:
    """
    Compare current DDL against stored state.

    tables must be the output of DDLParser.parse(ddl_text). If omitted,
    per-table change detection falls back to treating all tables as changed.

    Returns:
        changed_tables  — set of table names that changed (empty = no change)
        new_ddl_hash    — top-level hash of current DDL
        is_first_run    — True if no stored state existed
    """
    current_ddl_hash = compute_ddl_hash(ddl_text)
    stored           = load_stored_state(hash_path)

    is_first_run = not stored

    if tables is not None:
        current_table_hash = compute_table_hashes(tables)
    else:
        # No parsed tables provided — cannot do per-table diff.
        # Treat all tables as changed so ingestion is comprehensive.
        logger.warning(
            event="table_hashes_unavailable",
            note="tables dict not provided; all tables treated as changed",
        )
        current_table_hash = {}

    if is_first_run:
        logger.info(event="first_run", tables=len(current_table_hash))
        changed = set(current_table_hash.keys()) if current_table_hash else set()
        return changed, current_ddl_hash, True

    stored_ddl_hash   = stored.get("ddl_hash", "")
    stored_table_hash = stored.get("table_hashes", {})

    if stored_ddl_hash == current_ddl_hash:
        logger.info(event="no_change", hash=current_ddl_hash[:12])
        return set(), current_ddl_hash, False

    # Full DDL changed — find per-table diffs
    changed: set[str] = set()
    for table, h in current_table_hash.items():
        if stored_table_hash.get(table) != h:
            changed.add(table)

    # Tables deleted since last run
    deleted = set(stored_table_hash.keys()) - set(current_table_hash.keys())
    changed |= deleted

    logger.info(
        event="changes_detected",
        changed_tables=sorted(changed),
        count=len(changed),
    )
    return changed, current_ddl_hash, False


def update_stored_state(
    ddl_text:  str,
    hash_path: str,
    tables:    dict | None = None,
) -> str:
    """Recompute and persist hash state after successful ingestion."""
    ddl_hash     = compute_ddl_hash(ddl_text)
    table_hashes = compute_table_hashes(tables) if tables else {}
    save_stored_state(hash_path, {
        "ddl_hash":     ddl_hash,
        "table_hashes": table_hashes,
    })
    logger.info(event="state_updated", hash=ddl_hash[:12], tables=len(table_hashes))
    return ddl_hash
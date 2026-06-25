"""
ingestion/chunk_generator.py
─────────────────────────────
Converts TableInventory objects (from the DDL parser) into SemanticChunk objects
ready for embedding and indexing.

FIX H1 (High) — original implementation used four hardcoded Python sets to
classify tables into AUDIT, WORKFLOW, STATUS, and PARTITION chunk types:

    _AUDIT_TABLES    = {"audit_log", "result_history", ...}
    _WORKFLOW_TABLES = {"answer_script", "evaluation_attempt", ...}
    _STATUS_COLUMNS  = {"lifecycle_status", "scan_status", ...}
    _PARTITION_TABLES = {"audit_log", "evaluation_marks"}

When the schema evolved (new audit table, new status column), a developer had
to update these sets and redeploy. With 50+ tables this was error-prone and
was already missing several v10 tables.

FIX: Replace with metadata-driven detection functions that inspect the parsed
TableInventory at runtime. Detection uses three signals in priority order:

  1. DDL comment annotations — @chunk:TYPE in table or column comments
     Example: COMMENT ON TABLE purge_job_log IS '... @chunk:audit ...'
     Highest priority — explicit beats heuristic. Allows DDL-only changes
     to control chunk classification without code deployment.

  2. Naming conventions — suffix patterns that reliably indicate chunk type
     *_history, *_log, *_audit → AUDIT
     *_status columns → STATUS

  3. Column signature matching — presence of known status column names
     Any table with a 'status' column containing IN (...) constraints
     is a candidate for WORKFLOW detection.

The hardcoded sets are retained as OVERRIDE_* constants for edge cases
where the naming convention or annotations do not apply (e.g., evaluation_marks
which is NOT partitioned — a notable absence that warrants a PARTITION chunk
explaining the design decision).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from models.schema import ChunkType, ColumnInfo, SemanticChunk, TableInventory
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Override sets — used ONLY for tables that cannot be detected by convention
# or annotation. Keep these minimal.
# ─────────────────────────────────────────────────────────────────────────────

# Notable non-partitioning decisions worth a PARTITION chunk explaining why
_PARTITION_NOTABLE_ABSENCE = {
    "evaluation_marks",  # D-3: NOT partitioned — retention keyed off board.closed_at
}

# Index methods that always warrant an INDEX chunk regardless of other signals
_NOTABLE_INDEX_METHODS = {"gin", "gist", "hash"}

# ─────────────────────────────────────────────────────────────────────────────
# DDL-driven detection helpers
# ─────────────────────────────────────────────────────────────────────────────

# Annotation pattern in DDL comments: @chunk:TYPE
_ANNOTATION_RE = re.compile(r"@chunk:(\w+)", re.IGNORECASE)

# Suffix patterns that reliably indicate AUDIT tables
_AUDIT_SUFFIXES = ("_history", "_log", "_audit", "_transition", "_purge_log")

# Column names that indicate STATUS chunk worthiness
_STATUS_COLUMN_NAMES = {
    "lifecycle_status", "scan_status", "evaluation_status", "block_status",
    "status", "approval_status", "erp_push_status", "sync_status",
}

# Column names that indicate WORKFLOW semantics
_WORKFLOW_COLUMN_NAMES = {
    "lifecycle_status", "evaluation_status", "scan_status", "block_status",
    "status",
}


def _get_chunk_annotation(inv: TableInventory) -> str | None:
    """
    Extract @chunk:TYPE annotation from table or column comments.
    Returns the chunk type string if found, else None.
    Signal 1 — explicit DDL annotation beats all heuristics.
    """
    if inv.comment:
        m = _ANNOTATION_RE.search(inv.comment)
        if m:
            return m.group(1).lower()
    for comment in inv.column_comments.values():
        m = _ANNOTATION_RE.search(comment)
        if m:
            return m.group(1).lower()
    return None


def _is_audit_table(inv: TableInventory) -> bool:
    """
    Signal 1: @chunk:audit annotation.
    Signal 2: table name ends with an audit suffix.
    Signal 3: table name contains 'audit' or 'history'.
    """
    annotation = _get_chunk_annotation(inv)
    if annotation == "audit":
        return True
    name = inv.table_name.lower()
    if any(name.endswith(s) for s in _AUDIT_SUFFIXES):
        return True
    if "audit" in name or "history" in name:
        return True
    return False


def _is_workflow_table(inv: TableInventory) -> bool:
    """
    Signal 1: @chunk:workflow annotation.
    Signal 2: table has multiple orthogonal status columns (lifecycle pattern).
    Signal 3: table has a 'status' column with a comment containing transition language.
    """
    annotation = _get_chunk_annotation(inv)
    if annotation == "workflow":
        return True

    # Multiple status columns → likely has a lifecycle
    status_cols = [c for c in inv.columns if c in _WORKFLOW_COLUMN_NAMES]
    if len(status_cols) >= 2:
        return True

    # Single status column whose comment describes transitions
    for col_name, comment in inv.column_comments.items():
        if col_name in _WORKFLOW_COLUMN_NAMES and comment:
            transition_words = {"pending", "assigned", "frozen", "submitted",
                                "→", "->", "transition", "lifecycle"}
            if any(w in comment.lower() for w in transition_words):
                return True
    return False


def _get_status_columns(inv: TableInventory) -> list[str]:
    """
    Return column names in this table that warrant a STATUS chunk.
    Signal 1: column name is in _STATUS_COLUMN_NAMES.
    Signal 2: column comment contains status code patterns (uppercase words).
    """
    result = []
    for col_name in inv.columns:
        if col_name in _STATUS_COLUMN_NAMES:
            result.append(col_name)
            continue
        # Signal 2: comment mentions uppercase codes (e.g. 'PRIMARY | REVIEW | REVAL')
        comment = inv.column_comments.get(col_name, "")
        if comment and re.search(r"\b[A-Z_]{3,}\b\s*[|/]", comment):
            result.append(col_name)
    return result


def _is_partition_table(inv: TableInventory) -> bool:
    """
    Signal 1: @chunk:partition annotation.
    Signal 2: table is_partition=True (child partition — document parent strategy).
    Signal 3: table name is in the notable-absence override set.
    Signal 4: table comment mentions partitioning.
    """
    annotation = _get_chunk_annotation(inv)
    if annotation == "partition":
        return True
    if inv.is_partition:
        return True
    if inv.table_name in _PARTITION_NOTABLE_ABSENCE:
        return True
    if inv.comment and "partition" in inv.comment.lower():
        return True
    return False


class ChunkGenerator:
    """
    Generates all semantic chunks from parsed DDL inventory.

    Usage:
        gen    = ChunkGenerator(schema_version="abc123")
        chunks = gen.generate(tables, glossary_path="data/glossary.json")
    """

    def __init__(self, schema_version: str = "") -> None:
        self.schema_version = schema_version

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def generate(
        self,
        tables:        dict[str, TableInventory],
        # FIX-M4: full_tables provides the complete schema dict for FK_MAP
        # inbound relationship resolution.  On incremental re-ingestion,
        # `tables` is only the changed subset — passing only that to
        # _fk_map_chunk caused inbound FK paths from unchanged tables to be
        # invisible, producing incomplete FK_MAP chunks.
        # Defaults to None (falls back to `tables`) for backward compatibility
        # on full re-ingestion calls that don't supply this parameter.
        full_tables:   dict[str, TableInventory] | None = None,
        glossary_path: str = "data/glossary.json",
        examples_path: str = "data/few_shot_examples.json",
        rules_path:    str = "config/heuristics.yaml",
    ) -> list[SemanticChunk]:
        """Generate all chunk types and return the full list."""
        # FIX-M4: use full_tables for FK context if provided; otherwise fall
        # back to tables (correct for full re-ingestion where they are identical)
        context_tables = full_tables if full_tables is not None else tables

        chunks: list[SemanticChunk] = []

        for table_name, inv in tables.items():
            if inv.is_view:
                chunks.append(self._view_chunk(inv))
            else:
                chunks.append(self._table_chunk(inv))

            # FK_MAP — every table with at least one FK relationship, or that
            # is referenced by another table.
            # FIX-M4: pass context_tables (full schema) so inbound FKs from
            # unchanged tables are included in the chunk text.
            has_inbound = any(
                fk.to_table == table_name
                for other_inv in context_tables.values()
                for fk in other_inv.foreign_keys
                if not fk.is_self_referential
            )
            if inv.foreign_keys or has_inbound:
                chunks.append(self._fk_map_chunk(inv, context_tables))

            # WORKFLOW — DDL-driven detection (FIX H1)
            if _is_workflow_table(inv):
                wf_chunk = self._workflow_chunk(inv)
                if wf_chunk:
                    chunks.append(wf_chunk)

            # STATUS — DDL-driven detection (FIX H1)
            status_cols = _get_status_columns(inv)
            if status_cols:
                chunks.append(self._status_chunk(inv, status_cols))

            # AUDIT — DDL-driven detection (FIX H1)
            if _is_audit_table(inv):
                chunks.append(self._audit_chunk(inv))

            # PARTITION — DDL-driven detection (FIX H1)
            if _is_partition_table(inv):
                chunks.append(self._partition_chunk(inv))

            # INDEX — non-trivial indexes only
            notable_indexes = [
                idx for idx in inv.indexes
                if (idx.method in _NOTABLE_INDEX_METHODS
                    or idx.is_partial
                    or len(idx.columns) >= 3)
            ]
            if notable_indexes:
                chunks.append(self._index_chunk(inv, notable_indexes))

        # GLOSSARY chunks (from JSON file)
        chunks.extend(self._glossary_chunks(glossary_path))

        # BUSINESS RULE chunks (from JSON file)
        chunks.extend(self._business_rule_chunks(rules_path))

        # FEW_SHOT chunks (from JSON file — Qdrant only)
        chunks.extend(self._few_shot_chunks(examples_path))

        logger.info(
            event="generation_complete",
            total=len(chunks),
            by_type={ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType},
        )
        return chunks

    # ─────────────────────────────────────────────────────────────────────
    # Chunk builders
    # ─────────────────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        text:              str,
        chunk_type:        ChunkType,
        table_name:        str        = "",
        referenced_tables: list[str]  | None = None,
        domain_tags:       list[str]  | None = None,
        fk_neighbors:      list[str]  | None = None,
        **extra,
    ) -> SemanticChunk:
        # FIX M3 (Medium) — original used uuid.uuid5(NAMESPACE_DNS, text[:64]).
        # Two chunks with different full texts but identical first 64 characters
        # (e.g. two TABLE chunks for tables with the same prefix) got the same ID,
        # causing silent overwrites in Qdrant.
        # FIX: derive the ID from SHA-256 of the full text + type + table_name.
        # This is both collision-free and stable (same content always same ID).
        fingerprint = hashlib.sha256(
            f"{table_name}:{chunk_type.value}:{text}".encode("utf-8")
        ).hexdigest()
        # Format as a valid UUID string for Qdrant compatibility
        chunk_id = str(uuid.UUID(fingerprint[:32]))

        return SemanticChunk(
            text               = text.strip(),
            chunk_type         = chunk_type,
            table_name         = table_name,
            referenced_tables  = referenced_tables or ([table_name] if table_name else []),
            domain_tags        = domain_tags or [],
            fk_neighbors       = fk_neighbors or [],
            chunk_id           = chunk_id,
            schema_version     = self.schema_version,
            **extra,
        )

    def _table_chunk(self, inv: TableInventory) -> SemanticChunk:
        """
        TABLE chunk: purpose, columns, PK, notable constraints, JSONB docs.
        Written as domain narrative, not DDL.
        """
        lines = [f"TABLE: {inv.table_name}"]

        if inv.comment:
            lines.append(f"\nPurpose:\n{inv.comment}")

        # Columns
        lines.append("\nColumns:")
        for col_name, col in inv.columns.items():
            pk_tag  = " [PK]" if col.is_pk else ""
            null_tag= "" if col.nullable else " [NOT NULL]"
            comment = f" — {col.comment}" if col.comment else ""
            jsonb_note = " [JSONB — see key schema below]" if col.has_jsonb else ""
            lines.append(f"  {col_name}: {col.data_type}{pk_tag}{null_tag}{comment}{jsonb_note}")

        # JSONB column documentation (important — small models fail on JSONB without guidance)
        jsonb_cols = [c for c in inv.columns.values() if c.has_jsonb]
        if jsonb_cols:
            lines.append("\nJSONB Column Notes:")
            for col in jsonb_cols:
                lines.append(f"  {col.name}: access scalar values with ->>, typed cast required.")
                lines.append(f"    Example: {inv.table_name}.{col.name}->>'key_name'")
                lines.append(f"    For typed access: ({inv.table_name}.{col.name}->>'key_name')::INTEGER")
                lines.append(f"    Never use -> for direct comparison — always use ->> then cast.")

        # FK summary (brief — full FK paths in FK_MAP chunk)
        if inv.foreign_keys:
            lines.append("\nForeign Key References:")
            for fk in inv.foreign_keys:
                if not fk.is_self_referential:
                    lines.append(f"  {fk.from_col} → {fk.to_table}.{fk.to_col}")
                else:
                    lines.append(f"  {fk.from_col} → self ({inv.table_name}) — hierarchy/chain reference")

        domain_tags = [inv.table_name]
        if any(c.has_jsonb for c in inv.columns.values()):
            domain_tags.append("jsonb")

        return self._make_chunk(
            text              = "\n".join(lines),
            chunk_type        = ChunkType.TABLE,
            table_name        = inv.table_name,
            referenced_tables = [inv.table_name],
            domain_tags       = domain_tags,
            fk_neighbors      = [fk.to_table for fk in inv.foreign_keys if not fk.is_self_referential],
        )

    def _view_chunk(self, inv: TableInventory) -> SemanticChunk:
        """VIEW chunk: semantics, underlying tables, use cases."""
        lines = [f"VIEW: {inv.table_name}"]
        if inv.comment:
            lines.append(f"\nDescription:\n{inv.comment}")
        lines.append("\nUse this view for queries that require: (see view definition in DDL)")
        lines.append("Query directly as you would a table — no joins required through this view.")

        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.VIEW,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "view"],
        )

    def _fk_map_chunk(
        self,
        inv:    TableInventory,
        tables: dict[str, TableInventory],
    ) -> SemanticChunk:
        """
        FK_MAP chunk: explicit join paths expressed as SQL JOIN patterns.
        This is the primary chunk type for join generation quality.
        """
        lines = [f"FK_MAP: {inv.table_name} — Join Paths"]
        lines.append("\nOutbound FK relationships (how to JOIN from this table):")

        all_referenced: set[str] = {inv.table_name}

        for fk in inv.foreign_keys:
            all_referenced.add(fk.to_table)
            if fk.is_self_referential:
                lines.append(
                    f"\n  Self-reference via {fk.from_col}:\n"
                    f"  JOIN {fk.from_table} parent ON parent.{fk.to_col} = child.{fk.from_col}"
                )
            else:
                lines.append(
                    f"\n  {fk.from_col} → {fk.to_table}.{fk.to_col}:\n"
                    f"  JOIN {fk.to_table} ON {fk.to_table}.{fk.to_col} = {inv.table_name}.{fk.from_col}"
                )

        # Inbound FKs (tables that reference this table)
        inbound = [
            (other_name, fk)
            for other_name, other_inv in tables.items()
            for fk in other_inv.foreign_keys
            if fk.to_table == inv.table_name and not fk.is_self_referential
        ]
        if inbound:
            lines.append("\nInbound FK relationships (tables that JOIN to this table):")
            for other_name, fk in inbound:
                all_referenced.add(other_name)
                lines.append(
                    f"  {other_name}.{fk.from_col} → {inv.table_name}.{fk.to_col}:\n"
                    f"  JOIN {inv.table_name} ON {inv.table_name}.{fk.to_col} = {other_name}.{fk.from_col}"
                )

        fk_neighbors = [fk.to_table for fk in inv.foreign_keys if not fk.is_self_referential]
        fk_neighbors += [other for other, _ in inbound]

        return self._make_chunk(
            text               = "\n".join(lines),
            chunk_type         = ChunkType.FK_MAP,
            table_name         = inv.table_name,
            referenced_tables  = sorted(all_referenced),
            domain_tags        = [inv.table_name, "join", "foreign_key"],
            fk_neighbors       = list(set(fk_neighbors)),
        )

    def _workflow_chunk(self, inv: TableInventory) -> SemanticChunk | None:
        """
        WORKFLOW chunk: state machine description for tables with lifecycle status.
        Extracted from column comments and table comments containing workflow language.
        """
        workflow_text = ""

        # Build from comments that describe transitions
        status_comment_lines = []
        for col_name, comment in inv.column_comments.items():
            if any(kw in comment.lower() for kw in
                   ["status", "transition", "pending", "approved", "frozen",
                    "submitted", "assigned", "progress"]):
                status_comment_lines.append(f"  {col_name}: {comment}")

        if not status_comment_lines and not inv.comment:
            return None

        lines = [f"WORKFLOW: {inv.table_name}"]
        if inv.comment:
            lines.append(f"\n{inv.comment}")
        if status_comment_lines:
            lines.append("\nStatus / Lifecycle Fields:")
            lines.extend(status_comment_lines)

        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.WORKFLOW,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "workflow", "status", "lifecycle"],
        )

    def _status_chunk(self, inv: TableInventory, status_cols: list[str]) -> SemanticChunk:
        """
        STATUS chunk: enum value semantics for status columns.
        Critical for the model to understand what W, P, THIRD, BLOCKED etc. mean.
        """
        lines = [f"STATUS CODES: {inv.table_name}"]

        for col_name in status_cols:
            # FIX-M3: the original fallback constructed a throwaway ColumnInfo
            # object (ColumnInfo(col_name, "").comment) just to read .comment.
            # ColumnInfo.comment is already populated from column_comments by
            # the DDL parser, so column_comments is the canonical source.
            # The throwaway construction added unnecessary object allocation
            # and used positional args that could break on dataclass changes.
            comment = inv.column_comments.get(col_name, "") or \
                      getattr(inv.columns.get(col_name), "comment", "")
            lines.append(f"\n{col_name}:")
            if comment:
                lines.append(f"  {comment}")

        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.STATUS,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "status", "enum", "code"],
        )

    def _audit_chunk(self, inv: TableInventory) -> SemanticChunk:
        """AUDIT chunk: history / audit table semantics."""
        lines = [
            f"AUDIT TABLE: {inv.table_name}",
            "",
            inv.comment or f"{inv.table_name} is an audit/history table.",
            "",
            "Query pattern: SELECT from this table to see historical records.",
            "Never JOIN from this table as a primary fact table.",
            "ORDER BY created_at DESC for most recent changes first.",
        ]
        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.AUDIT,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "audit", "history", "log"],
        )

    def _partition_chunk(self, inv: TableInventory) -> SemanticChunk:
        """PARTITION chunk: partitioning strategy and query implications."""
        lines = [f"PARTITION STRATEGY: {inv.table_name}"]
        if inv.comment:
            lines.append(f"\n{inv.comment}")

        if inv.table_name == "audit_log":
            lines.extend([
                "",
                "Partitioning: Hierarchical LIST(is_critical) → RANGE(created_at)",
                "  Critical partitions: 5-year retention, yearly sub-partitions",
                "  Normal partitions:   90-day retention, monthly sub-partitions",
                "",
                "Query tip: Always include is_critical = TRUE/FALSE in WHERE clause",
                "to enable partition pruning. Without it, PostgreSQL scans all partitions.",
                "Example: WHERE is_critical = TRUE AND created_at >= '2024-01-01'",
            ])
        elif inv.table_name == "evaluation_marks":
            lines.extend([
                "",
                "Design decision: evaluation_marks is NOT partitioned (Design Decision D-3).",
                "Reason: retention is keyed off board.closed_at, not evaluation_marks.created_at.",
                "Partition drops by date would not align with board-based retention logic.",
                "",
                "~20 million rows at 5-year steady state.",
                "Always filter by board_id or attempt_id to avoid full table scans.",
                "Indexed on: attempt_id, question_id, is_final.",
            ])

        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.PARTITION,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "partition", "retention", "performance"],
        )

    def _index_chunk(self, inv: TableInventory, notable_indexes: list) -> SemanticChunk:
        """INDEX chunk: non-trivial index descriptions with access pattern guidance."""
        lines = [f"INDEXES: {inv.table_name} — Notable Access Patterns"]
        for idx in notable_indexes:
            tags = []
            if idx.is_unique:  tags.append("UNIQUE")
            if idx.is_partial: tags.append("PARTIAL")
            if idx.method != "btree": tags.append(idx.method.upper())

            tag_str = f" [{', '.join(tags)}]" if tags else ""
            cols    = ", ".join(idx.columns)
            lines.append(f"\n  {idx.name}{tag_str}")
            lines.append(f"  Columns: {cols}")
            if idx.condition:
                lines.append(f"  WHERE: {idx.condition}")
            if idx.comment:
                lines.append(f"  Note: {idx.comment}")
            if idx.method == "gin":
                lines.append("  Use: JSONB containment (@>), array overlap (&&), full-text search.")
            elif idx.method == "gist":
                lines.append("  Use: Range exclusion, geometric/temporal overlap queries.")

        return self._make_chunk(
            text       = "\n".join(lines),
            chunk_type = ChunkType.INDEX,
            table_name = inv.table_name,
            domain_tags= [inv.table_name, "index", "performance"],
        )

    # ─────────────────────────────────────────────────────────────────────
    # External data loaders
    # ─────────────────────────────────────────────────────────────────────

    def _glossary_chunks(self, glossary_path: str) -> list[SemanticChunk]:
        """
        Load domain glossary from JSON and produce one GLOSSARY chunk per term.

        Expected JSON format:
        [
          { "term": "URN",
            "definition": "Unique Registration Number ...",
            "related_tables": ["exam_student_mapping", "answer_script"],
            "domain_tags": ["identity"] }
        ]
        """
        path = Path(glossary_path)
        if not path.exists():
            logger.warning(component="chunk_generator", event="glossary_not_found", path=glossary_path)
            return []

        entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[SemanticChunk] = []

        for entry in entries:
            term    = entry.get("term", "")
            defn    = entry.get("definition", "")
            related = entry.get("related_tables", [])
            tags    = entry.get("domain_tags", [])

            text = f"GLOSSARY: {term}\n\nDefinition: {defn}"
            if related:
                text += f"\n\nRelated tables: {', '.join(related)}"

            chunks.append(self._make_chunk(
                text               = text,
                chunk_type         = ChunkType.GLOSSARY,
                table_name         = "",
                referenced_tables  = related,
                domain_tags        = [term.lower()] + tags,
            ))

        logger.info(component="chunk_generator", event="glossary_loaded", count=len(chunks))
        return chunks

    def _few_shot_chunks(self, examples_path: str) -> list[SemanticChunk]:
        """
        Load pre-seeded NL→SQL examples from JSON.
        FEW_SHOT chunks are stored in Qdrant only (semantic similarity retrieval).

        Expected JSON format:
        [
          { "nl": "Show all scripts pending third evaluation in board 5",
            "sql": "SELECT ...",
            "intent": "workflow_state",
            "explanation": "..." }
        ]
        """
        path = Path(examples_path)
        if not path.exists():
            logger.warning(component="chunk_generator", event="few_shot_not_found", path=examples_path)
            return []

        examples: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[SemanticChunk] = []

        for ex in examples:
            nl      = ex.get("nl", "")
            sql     = ex.get("sql", "")
            intent  = ex.get("intent", "")
            expl    = ex.get("explanation", "")

            # Embed the NL question as the primary text (what gets semantically matched)
            text = f"EXAMPLE: {nl}\n\nSQL:\n{sql}"
            if expl:
                text += f"\n\nExplanation: {expl}"

            chunks.append(self._make_chunk(
                text        = text,
                chunk_type  = ChunkType.FEW_SHOT,
                domain_tags = [intent, "example"],
                nl_question = nl,
                expected_sql= sql,
                intent      = intent,
            ))

        logger.info(component="chunk_generator", event="few_shot_loaded", count=len(chunks))
        return chunks

    def _business_rule_chunks(self, rules_path: str) -> list[SemanticChunk]:
        """Load business rules from YAML and produce one BUSINESS_RULE chunk per entry."""
        import yaml
        path = Path(rules_path)
        if not path.exists():
            logger.warning(component="chunk_generator", event="business_rules_not_found", path=rules_path)
            return []

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rules = data.get('business_rules', [])
        chunks = []

        for rule in rules:
            concept = rule.get("domain_concept", "")
            mapping = rule.get("sql_mapping", "")
            tables = rule.get("table_context", "")
            desc = rule.get("description", "")

            # This format is what the LLM will see when the retriever fetches it
            text = f"BUSINESS RULE: {concept}\n\nStrict SQL Mapping: {mapping}\nContext: {tables}\nReason: {desc}"

            # Extract table names from table_context for graph association
            referenced = [t.strip() for t in tables.split(",")] if tables else []

            chunks.append(self._make_chunk(
                text               = text,
                chunk_type         = ChunkType.BUSINESS_RULE,
                referenced_tables  = referenced,
                domain_tags        = [concept.lower(), "business_rule", "guardrail"],
            ))

        logger.info(component="chunk_generator", event="business_rules_loaded", count=len(chunks))
        return chunks
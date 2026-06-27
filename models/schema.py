"""
models/schema.py
────────────────
Shared dataclasses and enums used throughout the NL→SQL pipeline.
Using dataclasses (not Pydantic models) for internal objects —
Pydantic is reserved for config and API boundaries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Chunk types produced during Phase 1A ingestion
# ─────────────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TABLE      = "TABLE"        # Table purpose, columns, business notes
    VIEW       = "VIEW"         # View semantics and use cases
    FK_MAP     = "FK_MAP"       # Join paths and FK relationships
    WORKFLOW   = "WORKFLOW"     # Status transitions and lifecycle
    STATUS     = "STATUS"       # Enum value semantics
    INDEX      = "INDEX"        # Index access patterns (non-obvious only)
    AUDIT      = "AUDIT"        # History/audit table semantics
    PARTITION  = "PARTITION"    # Partitioning strategy and retention
    GLOSSARY   = "GLOSSARY"     # Domain term definitions
    FEW_SHOT   = "FEW_SHOT"     # NL→SQL example pairs (Qdrant only)
    BUSINESS_RULE = "BUSINESS_RULE"  # Guardrails and domain mappings


# ─────────────────────────────────────────────────────────────────────────────
# Schema ingestion structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForeignKey:
    """A single FK relationship from a source column to a target table/column."""
    from_table: str
    from_col:   str
    to_table:   str
    to_col:     str
    constraint_name: str = ""

    @property
    def is_self_referential(self) -> bool:
        """Self-refs are filtered from the graph (kept in semantic chunks only)."""
        return self.from_table == self.to_table


@dataclass
class ColumnInfo:
    name:        str
    data_type:   str
    nullable:    bool   = True
    is_pk:       bool   = False
    default:     str    = ""
    comment:     str    = ""
    has_jsonb:   bool   = False   # flagged for special chunk documentation
    # CHECK(col IN ('A','B',...)) enum values, if the column has one.
    # None means "no enum constraint" (not "empty set") — keep this
    # distinction so the validator can tell "no constraint to check"
    # apart from "constrained to nothing".
    allowed_values: set[str] | None = None


@dataclass
class IndexInfo:
    name:       str
    table_name: str
    columns:    list[str]
    is_unique:  bool   = False
    is_partial: bool   = False
    method:     str    = "btree"   # btree / gin / gist / hash
    condition:  str    = ""        # partial index WHERE clause
    comment:    str    = ""


@dataclass
class TableInventory:
    """Complete structural inventory of one table, produced by DDL parser."""
    table_name:   str
    columns:      dict[str, ColumnInfo]    = field(default_factory=dict)
    foreign_keys: list[ForeignKey]         = field(default_factory=list)
    indexes:      list[IndexInfo]          = field(default_factory=list)
    comment:      str                      = ""
    is_view:      bool                     = False
    is_partition: bool                     = False
    partition_of: str                      = ""   # parent partition table name
    triggers:     list[str]                = field(default_factory=list)
    # Column-level comments: column_name → comment text
    column_comments: dict[str, str]        = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic chunk — the unit stored in Qdrant / OpenSearch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticChunk:
    """
    A single retrievable unit of schema knowledge.

    text              — the human-readable content embedded and indexed
    chunk_type        — determines retrieval routing and context priority
    table_name        — primary table this chunk describes (may be empty for GLOSSARY)
    referenced_tables — ALL tables mentioned; used for DDL-change invalidation
    domain_tags       — searchable metadata labels for metadata filtering
    fk_neighbors      — adjacent tables in the FK graph (for graph expansion hints)
    chunk_id          — stable UUID; deterministic from (table_name, chunk_type, text[:64])
    schema_version    — DDL hash at ingestion time
    """
    text:               str
    chunk_type:         ChunkType
    table_name:         str                  = ""
    referenced_tables:  list[str]            = field(default_factory=list)
    domain_tags:        list[str]            = field(default_factory=list)
    fk_neighbors:       list[str]            = field(default_factory=list)
    chunk_id:           str                  = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version:     str                  = ""
    # FEW_SHOT specific fields
    nl_question:        str                  = ""
    expected_sql:       str                  = ""
    intent:             str                  = ""

    def to_payload(self) -> dict[str, Any]:
        """Serialise to Qdrant / OpenSearch payload dict."""
        return {
            "chunk_id":          self.chunk_id,
            "chunk_type":        self.chunk_type.value,
            "table_name":        self.table_name,
            "referenced_tables": self.referenced_tables,
            "domain_tags":       self.domain_tags,
            "fk_neighbors":      self.fk_neighbors,
            "schema_version":    self.schema_version,
            "text":              self.text,
            "nl_question":       self.nl_question,
            "expected_sql":      self.expected_sql,
            "intent":            self.intent,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SemanticChunk":
        return cls(
            text               = payload.get("text", ""),
            chunk_type         = ChunkType(payload.get("chunk_type", "TABLE")),
            table_name         = payload.get("table_name", ""),
            referenced_tables  = payload.get("referenced_tables", []),
            domain_tags        = payload.get("domain_tags", []),
            fk_neighbors       = payload.get("fk_neighbors", []),
            chunk_id           = payload.get("chunk_id", str(uuid.uuid4())),
            schema_version     = payload.get("schema_version", ""),
            nl_question        = payload.get("nl_question", ""),
            expected_sql       = payload.get("expected_sql", ""),
            intent             = payload.get("intent", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query understanding structures
# ─────────────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    AGGREGATION    = "aggregation"
    LOOKUP         = "lookup"
    COMPARISON     = "comparison"
    WORKFLOW_STATE = "workflow_state"
    TIME_SERIES    = "time_series"
    UNKNOWN        = "unknown"


@dataclass
class ParsedQuery:
    """Output of the query understanding layer."""
    original:           str
    normalised:         str               # full string including any refinement markers
    intent:             QueryIntent
    entities:           list[str]         # table names extracted
    status_codes:       list[str]         # e.g. ["FROZEN", "NOT_ASSIGNED"]
    domain_terms:       list[str]         # glossary terms found in query
    is_ambiguous:       bool              = False
    clarifications:     list[str]         = field(default_factory=list)
    confidence:         float             = 1.0
    # Issue 5: clean_query has REFINED_MARKER / VALUE_MARKER stripped.
    # runner.py uses this for retrieval + prompt so the LLM never sees the
    # CLI's internal marker strings. Default = normalised (no markers present).
    clean_query:        str               = ""
    # Extracted suffix after the marker (e.g. "below 40%" or the chosen option).
    # prompt_builder injects this as a [CLARIFICATION] block when present.
    clarification_note: str | None        = None
    # RapidFuzz course-code match result. None when no course code found or
    # rapidfuzz not installed. runner.py passes to prompt_builder for JOIN hint.
    course_code_match:  object            = None  # CourseCodeMatch | None
    # Label-filter hints: alphanumeric identifiers (e.g. "MBA101") found paired
    # with entity keywords (e.g. "course id", "board"). Each entry is a dict:
    #   {"raw": "MBA101", "table": "academic_unit", "column": "code",
    #    "context": "course id MBA101"}
    # Injected into prompt [FILTER HINTS] and correction prompts so the LLM
    # never conflates a text label with an integer PK.
    label_filters:      list[dict]        = field(default_factory=list)

    def __post_init__(self) -> None:
        # Default clean_query to normalised when not explicitly set
        if not self.clean_query:
            self.clean_query = self.normalised


# ─────────────────────────────────────────────────────────────────────────────
# Generation output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedSQL:
    """Structured output contract from the LLM."""
    sql:          str
    tables_used:  list[str]    = field(default_factory=list)
    confidence:   float        = 0.0
    explanation:  str          = ""
    raw_output:   str          = ""   # pre-parse LLM output for debugging
    prompt_tokens:     int | None = None
    completion_tokens: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Validation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:  bool
    step:    str    = ""    # which step failed
    message: str    = ""
    sql:     str    = ""    # possibly modified SQL (e.g. tenant filter injected)


# ─────────────────────────────────────────────────────────────────────────────
# Final query result returned to the user
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """End-to-end result returned to the CLI."""
    nl_query:       str
    sql:            str
    # explanation is str for normal results, list[str] for ambiguous results.
    # interface.py checks isinstance(explanation, list) to route to
    # _handle_ambiguous() vs display_result().
    explanation:    str | list
    tables_used:    list[str]
    confidence:     float
    intent:         str
    rows:           list[dict[str, Any]]   = field(default_factory=list)
    row_count:      int                    = 0
    dry_run:        bool                   = False
    retries:        int                    = 0
    error:          str                    = ""
    success:        bool                   = True
    # Observability fields populated by pipeline runner
    latency_ms:     dict[str, float]       = field(default_factory=dict)
    retrieval_meta: dict[str, Any]         = field(default_factory=dict)
    # MCP corpus entry ID — set when USE_MCP_SERVERS=true and failure is logged.
    # CLI :correct command uses this to call save_correction() without scanning failures/.
    failure_entry_id: str | None           = None
    # Structural (non-LLM) confidence signal from the NL→requirements audit.
    # Populated by Step 5.8 (logical_audit.py L6/L7).  None means "no NL
    # requirements were extractable, so no signal".  A value < 1.0 means
    # the SQL is missing some constraint or output column the NL asked
    # for — surfaced to the CLI as an objective secondary confidence.
    requirement_coverage: float | None     = None
    # Human-readable list of which NL requirements the SQL failed to
    # satisfy (e.g. "constraint:enum=expired", "output:student name").
    # Empty when requirement_coverage is 1.0 or None.  Used by batch_run.py
    # and by the retry-loop correction prompt builder for targeted feedback.
    coverage_misses: list[str]             = field(default_factory=list)
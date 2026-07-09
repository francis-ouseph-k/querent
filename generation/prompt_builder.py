"""
generation/prompt_builder.py
──────────────────────────────
Constructs the structured prompt for the LLM.

╔══════════════════════════════════════════════════════════════════════════════╗
║  PROMPT ENGINEERING GUIDE — Read this to understand every design decision  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. PROMPT STRUCTURE
   The prompt is divided into named sections. Section ORDER matters because
   transformer attention is not uniform — tokens at the START and END of the
   context window receive the strongest attention weights (the "lost in the
   middle" effect, documented by Liu et al., 2023).  We place the most
   critical information (system rules, schema) at the START and the user
   question at the END.  Lower-priority context (indexes, audit, partition)
   goes in the middle where attention is weakest.

   Section order:
     [SYSTEM]       → Instructions and rules (highest attention — start)
     [SCHEMA]       → TABLE/VIEW DDL (critical — placed immediately after)
     [JOIN MAP]     → FK relationships (attention still high here)
     [WORKFLOW]     → Status lifecycle semantics
     [STATUS MODEL] → Fixed block — injected conditionally
     [POLYMORPHISM] → Academic unit type hierarchy — injected conditionally
     [GLOSSARY]     → Domain vocabulary definitions
     [JOINS]        → FK graph Steiner tree paths
     [JOIN RECIPES] → Pre-tested multi-table JOIN patterns
     [ADDITIONAL]   → INDEX/AUDIT/PARTITION chunks (lowest priority — middle)
     [EXAMPLES]     → Few-shot NL→SQL pairs
     [TENANT]       → RLS/security context
     [FILTER HINTS] → Label-filter warnings for text codes
     [CLARIFICATION]→ Disambiguation context
     [QUERY]        → The user's question (high attention — end)

2. FIXED vs RETRIEVED BLOCKS
   Some blocks are "fixed" (hard-coded in this file) and some are "retrieved"
   (fetched from Qdrant/OpenSearch at query time based on semantic similarity).

   Fixed blocks are used when:
   - The information is needed for MOST queries (e.g. status model)
   - Retrieval might miss it (e.g. a query about "scripts" might not
     retrieve the status model chunk if the word "status" isn't in the query)
   - The cost of missing it is HIGH (hallucination of phantom columns)

   Retrieved blocks are used when:
   - The information is query-specific (e.g. which TABLE chunks to include)
   - Including it for every query would waste the token budget

3. CONDITIONAL INJECTION
   Fixed blocks are expensive (~100-160 tokens each).  To save budget on
   queries that don't need them, we inject them ONLY when the query's
   entity tables overlap with a trigger set.  For example, the status model
   block is only injected when the query touches answer_script, evaluation_
   attempt, etc.  A query about question_paper or academic_unit skips it.

4. TOKEN BUDGET
   The LLM has a 16,384-token context window.  The prompt should use
   ~8,000-10,000 tokens, leaving ~6,000-8,000 for the model's output
   (JSON with schema_reasoning + SQL + explanation).  If the prompt
   exceeds 90% of the context window, a warning is logged.

5. OUTPUT CONTRACT
   The model is instructed to respond with ONLY a JSON object.  This is
   a "structured output" pattern — by constraining the output format,
   we make parsing deterministic (json.loads) rather than relying on
   fragile regex extraction.  The JSON includes:
   - schema_reasoning: chain-of-thought (helps the model "think before
     writing SQL" — this is the CoT technique)
   - sql: the actual query
   - tables_used: for validation cross-check
   - confidence: self-assessed score
   - explanation: human-readable summary

6. FINE-TUNING ALIGNMENT
   The Phase 2 fine-tuning dataset MUST use the exact same prompt format.
   If you change section headers (e.g. "=== SCHEMA CONTEXT ==="), order,
   or the JSON output contract, the fine-tuned model will produce outputs
   that don't match what it learned during training.  This is called
   "prompt distribution mismatch" and silently degrades accuracy.

"Lost in the middle" mitigation:
  - Most critical schema chunks at the START of [SCHEMA]
  - Lower-priority chunks (INDEX, AUDIT, PARTITION) at the END
  This is enforced by the retrieval orchestrator's ordering before
  the prompt builder receives the chunk list.

IMPORTANT: If you change this format, you MUST change the fine-tuning
dataset format to match. Divergence between inference and training prompts
degrades fine-tuning generalisation.
"""

from __future__ import annotations

import re

from models.schema import ChunkType, ParsedQuery, SemanticChunk
from utils.logging_config import get_logger
from utils.heuristics import HEURISTICS

logger = get_logger(__name__)

from utils.tokenizer import count_tokens as _count_tokens


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — the "persona + rules" block
# ══════════════════════════════════════════════════════════════════════════════
#
# PROMPT ENGINEERING PRINCIPLE: "Role Prompting" + "Constraint Injection"
# ─────────────────────────────────────────────────────────────────────────
# The system prompt does two things:
#   1. Sets the model's ROLE ("expert PostgreSQL 16 query writer") — this
#      activates the model's SQL-writing capabilities more reliably than
#      asking a generic assistant.  Research shows role prompting improves
#      accuracy by 5-15% on domain-specific tasks.
#   2. Injects HARD CONSTRAINTS (Rules 1-12) — these are guardrails that
#      prevent known failure patterns.  Each rule exists because the model
#      was observed making that specific mistake in batch evaluation.
#
# RULE DESIGN PRINCIPLES:
#   - Rules are NUMBERED so the correction prompt can reference them by number
#   - Rules are ORDERED by importance (most critical first)
#   - Rules use imperative language ("NEVER", "ALWAYS", "Use ONLY")
#   - Rules include EXAMPLES of the correct pattern where helpful
#   - Rule 10 uses a BLOCKLIST approach — explicitly naming phantom columns
#     that do NOT exist.  This is more effective than a generic "don't invent
#     columns" instruction because the model's star-schema heuristics cause
#     it to repeatedly hallucinate the same specific columns.
#
# WHY NOT MORE RULES?
#   Each rule costs tokens (~30-50 per rule) and dilutes attention on other
#   rules.  We only add a rule when:
#   1. The failure pattern is SYSTEMATIC (appears in >5% of batch queries)
#   2. The failure CANNOT be caught by the validator (or catching it wastes a retry)
#   3. The rule is ACTIONABLE (tells the model exactly what to do instead)
#
#   - Rules 8 & 9: Anti-join and percentage patterns — removes logical errors
#     on aggregations and negations that are hard for the validator to auto-fix.
#   - Rule 10 (WARNING): Intercepts LLM star-schema heuristics by explicitly
#     stating which phantom columns do NOT exist and where they actually are.
#   - Rule 11: Constrains the database function signature to prevent argument
#     count or type mismatch hallucinations.
#   - Rule 12: Prevents nested aggregate functions (a common 3B model error).
#   - Rule 3 & 14: Mandates explicit table aliases on all column references
#     and prohibits SELECT-projection aliases inside GROUP BY/HAVING clauses.
#     Fixes PostgreSQL dialect execution errors and "Pattern C" ambiguous column 
#     errors (e.g. Q63 undefined aliases).
import yaml
from pathlib import Path

# Load externalized prompts
_PROMPTS_FILE = Path(__file__).parent.parent / "config" / "prompts.yaml"
try:
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        _PROMPTS = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load prompts from {_PROMPTS_FILE}: {e}")
    _PROMPTS = {"blocks": {}, "join_recipes": {}}

_SYSTEM_PROMPT = _PROMPTS.get("system_prompt", "")

# ── Training-only system prompt ───────────────────────────────────────────────
# The full _SYSTEM_PROMPT (~926 tok) embeds the D-23 surrogate-key rulebook,
# per-table FK enumeration and exact status codes. At serve time that context is
# cheap insurance; at TRAIN time it (a) overflows the 1024-token window — the
# reserve (system+question+SQL) alone is 1068 median — truncating the SQL label,
# and (b) teaches the model to depend on the very rulebook fine-tuning is meant to
# internalise into the weights. This short variant keeps only the task role, the
# schema-agnostic SQL-hygiene rules (aliasing / literals / column qualification —
# the ones that map to the observed alias/placeholder/GROUP-BY failure modes), and
# the JSON output contract. Schema-specific knowledge is taught by the gold
# examples instead. Used ONLY by fine_tuning/preprocess/build.format_pairs; the
# serve path is unchanged, so parity is restored later by shortening the serve
# prompt once the fine-tune is verified to have learned D-23.
_TRAIN_SYSTEM_PROMPT = _PROMPTS.get("train_system_prompt", "").strip() or (
    "You are an expert PostgreSQL 16 query writer for the Digital Evaluation System.\n\n"
    "RULES:\n"
    "1. Use ONLY tables/columns from the schema context. Generate ONLY SELECT statements.\n"
    "2. Use explicit JOIN ... ON with a unique, non-reserved alias per table; qualify EVERY "
    "column with its table alias (in SELECT, WHERE, GROUP BY, ORDER BY, HAVING, and subqueries).\n"
    "3. Use literal values — never parameter placeholders ($1, :param). Filter named entities by "
    "name, e.g. WHERE qp.title ILIKE '%Algorithms%'.\n"
    "4. Never reference a SELECT projection alias inside the same SELECT list, GROUP BY, or HAVING "
    "— repeat the full expression or use a CTE.\n\n"
    "Output — respond with ONLY this JSON, nothing else:\n"
    "{\n"
    '  "schema_reasoning": "Step-by-step: tables, columns, joins I will use.",\n'
    '  "sql": "SELECT ...",\n'
    '  "tables_used": ["table1"],\n'
    '  "confidence": 0.85,\n'
    '  "explanation": "What this query does."\n'
    "}\n"
)

# ── Conditional rule blocks ──────────────────────────────────────────────────
# Rules conditionally injected based on query intent or trigger words.
_ANTI_JOIN_RULES_BLOCK = _PROMPTS.get("blocks", {}).get("anti_join", "")
_ANTI_JOIN_TRIGGER_WORDS: frozenset[str] = frozenset(HEURISTICS.get('anti_join_triggers', []))

_AGGREGATION_RULES_BLOCK = _PROMPTS.get("blocks", {}).get("aggregation", "")

_TEMPORAL_RULES_BLOCK = _PROMPTS.get("blocks", {}).get("temporal", "")
_TEMPORAL_TRIGGER_TABLES: frozenset[str] = frozenset(HEURISTICS.get('trigger_tables', {}).get('temporal', []))

_AGG_TRIGGER_WORDS: frozenset[str] = frozenset(HEURISTICS.get('trigger_words', {}).get('aggregation', []))
_TEMPORAL_TRIGGER_WORDS: frozenset[str] = frozenset(HEURISTICS.get('trigger_words', {}).get('temporal', []))

_LOGIC_RULES_BLOCK = _PROMPTS.get("blocks", {}).get("logic", "")

# ── Fixed Knowledge Blocks ───────────────────────────────────────────────────
# Domain knowledge blocks injected based on referenced tables.
_STATUS_MODEL_BLOCK = _PROMPTS.get("blocks", {}).get("status_model", "")
_POLYMORPHISM_BLOCK = _PROMPTS.get("blocks", {}).get("polymorphism", "")
_ENUM_APPENDIX_BLOCK = _PROMPTS.get("blocks", {}).get("enum_appendix", "")
_JSONB_METADATA_MAP = _PROMPTS.get("blocks", {}).get("jsonb_metadata_map", "")

_STATUS_TRIGGER_TABLES: frozenset[str] = frozenset(HEURISTICS.get('trigger_tables', {}).get('status', []))
_ENUM_TRIGGER_TABLES: frozenset[str] = frozenset(HEURISTICS.get('trigger_tables', {}).get('enum', []))
_JSONB_TRIGGER_TABLES: frozenset[str] = frozenset(HEURISTICS.get('trigger_tables', {}).get('jsonb', []))

_CORRECTION_PROMPT_FOOTER: list[str] = _PROMPTS.get("correction_prompt_footer", [])


def _authoritative_columns_block(error_message: str, tables) -> list[str]:
    """
    Given a validator error and the live schema inventory (`tables`: name ->
    object with a `.columns` collection), return correction-prompt lines that
    list the EXACT columns of every table named in the error. Empty list if the
    error names no known table or no inventory is available.
    """
    if not error_message or not tables:
        return []

    inv_by_name = {str(k).lower(): v for k, v in tables.items()}
    err = error_message.lower()
    found: list[str] = []

    # 1) explicit "table.column" tokens in the error
    for m in re.findall(r"\b([a-z_]+)\.[a-z_]+", err):
        if m in inv_by_name and m not in found:
            found.append(m)
    # 2) any known table name mentioned verbatim (e.g. "<t> does NOT have <c>")
    if not found:
        for name in inv_by_name:
            if re.search(rf"\b{re.escape(name)}\b", err):
                found.append(name)

    lines: list[str] = []
    for name in found[:4]:
        inv = inv_by_name.get(name)
        cols = getattr(inv, "columns", None)
        if not cols:
            continue
        col_list = ", ".join(sorted(cols))
        lines.append(f"  {name}: {col_list}")

    if not lines:
        return []

    return [
        "AUTHORITATIVE COLUMNS — the failed SQL referenced a column that does "
        "NOT exist. Use ONLY the columns listed below for these tables, and no "
        "others. Do not invent columns; if a value is not here, derive it or "
        "join to the correct table:",
        *lines,
    ]


# ── JOIN recipe block ────────────────────────────────────────────────────────
# Pre-tested multi-table JOIN patterns.
_JOIN_RECIPES: dict[str, tuple[set[str], str]] = {
    key: (set(val.get("tables", [])), val.get("pattern", ""))
    for key, val in _PROMPTS.get("join_recipes", {}).items()
}




class PromptBuilder:
    """
    Assembles the structured LLM prompt from retrieved chunks and query context.

    Usage:
        builder = PromptBuilder()
        prompt  = builder.build(
            parsed_query  = parsed,
            schema_chunks = chunks,
            join_paths    = ["JOIN board ON board.id = ea.board_id"],
            few_shots     = few_shot_chunks,
            tenant_context= "app.current_user_id = '12345'",
        )
    """

    def build(
        self,
        parsed_query:       ParsedQuery,
        schema_chunks:      list[SemanticChunk],
        join_paths:         list[str]           = None,
        few_shots:          list[SemanticChunk] = None,
        tenant_context:     str                 = "",
        clarification_note: str | None          = None,
        course_code_match                       = None,  # CourseCodeMatch | None
        label_filters:      list[dict]          = None,  # from parsed_query.label_filters
        tables:             dict                = None,  # dict[str, TableInventory] — for COLUMN CHEATSHEET
    ) -> str:
        """
        Build and return the full prompt string.

        clarification_note — extracted suffix from a refined query
          (e.g. "below 40%" from "show failed students — value: below 40%").
          Injected as a [CLARIFICATION] block immediately before [QUERY] so
          the model receives it as explicit context, not embedded in the question.

        course_code_match — RapidFuzz CourseCodeMatch result from
          query_understanding._resolve_course_codes(). When present, a note
          is injected in [CLARIFICATION] telling the model to use the canonical
          code string in its JOIN condition.

        label_filters — list of dicts from query_understanding._extract_label_filters().
          Each entry is {"raw", "table", "column", "hint", "context"}.
          Injected as a [FILTER HINTS] block so the model knows to use text
          columns (e.g. .code, .name) instead of integer PKs when the user
          supplies alphanumeric identifiers like MBA101.

        tables — full {table_name: TableInventory} map produced by the DDL
          parser.  When provided, a [COLUMN CHEATSHEET] block is injected
          immediately before the user's question.  The cheatsheet is a
          terse, positive-form per-table column list for every table whose
          schema is in scope for this query (entities + Steiner-tree
          connector tables + tables that appear in retrieved chunks).
          This block is the single highest-impact mitigation for the
          column-on-wrong-table hallucination class observed in the
          batch evaluation (22/43 = 51% of failures).  Positive-form
          enumeration is far more effective than the prompt's many
          "table X has NO column Y" negative rules — a 3B model is poor
          at applying negations across a 10k-token prompt but excellent
          at copying from a short structured list it just read.
        """
        join_paths    = join_paths    or []
        few_shots     = few_shots     or []
        label_filters = label_filters or []

        sections: list[str] = []

        # Build entity table set early — used for conditional block injection
        # and JOIN recipe matching.  `or []` guards against entities being None.
        entity_tables_set = set(
            t.lower() for t in (getattr(parsed_query, 'entities', None) or [])
        )

        # FIX-NEW-N6: deduplicate chunks by chunk_id before distributing into
        # sections.  A chunk whose type appears in multiple filter conditions
        # (or that was inserted by both mandatory-entity promotion and RRF
        # ranking) would otherwise appear twice, wasting context budget.
        seen_chunk_ids: set[str] = set()

        # Intentionally capturing and mutating the 'seen_chunk_ids' set from the outer scope 
        # so that we deduplicate across ALL chunk lists passed to this function.
        def _dedup(chunks: list[SemanticChunk]) -> list[SemanticChunk]:
            result = []
            for c in chunks:
                if c.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c.chunk_id)
                    result.append(c)
            return result

        # ══════════════════════════════════════════════════════════════════
        # SECTION ASSEMBLY — order matters for transformer attention
        # ══════════════════════════════════════════════════════════════════
        #
        # PROMPT ENGINEERING PRINCIPLE: "Primacy + Recency Bias"
        # ─────────────────────────────────────────────────────
        # Transformer models attend most strongly to tokens at the START
        # and END of the context window.  Tokens in the MIDDLE receive
        # weaker attention (the "lost in the middle" effect).
        #
        # Our ordering strategy:
        #   START (high attention): System rules -> Schema -> Join paths
        #   MIDDLE (low attention): Glossary -> Additional context
        #   END (high attention):   Examples -> Clarification -> Question
        #
        # This means the model "sees" the schema and rules clearly,
        # and has the question fresh in memory when it starts generating.

        # ── [SYSTEM] ──────────────────────────────────────────────────────
        # Always first.  Sets the model's role and constraints.
        sections.append(_SYSTEM_PROMPT)

        # ── [CONDITIONAL RULES] ──────────────────────────────────────────
        # Tiered rule injection: pattern-specific rules injected only when
        # the query matches their trigger condition.

        # Derive NL word set for trigger matching
        _nl_words = set(
            (getattr(parsed_query, 'clean_query', '') or
             getattr(parsed_query, 'normalised', '') or '').lower().split()
        )
        _intent = getattr(parsed_query, 'intent', None)
        _intent_val = _intent.value if hasattr(_intent, 'value') else str(_intent or '')

        # Anti-join rules: triggered by negation words in the NL
        if _nl_words & _ANTI_JOIN_TRIGGER_WORDS:
            sections.append(_ANTI_JOIN_RULES_BLOCK)
            sections.append("")

        # Aggregation rules: triggered by aggregation/comparison intent or
        # aggregation keywords in NL
        if _intent_val in ('aggregation', 'comparison') or (_nl_words & _AGG_TRIGGER_WORDS):
            sections.append(_AGGREGATION_RULES_BLOCK)
            sections.append("")

        # Temporal rules: triggered by temporal tables or time-related keywords
        if (entity_tables_set & _TEMPORAL_TRIGGER_TABLES) or (_nl_words & _TEMPORAL_TRIGGER_WORDS):
            sections.append(_TEMPORAL_RULES_BLOCK)
            sections.append("")

        # Domain logic rules: always injected (phantom columns, entity disambiguation)
        sections.append(_LOGIC_RULES_BLOCK)
        sections.append("")


        # PROMPT ENGINEERING PRINCIPLE: "Context Grounding"
        # The model MUST see the actual table definitions to write correct SQL.
        # These are the most important chunks — placed immediately after the
        # system prompt where attention is highest.
        table_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type in (ChunkType.TABLE, ChunkType.VIEW)
        ])
        if table_chunks:
            # Section headers like "=== SCHEMA CONTEXT ===" serve as
            # "attention anchors" — they help the model understand what
            # type of information follows, improving accuracy vs raw text.
            sections.append("=== SCHEMA CONTEXT ===")
            for chunk in table_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [JOIN MAP] — FK_MAP chunks (promoted from ADDITIONAL CONTEXT) ──
        # Placed immediately after SCHEMA so the model sees explicit JOIN
        # paths while its attention on table structure is still high.
        fk_map_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type == ChunkType.FK_MAP
        ])
        if fk_map_chunks:
            sections.append("=== JOIN MAP ===")
            for chunk in fk_map_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [BUSINESS RULES] ──────────────────────────────────────────────
        # Highly prioritized domain constraints and strict mappings.
        business_rule_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type == ChunkType.BUSINESS_RULE
        ])
        if business_rule_chunks:
            sections.append("=== BUSINESS RULES ===")
            for chunk in business_rule_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [WORKFLOW] + [STATUS] ─────────────────────────────────────────
        workflow_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type in (ChunkType.WORKFLOW, ChunkType.STATUS)
        ])
        if workflow_chunks:
            sections.append("=== WORKFLOW AND STATUS SEMANTICS ===")
            for chunk in workflow_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [STATUS MODEL] — conditionally injected when entity tables overlap ──
        # Saves ~120 tokens on queries that don't touch evaluation tables.
        # _STATUS_TRIGGER_TABLES is a module-level frozenset (allocated once).
        if entity_tables_set & _STATUS_TRIGGER_TABLES:
            sections.append(_STATUS_MODEL_BLOCK)
            sections.append("")

        # ── [POLYMORPHISM] — conditionally injected when academic_unit is relevant ──
        if 'academic_unit' in entity_tables_set or 'academic_unit_closure' in entity_tables_set:
            sections.append(_POLYMORPHISM_BLOCK)
            sections.append("")

        # ── [ENUM APPENDIX] — conditionally injected when status-filtered tables are used ──
        # Prevents wrong enum values (e.g. 'MODERATED', 'ASSIGNED' for in-progress).
        if entity_tables_set & _ENUM_TRIGGER_TABLES:
            sections.append(_ENUM_APPENDIX_BLOCK)
            sections.append("")

        # ── [JSONB MAP] — conditionally injected when JSONB-bearing tables are used ──
        # Prevents JSONB extraction from wrong table/column (e.g. OCR from scan_history).
        if entity_tables_set & _JSONB_TRIGGER_TABLES:
            sections.append(_JSONB_METADATA_MAP)
            sections.append("")

        # _TEMPORAL_BLOCK removed (P3) — vague content added noise.

        # ── [GLOSSARY] ────────────────────────────────────────────────────
        glossary_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type == ChunkType.GLOSSARY
        ])
        if glossary_chunks:
            sections.append("=== DOMAIN TERMINOLOGY ===")
            for chunk in glossary_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [JOINS] — FK graph paths ──────────────────────────────────────
        # These are computed by the NetworkX Steiner Tree traversal in
        # graph_builder.py.  They tell the model the shortest FK path
        # between the entity tables identified by query_understanding.
        # Example: "answer_script.id -> evaluation_attempt.script_id"
        if join_paths:
            sections.append("=== RELEVANT JOIN PATHS ===")
            sections.append("\n".join(join_paths))
            sections.append("")

        # ── [JOIN RECIPES] — common multi-table patterns (Change 8) ─────
        # Injected only when the query's entity tables overlap with a recipe's
        # trigger tables, keeping non-relevant prompts lean.
        # (entity_tables_set computed at top of build())
        matched_recipes: list[str] = []
        for _recipe_name, (trigger_tables, recipe_text) in _JOIN_RECIPES.items():
            if entity_tables_set & trigger_tables:
                matched_recipes.append(recipe_text)
        if matched_recipes:
            sections.append("=== COMMON JOIN RECIPES ===")
            sections.append(
                "When the question involves these tables, use these exact JOIN patterns:"
            )
            for recipe in matched_recipes:
                sections.append(recipe)
            sections.append("")

        # ── Lower-priority context (placed in the MIDDLE — lost-in-middle) ─
        # PROMPT ENGINEERING PRINCIPLE: "Attention-Aware Ordering"
        # ─────────────────────────────────────────────────────────
        # INDEX, AUDIT, and PARTITION chunks are useful but rarely critical.
        # Placing them in the middle of the prompt means they get the weakest
        # attention — which is intentional.  If the model ignores an index
        # hint, the query still works (just slower).  If it ignores a TABLE
        # chunk, the query fails entirely.
        other_chunks = _dedup([
            c for c in schema_chunks
            if c.chunk_type in (ChunkType.INDEX,
                                ChunkType.AUDIT, ChunkType.PARTITION)
        ])
        if other_chunks:
            sections.append("=== ADDITIONAL CONTEXT ===")
            for chunk in other_chunks:
                sections.append(chunk.text)
                sections.append("")

        # ── [EXAMPLES] — FEW_SHOT examples ───────────────────────────────
        # PROMPT ENGINEERING PRINCIPLE: "Few-Shot Learning"
        # ─────────────────────────────────────────────────
        # Providing 1-3 examples of correct NL->SQL pairs that are
        # semantically similar to the current query dramatically improves
        # accuracy.  The model learns the expected output FORMAT and
        # STYLE from these examples (e.g. alias conventions, JOIN order).
        #
        # These are retrieved from Qdrant by semantic similarity — so a
        # query about "honorarium per evaluator" gets examples about
        # honorarium calculations, not about scanning or moderation.
        if few_shots:
            sections.append("=== EXAMPLE QUERIES ===")
            for i, ex in enumerate(few_shots, 1):
                sections.append(f"Example {i}:")
                sections.append(f"Question: {ex.nl_question}")
                sections.append(f"SQL: {ex.expected_sql}")
                sections.append("")

        # ── [TENANT] — RLS / tenant context ──────────────────────────────
        if tenant_context:
            sections.append("=== SECURITY CONTEXT ===")
            sections.append(f"Always apply: {tenant_context}")
            sections.append("")

        # ── [FILTER HINTS] — label-filter warnings ────────────────────────
        # Injected when the query contains alphanumeric identifiers (e.g.
        # "MBA101", "CS4") paired with entity keywords ("course id", "student id").
        # These are text codes — they must be matched against VARCHAR columns
        # (.code, .name, .student_id) NOT against integer PK/FK columns (.id).
        # Placed before [CLARIFICATION] and [QUERY] for maximum attention.
        if label_filters:
            sections.append("=== FILTER HINTS ===")
            sections.append(
                "WARNING: The query contains text identifiers that look like codes, "
                "not integer IDs. Use the VARCHAR columns shown below — NOT .id (BIGINT)."
            )
            for lf in label_filters:
                sections.append(
                    f"  '{lf['raw']}' → filter using {lf['hint']}. "
                    f"Example: WHERE {lf['table']}.{lf['column']} = '{lf['raw']}'"
                )
            sections.append("")

        # ── [CLARIFICATION] — user-supplied disambiguation context ────────
        # Injected when the CLI's disambiguation loop provided extra context:
        #   - clarification_note: the user's free-text value or chosen option
        #     (e.g. "below 40%" after "show failed students — value: below 40%")
        #   - course_code_match: RapidFuzz-resolved canonical course code
        #     (e.g. raw "mba 01" → canonical "MBA01")
        # Placed immediately before [QUERY] so the model reads it last before
        # generating SQL — highest recency attention.
        clarification_lines: list[str] = []

        if clarification_note:
            clarification_lines.append(f"User clarification: {clarification_note}")

        if course_code_match is not None:
            clarification_lines.append(
                f"Course code note: the user typed '{course_code_match.raw_token}' "
                f"which resolves to canonical code '{course_code_match.canonical}' "
                f"in academic_unit.code. Use '{course_code_match.canonical}' in your "
                f"JOIN condition: JOIN academic_unit au ON au.id = <fk_col> "
                f"WHERE au.code = '{course_code_match.canonical}'."
            )

        if clarification_lines:
            sections.append("=== CLARIFICATION ===")
            sections.extend(clarification_lines)
            sections.append("")

        # ── [COLUMN CHEATSHEET] — deterministic alias→columns map ────────
        # Terse, positive-form list of the columns that actually exist on
        # each table currently in scope.  Built deterministically from the
        # TableInventory map produced by the DDL parser — independent of
        # what the retrieval pipeline ranked highest.
        #
        # WHY THIS BLOCK EXISTS:
        # The dominant failure pattern in batch evaluation
        # (22/43 = 51% of failures) is the LLM emitting a real column name
        # against the wrong table (e.g. `bc.name` when name is on
        # faculty_cache, or `sa.board_id` when board_id is reached via
        # answer_script).  The system prompt's WARNING block tells the
        # model what NOT to do (negative rules) — but a 3B model cannot
        # reliably apply negations across a 10k-token prompt.  This block
        # supplies the same information in POSITIVE form: "the only
        # columns that exist on board_coordinator are …".  Positive lists
        # of legal names are far easier for small models to copy from
        # than negative warnings to track and avoid.
        #
        # PLACEMENT:
        # Last block before [QUERY] — exploits transformer recency bias.
        # The model reads this immediately before it starts writing SQL.
        #
        # SCOPE:
        # Only tables in this query's working set are included:
        #   - parsed_query.entities       (entity tables from NL parse)
        #   - tables present in retrieved schema_chunks (TABLE/FK_MAP/etc.)
        #   - tables present in the Steiner-tree connector path
        # This keeps the block compact (~150-400 tokens typical) and
        # focused on the schema actually needed for this query.
        if tables:
            scope: set[str] = set()
            scope.update(entity_tables_set)
            for c in schema_chunks:
                if c.table_name:
                    scope.add(c.table_name.lower())
                for rt in (c.referenced_tables or []):
                    scope.add(rt.lower())
            # Also pull any table names referenced in the join_paths text
            # (e.g. "ea.script_id → answer_script.id" mentions both).
            for jp in join_paths:
                for tok in re.findall(r'[a-z][a-z0-9_]+', jp.lower()):
                    if tok in tables:
                        scope.add(tok)

            scope = {t for t in scope if t in tables}
            if scope:
                cheatsheet_lines = ["=== COLUMN CHEATSHEET ==="]
                cheatsheet_lines.append(
                    "These are the ONLY columns that exist on each table. "
                    "When you reference T.col, T must appear in this list "
                    "AND col must appear after T's colon. "
                    "If a column you need is not here, the column lives on "
                    "a different table — find the right one before using it."
                )
                # Sort: entity tables first (most likely to be the SELECT
                # target), then alphabetical for stability.
                sort_key = lambda t: (0 if t in entity_tables_set else 1, t)
                for tname in sorted(scope, key=sort_key):
                    inv = tables[tname]
                    cols = getattr(inv, 'columns', None)
                    if not cols:
                        continue
                    col_names = sorted(cols.keys())
                    cheatsheet_lines.append(f"{tname}: {', '.join(col_names)}")
                sections.extend(cheatsheet_lines)
                sections.append("")

        # ── [QUERY] — the user's clean question (markers stripped) ────────
        # PROMPT ENGINEERING PRINCIPLE: "Recency Bias"
        # ──────────────────────────────────────────────
        # The user's question is placed LAST in the prompt.  This exploits
        # the recency bias in transformer attention — the model's generation
        # starts immediately after this text, so it has the strongest
        # influence on what the model generates.  All the schema context
        # above is "background"; the question is the "foreground".
        #
        # Issue 5 fix: use parsed_query.clean_query (markers stripped) rather
        # than parsed_query.normalised which may contain "— specifically:" or
        # "— value:" implementation markers the LLM should never see.
        sections.append("=== QUESTION ===")
        sections.append(getattr(parsed_query, 'clean_query', getattr(parsed_query, 'normalised', '')))
        sections.append("")
        # This closing instruction reinforces the output contract one more time.
        # PROMPT ENGINEERING PRINCIPLE: "Bookending"
        # Repeating the output format at the END is a common technique —
        # it reminds the model of the expected format just before it starts
        # generating, reducing format violations.
        sections.append("Respond with ONLY the JSON object as specified above:")

        prompt = "\n".join(sections)

        # FIX-NEW-M3: verify assembled prompt fits in the LLM context window.
        from config.settings import settings as _settings
        prompt_tokens  = _count_tokens(prompt)
        context_limit  = _settings.llm.context_size
        warn_threshold = int(context_limit * 0.9)
        if prompt_tokens > warn_threshold:
            logger.warning(
                component="prompt_builder",
                event="prompt_near_context_limit",
                prompt_tokens=prompt_tokens,
                context_limit=context_limit,
                warn_threshold=warn_threshold,
                note="Consider reducing chunks or few-shot examples to stay within budget",
            )

        logger.debug(
            component="prompt_builder",
            event="prompt_built",
            schema_chunks=len(table_chunks),
            workflow_chunks=len(workflow_chunks),
            glossary_chunks=len(glossary_chunks),
            few_shots=len(few_shots),
            join_paths=len(join_paths),
            clarification=bool(clarification_lines),
            approx_tokens=prompt_tokens,
            approx_chars=len(prompt),
        )

        return prompt

    def build_correction_prompt(
        self,
        original_query:  str,
        failed_sql:      str,
        error_message:   str,
        schema_context:  str        = "",
        label_filters:   list[dict] = None,
        parsed_query                = None,
        # ── Full-context params (Fix 1+4+5: retry context parity) ──
        schema_chunks:   list       = None,
        join_paths:      list[str]  = None,
        few_shots:       list       = None,
        tenant_context:  str        = "",
        # ── Audit-driven correction (NEW) ─────────────────────────────
        # When the logical audit produced coverage_misses, pass them
        # here.  The correction prompt then tells the LLM the SPECIFIC
        # requirements its previous SQL failed to satisfy, instead of
        # only echoing the generic validator error.  Format of each
        # entry: "constraint:<kind>=<raw>" or "output:<name>" or
        # "entity_type:<id>=<expected>-><actual>".
        audit_misses:    list[str]  = None,
        # ── Column-cheatsheet pass-through (NEW) ─────────────────────────
        # Forwarded to build() so the retry sees the same positive-form
        # alias→columns list the original attempt did.
        tables:          dict       = None,
    ) -> str:
        """
        Build a correction prompt for the retry loop.

        When schema_chunks and parsed_query are provided, rebuilds the full
        initial-quality prompt (via build()) and appends the error correction
        section.  This ensures the retry has the SAME rich context as the
        first attempt — schema DDL, join map, recipes, few-shots, glossary —
        fixing the context loss where retries previously received only a
        stripped-down 10-chunk subset (~30% of original context).

        When audit_misses is non-empty, the prompt ALSO includes the
        specific NL requirements the previous attempt failed to satisfy.
        This closes the loop between detection (logical_audit.py L6-L9)
        and correction — without it, the audit knows what's wrong but
        the retry has no idea what to fix.

        Falls back to the legacy stripped-down prompt when schema_chunks
        is not provided (backward compatibility).
        """
        label_filters = label_filters or []
        audit_misses  = audit_misses  or []

        # ── Full-context path (Fix 1+4+5) ────────────────────────────────
        # Rebuild the exact same prompt the LLM saw on the first attempt,
        # then append the correction section at the end.
        if schema_chunks is not None and parsed_query is not None:
            full_prompt = self.build(
                parsed_query       = parsed_query,
                schema_chunks      = schema_chunks,
                join_paths         = join_paths or [],
                few_shots          = few_shots or [],
                tenant_context     = tenant_context,
                clarification_note = getattr(parsed_query, 'clarification_note', None),
                course_code_match  = getattr(parsed_query, 'course_code_match', None),
                label_filters      = label_filters,
                tables             = tables,
            )

            # Build correction suffix to append after the full prompt
            correction_lines = [
                "",
                "=== CORRECTION REQUIRED ===",
                "The following SQL query failed validation. Fix the error and regenerate.",
                "",
                "Failed SQL:",
                failed_sql,
                "",
                f"Error: {error_message}",
                "",
            ]

            # ── Authoritative columns for the error table(s) (NEW) ───────────
            # The dominant Phase-1 failure mode is column hallucination: the
            # model invents columns (configuration.global_value, academic_unit.
            # display_name, script_page.page_count, ...) and then repeats the
            # SAME error across all retries because re-retrieval returns the
            # same chunks. Inject the EXACT, live column list for the table(s)
            # named in the error, straight from the schema inventory, right next
            # to the error. This is deterministic ground truth adjacent to the
            # mistake -- far more reliable than hoping retrieval surfaces it.
            auth_block = _authoritative_columns_block(error_message, tables)
            if auth_block:
                correction_lines.extend(auth_block)
                correction_lines.append("")

            if label_filters:
                correction_lines.append("IDENTIFIER CORRECTION:")
                correction_lines.append(
                    "The query contains text identifiers (codes/labels) that were "
                    "incorrectly matched to integer columns. Use the text columns below:"
                )
                for lf in label_filters:
                    correction_lines.append(
                        f"  '{lf['raw']}' is a text code — use {lf['hint']}. "
                        f"Correct filter: WHERE {lf['table']}.{lf['column']} = '{lf['raw']}'"
                    )
                correction_lines.append("")

            # ── Audit-driven feedback (NEW) ──────────────────────────
            # When the previous SQL passed structural validation but
            # the NL→requirements audit found missing constraints,
            # output columns, or entity-type mismatches, tell the
            # LLM SPECIFICALLY what to fix.  This is the actionable
            # piece of the audit signal — without it the LLM only
            # hears "validation failed" with no direction.
            if audit_misses:
                correction_lines.append("REQUIREMENTS NOT SATISFIED:")
                correction_lines.append(
                    "Your previous SQL failed to express the following "
                    "requirements from the question.  Fix EACH ONE in the "
                    "regenerated SQL:"
                )
                for m in audit_misses:
                    # Parse the structured miss into a human directive.
                    # Format strings come from logical_audit.py.
                    if m.startswith("constraint:"):
                        # constraint:<kind>=<raw>
                        kind_raw = m[len("constraint:"):]
                        kind, _, raw = kind_raw.partition("=")
                        if kind == "enum":
                            correction_lines.append(
                                f"  - The NL filter value '{raw}' must appear as a "
                                f"literal in the WHERE clause (uppercase form)."
                            )
                        elif kind == "time_range":
                            correction_lines.append(
                                f"  - The time-range '{raw}' must be expressed in "
                                f"the WHERE clause (use BETWEEN, INTERVAL, or "
                                f"DATE_TRUNC as appropriate)."
                            )
                        elif kind == "numeric":
                            correction_lines.append(
                                f"  - The numeric threshold from '{raw}' must "
                                f"appear in the WHERE clause."
                            )
                        elif kind == "boolean":
                            correction_lines.append(
                                f"  - The boolean condition '{raw}' must appear "
                                f"in the WHERE clause (e.g. is_active = FALSE for "
                                f"'inactive', archived_at IS NOT NULL for "
                                f"'decommissioned')."
                            )
                        elif kind == "text_like":
                            correction_lines.append(
                                f"  - The text-substring constraint '{raw}' must "
                                f"appear as an ILIKE in the WHERE clause."
                            )
                        else:
                            correction_lines.append(
                                f"  - The {kind} constraint '{raw}' is missing "
                                f"from the SQL."
                            )
                    elif m.startswith("output:"):
                        col = m[len("output:"):]
                        correction_lines.append(
                            f"  - The output column '{col}' must be projected "
                            f"in the SELECT list."
                        )
                    elif m.startswith("entity_type:"):
                        # entity_type:<id>=<expected>-><actual>
                        body = m[len("entity_type:"):]
                        ident, _, type_part = body.partition("=")
                        expected, _, actual = type_part.partition("->")
                        correction_lines.append(
                            f"  - The identifier '{ident}' refers to a "
                            f"{expected}, not {actual}.  Filter the "
                            f"academic_unit lookup with unit_type = '{expected}'."
                        )
                    else:
                        correction_lines.append(f"  - {m}")
                correction_lines.append("")

            correction_lines.extend(_CORRECTION_PROMPT_FOOTER)
            correction_suffix = "\n".join(correction_lines)

            # Replace the closing JSON instruction with the correction section
            # (the footer already contains its own JSON instruction)
            closing_marker = "Respond with ONLY the JSON object as specified above:"
            if closing_marker in full_prompt:
                idx = full_prompt.rindex(closing_marker)
                return full_prompt[:idx] + correction_suffix
            else:
                return full_prompt + "\n" + correction_suffix

        # ── Guard: full context is required ─────────────────────────────────
        # The legacy stripped-down path was the root cause of retry context
        # loss (retries received ~30% of schema context).  All callers must
        # now provide schema_chunks + parsed_query.  Failing loudly here is
        # intentional — silent degradation was the original bug.
        raise ValueError(
            "build_correction_prompt requires schema_chunks and parsed_query. "
            "The legacy stripped-down prompt path has been removed to prevent "
            "retry context loss."
        )
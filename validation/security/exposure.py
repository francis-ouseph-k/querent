"""
validation/security/exposure.py
───────────────────────────────
ExposureValidator (pipeline step 6b, reports `safety`).

SafetyValidator reasons about statement TYPE — it guarantees the statement is a
SELECT. It never reasons about what that SELECT is allowed to CALL or to
PROJECT. Two consequences, both confirmed against batch run 20260814_102207:

  G4  A well-formed SELECT may invoke any function the database role permits.
      `SELECT pg_read_file('/etc/passwd')`, `SELECT pg_sleep(300)`,
      `dblink(...)`, `query_to_xml('SELECT * FROM data_encryption_key', ...)`
      all parse as exp.Select and pass every one of the twelve existing steps.
      The only control today is the PostgreSQL role grant — a real control, but
      the only one, and it lives outside this codebase.

  G5  Q184 returned `dek.wrapped_key, dek.iv` for four rows. Wrapped DEK
      material is not decryptable without the KEK, so this is not an immediate
      key compromise; it is ciphertext-adjacent material that a natural-language
      reporting surface has no reason to emit, and it is useful to an offline
      attacker. COLUMN_BLOCKLIST in validation/utils/blocklist.py is keyed to
      HALLUCINATED columns, not SENSITIVE ones — a different axis.

Both checks are deny-by-pattern rather than allow-by-pattern for functions the
model legitimately uses in bulk (COUNT, EXTRACT, COALESCE, ...), because an
allowlist over the full PostgreSQL function surface would reject correct SQL on
day one. The catalog-schema check IS an allowlist: nothing in this application
has any business reading pg_catalog or information_schema.

Placed AFTER SafetyValidator and BEFORE SecurityTransformer: there is no point
rewriting a query for tenant isolation if it is going to be rejected anyway.
"""

from __future__ import annotations

import dataclasses
import re

import sqlglot.expressions as exp

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.heuristics import HEURISTICS
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ── G4: server-side functions that read, write, or stall the host ────────────
# Sourced from heuristics.yaml when present so the list is tunable without a
# code change; the literals below are the floor, not the ceiling, and are used
# when the YAML key is absent.
_DEFAULT_DENIED_FUNCTIONS: frozenset[str] = frozenset({
    # filesystem read/write
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_ls_logdir", "pg_ls_waldir", "pg_ls_tmpdir", "pg_ls_archivestatusdir",
    "lo_import", "lo_export", "lo_get", "lo_put",
    # outbound network / federated read
    "dblink", "dblink_exec", "dblink_connect", "dblink_send_query",
    "postgres_fdw_handler",
    # arbitrary SQL evaluated inside a SELECT — the classic sandbox escape
    "query_to_xml", "query_to_xmlschema", "query_to_xml_and_xmlschema",
    "table_to_xml", "database_to_xml",
    # denial of service
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    # session/role manipulation reachable from an expression context
    "set_config", "pg_reload_conf", "pg_rotate_logfile",
    "pg_terminate_backend", "pg_cancel_backend",
    # credential surface
    "pg_read_server_files", "current_setting",
})

_DENIED_FUNCTIONS: frozenset[str] = frozenset(
    str(n).lower()
    for n in HEURISTICS.get("denied_sql_functions", sorted(_DEFAULT_DENIED_FUNCTIONS))
)

# Schemas no application query may read. Unlike the function list this is an
# allowlist inversion with no legitimate exception in this system.
_DENIED_SCHEMAS: frozenset[str] = frozenset({"pg_catalog", "information_schema"})


# ── G5: columns that must never reach a result set ───────────────────────────
# (table, column). Matched on the RESOLVED base table, so an alias does not
# evade it. A column may still be used in a predicate or a join — only the
# projection is denied — because "which scripts share a DEK" is a legitimate
# question and "print me the wrapped key" is not.
_DEFAULT_SENSITIVE_COLUMNS: frozenset[tuple[str, str]] = frozenset({
    ("data_encryption_key", "wrapped_key"),
    ("data_encryption_key", "iv"),
    ("key_encryption_key", "external_id"),
    ("app_user", "external_idp_user_id"),
    ("answer_script", "s3_key"),
    ("answer_script", "s3_version_id"),
    ("script_page", "s3_key"),
    ("answer_key", "s3_key"),
    ("scan_history", "s3_key"),
    ("scan_history", "s3_version_id"),
})

_CONFIGURED_SENSITIVE_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    (str(item["table"]).lower(), str(item["column"]).lower())
    for item in HEURISTICS.get("sensitive_columns", [])
) or _DEFAULT_SENSITIVE_COLUMNS

# Marker recognised inside a COMMENT ON COLUMN body.
_SENSITIVE_MARKER = "@sensitive"


def sensitive_columns(schema_map: dict | None) -> frozenset[tuple[str, str]]:
    """
    The sensitive set = columns tagged in the DDL, UNION the configured list.

    A hand-maintained Python/YAML list is a denylist that silently goes stale
    the moment someone adds a column: the schema changes, the list does not,
    and the gap is invisible until something leaks. Reading the tag from the
    DDL puts the classification next to the column it classifies, inside the
    artefact that already carries a version number and a design-decisions log,
    so a new sensitive column is protected by the same commit that creates it:

        COMMENT ON COLUMN data_encryption_key.wrapped_key IS
            '@sensitive Envelope-encrypted DEK. Never expose in a result set.';

    The configured list is retained as a UNION rather than a fallback so that
    turning the tag on does not silently drop protection from a column nobody
    has tagged yet. Migration path: tag the columns, then shrink the YAML.
    """
    tagged: set[tuple[str, str]] = set()
    for table_name, inv in (schema_map or {}).items():
        comments = getattr(inv, "column_comments", None) or {}
        for column, comment in comments.items():
            if _SENSITIVE_MARKER in (comment or "").lower():
                tagged.add((str(table_name).lower(), str(column).lower()))
    return frozenset(tagged | set(_CONFIGURED_SENSITIVE_COLUMNS))


@dataclasses.dataclass(frozen=True)
class _SensitiveFinding:
    table: str
    column: str
    projection: exp.Expression   # the top-level SELECT-list item to remove
    select_node: exp.Select
    unsafe_to_edit: bool         # True: inside an expression, or a `SELECT *`


# word-boundary match of the column name as a whole word, or its underscore
# parts run together as a phrase ("wrapped_key" -> "wrapped key"). Deliberately
# conservative: short/ambiguous names (e.g. "iv", "id") almost never appear
# this way by coincidence in a natural-language question, so the false-permit
# rate stays low without an explicit stoplist.
def _inflect(term: str) -> set[str]:
    """
    Surface forms of a term that a natural-language question might use.

    English pluralises the HEAD of a noun phrase, so "s3_version_id" is asked
    for as "S3 version IDs". Matching only the singular made the check miss it
    -- and a MISS here is the dangerous direction: it downgrades a hard block
    into a silent redaction, returning a thinner answer to someone who
    explicitly asked for the column. Q165 of batch run 20260817_133854
    ("Show the S3 version IDs...") came back with the requested column quietly
    dropped for exactly this reason.

    Regular plurals only. Irregulars ("indices", "matrices") are not generated:
    an over-broad rule here would start blocking queries that merely mention a
    similar word, and the cost of that is worse than the cost of a rare miss.
    """
    forms = {term}
    if term.endswith("s"):
        forms.add(term[:-1])
        if term.endswith("es"):
            forms.add(term[:-2])
        if term.endswith("ies"):
            forms.add(term[:-3] + "y")
    else:
        forms.add(term + "s")
        if term.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(term + "es")
        if term.endswith("y") and len(term) > 1 and term[-2] not in "aeiou":
            forms.add(term[:-1] + "ies")
    return {f for f in forms if f}


def _explicitly_requested(nl_question: str, column: str) -> bool:
    nl = (nl_question or "").lower()
    if not nl:
        return False
    parts = [p for p in column.lower().split("_") if p]
    candidates: set[str] = set()

    # The bare column name, and its underscore-to-space phrase form. Only the
    # LAST token is inflected -- that is the head of the noun phrase, and it is
    # the only part English pluralises ("s3 version ids", never "s3s version id").
    for base in ({column.lower(), " ".join(parts)} if parts else {column.lower()}):
        base_parts = base.rsplit(" ", 1) if " " in base else base.rsplit("_", 1)
        if len(base_parts) == 2:
            head, tail = base_parts
            sep = " " if " " in base else "_"
            candidates.update(f"{head}{sep}{form}" for form in _inflect(tail))
        else:
            candidates.update(_inflect(base))

    return any(re.search(rf"\b{re.escape(c)}\b", nl) for c in candidates if c)


def _drop_projections(
    stmt: exp.Expression, findings: list["_SensitiveFinding"],
) -> str | None:
    """
    Remove each finding's SELECT-list item from its scope. Returns the
    rewritten SQL, or None if any scope would be left with zero projected
    columns (nothing safe remains to return).

    TWO PASSES, deliberately. A single pass that mutated as it went would
    leave the tree HALF-EDITED when a later scope turned out to be empty and
    forced a `return None` -- and because this operates on the caller's live
    `ctx.ast` node (not a copy), that partially-redacted tree would then be
    what every downstream validator sees, while the caller reports a hard
    block. Verify every scope survives first, mutate only once that is known.

    Operating on the live node rather than a copy is intentional: `by_scope`
    is keyed by `id(select_node)`, which a `.copy()` would invalidate, and
    mutating in place keeps `ctx.ast` consistent with the rewritten SQL for
    the steps that run after this one.
    """
    by_scope: dict[int, list[exp.Expression]] = {}
    for f in findings:
        by_scope.setdefault(id(f.select_node), []).append(f.projection)

    # Pass 1 -- verify, mutate nothing.
    for select_node in stmt.find_all(exp.Select):
        to_remove = by_scope.get(id(select_node))
        if not to_remove:
            continue
        if not [e for e in select_node.expressions if e not in to_remove]:
            return None

    # Pass 2 -- now it is safe to apply.
    for select_node in stmt.find_all(exp.Select):
        to_remove = by_scope.get(id(select_node))
        if not to_remove:
            continue
        select_node.set(
            "expressions",
            [e for e in select_node.expressions if e not in to_remove],
        )

    return stmt.sql(dialect="postgres")


def _alias_map(stmt: exp.Expression) -> dict[str, str]:
    """alias (and bare name) -> base table name, lowercased."""
    out: dict[str, str] = {}
    cte_names = {(c.alias or "").lower() for c in stmt.find_all(exp.CTE)}
    for tbl in stmt.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name or name in cte_names:
            continue
        alias = (tbl.alias or name).lower()
        out[alias] = name
        out.setdefault(name, name)
    return out


class ExposureValidator(BaseValidationStep):
    """Rejects server-side function abuse and sensitive-column projection."""

    name = "ExposureValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql

        sensitive = sensitive_columns(getattr(ctx, "schema_map", None))

        if not ctx.ast:
            # No AST means SyntaxValidator already failed, or the fallback
            # regex path is in play. Do not guess from raw text here — a
            # substring match on "pg_sleep" would fire on a column comment.
            return ValidationResult(passed=True, step="safety", sql=sql)

        for stmt in ctx.ast:
            if stmt is None:
                continue

            denied = self._denied_function(stmt)
            if denied:
                logger.warning(
                    component="sql_validator",
                    event="denied_function_blocked",
                    function=denied,
                    sql_preview=sql[:120],
                )
                return ValidationResult(
                    passed=False, step="safety",
                    message=(
                        f"Function '{denied}' is not permitted. It reads the "
                        f"database host, opens an outbound connection, "
                        f"evaluates arbitrary SQL, or stalls the server — none "
                        f"of which can answer a question about evaluation data. "
                        f"Rewrite the query using only the application tables."
                    ),
                    sql=sql,
                )

            bad_schema = self._denied_schema(stmt)
            if bad_schema:
                logger.warning(
                    component="sql_validator",
                    event="denied_schema_blocked",
                    schema=bad_schema,
                    sql_preview=sql[:120],
                )
                return ValidationResult(
                    passed=False, step="safety",
                    message=(
                        f"Reading the '{bad_schema}' catalog is not permitted. "
                        f"Answer the question from the application tables in "
                        f"the schema you were given."
                    ),
                    sql=sql,
                )

            findings = self._sensitive_projections(stmt, sensitive)
            if findings:
                nl = (ctx.original_query or "")
                redactable, blocking = [], []
                for finding in findings:
                    if finding.unsafe_to_edit:
                        blocking.append(finding)
                    elif _explicitly_requested(nl, finding.column):
                        blocking.append(finding)
                    else:
                        redactable.append(finding)

                if blocking:
                    table, column = blocking[0].table, blocking[0].column
                    asked_for = (
                        _explicitly_requested(nl, column)
                        and not blocking[0].unsafe_to_edit
                    )
                    reason = (
                        "the question explicitly asks for it"
                        if asked_for
                        else "it is used inside an expression or a SELECT *, "
                             "so it cannot be safely dropped"
                    )
                    logger.warning(
                        component="sql_validator",
                        event="sensitive_column_projection_blocked",
                        table=table,
                        column=column,
                        reason=reason,
                        sql_preview=sql[:120],
                    )
                    return ValidationResult(
                        passed=False, step="safety",
                        message=(
                            f"Column '{table}.{column}' must not appear in a "
                            f"result set: it is key material, an object-store "
                            f"locator, or an external identity reference. This "
                            f"query cannot proceed because {reason}. You may "
                            f"filter, join, or COUNT on it, but not SELECT it. "
                            f"Return a non-sensitive identifier such as "
                            f"{table}.id instead."
                        ),
                        sql=sql,
                        # When the QUESTION is what conflicts with the policy,
                        # no rewrite resolves it -- the only queries that pass
                        # are ones that stop answering the question. Refusing
                        # is the correct outcome; quietly answering something
                        # narrower is not.
                        retryable=not asked_for,
                    )

                if redactable:
                    rewritten = _drop_projections(stmt, redactable)
                    if rewritten is None:
                        # Dropping would leave an empty SELECT list (every
                        # projected column was sensitive) -- there is nothing
                        # safe left to return, so this is a genuine block, not
                        # a redaction.
                        table, column = redactable[0].table, redactable[0].column
                        logger.warning(
                            component="sql_validator",
                            event="sensitive_column_projection_blocked",
                            table=table, column=column,
                            reason="redaction would leave an empty SELECT list",
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="safety",
                            message=(
                                f"Every projected column in this query is "
                                f"sensitive ({table}.{column} and possibly "
                                f"others). Rewrite to select a non-sensitive "
                                f"identifier such as {table}.id."
                            ),
                            sql=sql,
                        )
                    logger.info(
                        component="sql_validator",
                        event="sensitive_column_redacted",
                        columns=[f"{f.table}.{f.column}" for f in redactable],
                        note="dropped from SELECT list; not mentioned in the "
                             "question and safe to omit",
                    )
                    sql = rewritten
                    ctx.working_sql = rewritten
                    # No re-parse needed: _drop_projections mutates this very
                    # node in place, so ctx.ast is ALREADY the redacted tree
                    # and every later step sees it. (An earlier revision
                    # rebound the loop variable here, which did nothing at
                    # all -- the list held by ctx.ast was unaffected.)

        return ValidationResult(passed=True, step="safety", sql=sql)

    # ── G4 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _denied_function(stmt: exp.Expression) -> str | None:
        for node in stmt.find_all(exp.Anonymous):
            fn = (node.name or "").lower()
            if fn in _DENIED_FUNCTIONS:
                return fn
        # sqlglot models some of these as typed nodes rather than Anonymous.
        for node in stmt.find_all(exp.Func):
            fn = (getattr(node, "sql_name", lambda: "")() or "").lower()
            if fn in _DENIED_FUNCTIONS:
                return fn
        return None

    @staticmethod
    def _denied_schema(stmt: exp.Expression) -> str | None:
        for tbl in stmt.find_all(exp.Table):
            db = (tbl.text("db") or "").lower()
            if db in _DENIED_SCHEMAS:
                return db
            # Unqualified reference to a catalog relation, e.g. FROM pg_shadow.
            name = (tbl.name or "").lower()
            if name.startswith("pg_") and name not in ("pg_catalog",):
                return "pg_catalog"
        return None

    # ── G5 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _sensitive_projections(
        stmt: exp.Expression, sensitive: frozenset[tuple[str, str]],
    ) -> list["_SensitiveFinding"]:
        """
        Every sensitive column reference in every scope's SELECT list — not
        just the first. Only the SELECT list is inspected; a sensitive column
        in WHERE / ON / GROUP BY is left alone deliberately, since filtering on
        an s3_key answers a real operational question and leaks nothing.

        Each finding also records whether it is SAFE TO REDACT:
          * a bare `t.column` projection (optionally aliased) -- safe, the
            whole SELECT-list item can be dropped.
          * `SELECT *` / `t.*`, or the column used inside an expression such
            as `encode(dek.wrapped_key, 'hex')` -- unsafe. Redacting the first
            would need this validator to expand the star against the full
            schema; redacting the second would need to rewrite the expression
            around a hole. Both stay a hard block.
        """
        aliases = _alias_map(stmt)
        findings: list[_SensitiveFinding] = []

        for select_node in stmt.find_all(exp.Select):
            for projection in select_node.expressions:
                target = projection.this if isinstance(projection, exp.Alias) else projection

                if isinstance(target, exp.Star):
                    for canon in set(aliases.values()):
                        for tbl, col in sensitive:
                            if tbl == canon:
                                findings.append(_SensitiveFinding(
                                    tbl, col, projection, select_node,
                                    unsafe_to_edit=True,
                                ))
                    continue
                if isinstance(target, exp.Column) and isinstance(target.this, exp.Star):
                    canon = aliases.get((target.table or "").lower())
                    for tbl, col in sensitive:
                        if tbl == canon:
                            findings.append(_SensitiveFinding(
                                tbl, col, projection, select_node,
                                unsafe_to_edit=True,
                            ))
                    continue

                is_bare_column = isinstance(target, exp.Column)
                for col_node in target.find_all(exp.Column):
                    canon = aliases.get((col_node.table or "").lower())
                    if canon is None:
                        continue
                    key = (canon, (col_node.name or "").lower())
                    if key in sensitive:
                        findings.append(_SensitiveFinding(
                            key[0], key[1], projection, select_node,
                            unsafe_to_edit=not is_bare_column,
                        ))
        return findings

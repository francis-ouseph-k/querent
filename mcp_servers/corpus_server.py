"""
mcp_servers/corpus_server.py
──────────────────────────────
MCP server managing the Phase 2 training corpus — failure logs and corrections.

FASTMCP VERSION: 3.x (tested on 3.4.2)
  Import : from fastmcp import FastMCP
  Startup: mcp.run(transport="http", ...)   (FastMCP calls uvicorn internally)

  FastMCP 3.x manages uvicorn internally via run_http_async().
  You do not call uvicorn.run() directly — FastMCP does it for you.
  To pass uvicorn tuning options use the uvicorn_config parameter (see entry point).

Exposes five tools:
  log_failure     — write a failed query to the corpus (called by runner.py)
  save_correction — add corrected SQL to a failure entry (called by CLI :correct)
  list_failures   — list uncorrected failures for review and curation
  get_failure     — retrieve one failure entry by ID
  export_corpus   — dump corrected pairs as JSONL for data_pipeline.py

WHY A SEPARATE PROCESS
  The Phase 2 training flywheel depends on team members reviewing and correcting
  failures. With local file access only, this requires SSH into the server machine.
  Exposing the corpus as an MCP server means:
    1. Any authorised client can call log_failure / save_correction / list_failures
       without SSH access.
    2. The storage backend can be switched from local disk to Google Drive by
       changing one .env variable (MCP_CORPUS_BACKEND=drive) — no code changes
       in runner.py or cli/interface.py.
    3. Corpus stats (total entries, corrected count, phase2_ready flag) are
       logged at startup so you can see corpus readiness at a glance.

TWO STORAGE BACKENDS
  local (default):
    Reads and writes JSON files in the failures/ directory on local disk.
    Same format as runner.py._log_failure() and cli/interface.py._handle_correction().
    Fully compatible with the existing failures/ directory — no migration needed.

  drive (future):
    Google Drive folder. Set MCP_CORPUS_BACKEND=drive and
    MCP_CORPUS_DRIVE_FOLDER_ID=<folder_id> in .env.
    Implement the _*_drive() stub functions using the Google Drive MCP tools
    (already connected in your Claude.ai workspace).
    All tool signatures stay the same — only the backend changes.

ATOMIC WRITES
  All writes use tmp-file + os.replace() for atomicity.
  This prevents partial JSON files if the process is killed mid-write.
  os.replace() is atomic on Windows same-drive and on POSIX.
  Same pattern as runner.py._log_failure() and cli/interface.py._handle_correction().

PHASE 2 READINESS
  At startup, this server logs:
    total_entries  — total failure files in failures/
    uncorrected    — entries still needing a corrected_sql
    corrected      — entries with a corrected_sql filled in
    phase2_ready   — True when corrected >= 200 (minimum viable QLoRA corpus)

TRANSPORT
  FastMCP 3.x Streamable HTTP (MCP 2025-03-26 spec).
  Tools served at: POST http://<host>:<port>/mcp

STARTUP
  python mcp_servers/corpus_server.py

CONFIG (.env — all optional, defaults shown)
  MCP_CORPUS_HOST=127.0.0.1
  MCP_CORPUS_PORT=5013
  MCP_CORPUS_BACKEND=local
  MCP_CORPUS_DRIVE_FOLDER_ID=      ← only needed for drive backend
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Add project root to sys.path so config/settings.py resolves correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastmcp import FastMCP

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name        = "corpus-manager",
    instructions = "Phase 2 training corpus: failure log + correction management",
)

# ── Backend selection — read from settings, not os.environ directly ────────────
_BACKEND         = settings.mcp.corpus_backend.lower()          # "local" or "drive"
_DRIVE_FOLDER_ID = settings.mcp.corpus_drive_folder_id          # empty unless drive backend


# ── Helper functions ───────────────────────────────────────────────────────────

def _failure_dir() -> Path:
    """
    Resolve the failures/ directory path from settings and ensure it exists.
    Creates the directory (and any parents) if missing.
    """
    d = Path(settings.failure_log_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_entry_id() -> str:
    """
    Generate a unique failure entry ID.

    Format: YYYYMMDD_HHMMSS_<8-char-uuid>
    Example: 20251201_143022_a3f8c1d2

    The timestamp prefix makes entries sort chronologically in the filesystem.
    The UUID suffix ensures uniqueness even when two failures occur in the
    same second (e.g. during a test run).
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def _atomic_write(path: Path, data: dict) -> None:
    """
    Write a dict as JSON to path atomically using tmp-file + os.replace().

    Why atomic: if the process is killed mid-write, you get either the
    complete old file or the complete new file — never a partial/corrupt file.

    os.replace() is atomic on:
      - Windows: same drive (both tmp and final are in failures/ — same drive)
      - POSIX:   same filesystem (rename syscall is atomic)

    The .tmp extension ensures the partial write is never picked up by
    _load_all_failures() which only globs *.json files.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)   # atomic rename


def _load_all_failures() -> list[dict[str, Any]]:
    """
    Load all *.json failure files from the failures/ directory.

    Adds "_id" field (= file stem, without .json) to each entry so callers
    have the ID needed for save_correction() without needing to parse filenames.

    Skips any file that fails to parse (logs a warning) — one corrupted file
    does not prevent the others from loading.

    Returns entries sorted by filename (= chronological order due to
    timestamp prefix in the entry ID format).
    """
    entries = []
    for path in sorted(_failure_dir().glob("*.json")):
        try:
            entry        = json.loads(path.read_text(encoding="utf-8"))
            entry["_id"] = path.stem   # e.g. "20251201_143022_a3f8c1d2"
            entries.append(entry)
        except Exception as exc:
            logger.warning(
                component = "corpus_mcp",
                event     = "load_error",
                file      = path.name,
                error     = str(exc),
            )
    return entries


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def log_failure(
    nl_query:   str,
    failed_sql: str,
    error:      str,
    retries:    int = 0,
) -> dict[str, str]:
    """
    Write a failed query to the training corpus.

    Called by pipeline/runner.py._failure_result() after all retry attempts
    are exhausted. The entry is written with corrected_sql="" — the correct
    SQL must be added later via save_correction() or the CLI :correct command.

    File format written:
      {
        "timestamp":     "2025-12-01T14:30:22.123456+00:00",
        "nl_query":      "Show failed students in board 5",
        "failed_sql":    "SELECT ... (the wrong SQL)",
        "error":         "Validation failed (schema): Hallucinated column ...",
        "retries":       2,
        "corrected_sql": "",       ← filled in by save_correction() later
        "source":        "pipeline"
      }

    Args:
        nl_query:   Original natural language question from the user.
        failed_sql: The last SQL attempt that failed (after all retries).
        error:      Full error message from the validation step that failed.
                    Includes the step name, e.g. "schema: Hallucinated column(s)..."
        retries:    Number of correction attempts made (0 = first attempt failed,
                    2 = failed on original + 2 retries).

    Returns:
        {"id": "<entry_id>", "path": "<filename.json>"}
    """
    if _BACKEND == "drive":
        raise NotImplementedError(
            "Drive backend not yet implemented. "
            "Set MCP_CORPUS_BACKEND=local or implement the Drive stubs."
        )

    entry_id = _make_entry_id()
    entry    = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "nl_query":      nl_query,
        "failed_sql":    failed_sql,
        "error":         error,
        "retries":       retries,
        "corrected_sql": "",        # to be filled by save_correction()
        "source":        "pipeline",
    }

    path = _failure_dir() / f"{entry_id}.json"
    _atomic_write(path, entry)

    logger.info(
        component = "corpus_mcp",
        event     = "failure_logged",
        id        = entry_id,
        error     = error[:80],
    )
    return {"id": entry_id, "path": path.name}


@mcp.tool()
def save_correction(
    entry_id:      str,
    corrected_sql: str,
    source:        str = "user_correction",
) -> dict[str, Any]:
    """
    Add a corrected SQL to an existing failure entry.

    Called by the CLI :correct command after the user types the correct SQL
    for a failed query. The corrected entry becomes a (nl_query, corrected_sql)
    training pair that data_pipeline.py will pick up for Phase 2 fine-tuning.

    The entry file is updated in-place (atomic write) — the original failed_sql
    and error message are preserved for training context. Only corrected_sql,
    corrected_source, and corrected_at are added.

    Args:
        entry_id:      Failure entry ID — the "_id" field from list_failures(),
                       also the filename stem (e.g. "20251201_143022_a3f8c1d2").
        corrected_sql: The correct SQL that should have been generated.
                       Must be a valid PostgreSQL SELECT statement.
        source:        Who provided the correction.
                       "user_correction" = typed via CLI :correct
                       "automatic" = generated by an automated system
                       "review_team" = provided by the review team via Drive

    Returns:
        {"status": "ok", "id": entry_id}       if found and updated
        {"status": "not_found", "id": entry_id} if no file with that ID exists
    """
    if _BACKEND == "drive":
        raise NotImplementedError("Drive backend not yet implemented.")

    path = _failure_dir() / f"{entry_id}.json"
    if not path.exists():
        return {"status": "not_found", "id": entry_id}

    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["corrected_sql"]    = corrected_sql
    entry["corrected_source"] = source
    entry["corrected_at"]     = datetime.now(timezone.utc).isoformat()
    _atomic_write(path, entry)

    logger.info(
        component = "corpus_mcp",
        event     = "correction_saved",
        id        = entry_id,
        source    = source,
    )
    return {"status": "ok", "id": entry_id}


@mcp.tool()
def list_failures(
    uncorrected_only: bool = True,
    limit:            int  = 50,
) -> list[dict[str, Any]]:
    """
    List failure corpus entries for review and curation.

    The primary tool for the Phase 2 corpus review workflow:
      1. Call list_failures(uncorrected_only=True) to see what needs correction.
      2. Call get_failure(entry_id) to see the full entry.
      3. Call save_correction(entry_id, corrected_sql) to add the fix.
      4. Repeat until phase2_ready (≥200 corrected entries).

    Each entry includes an "_id" field which is the entry_id needed for
    save_correction() and get_failure().

    Args:
        uncorrected_only: True (default) = only entries where corrected_sql
                          is empty — i.e. still needing human review.
                          False = all entries including already-corrected ones
                          (useful for checking what's been done).
        limit:            Maximum entries to return. Oldest first (chronological
                          order matches the filename timestamp prefix).

    Returns:
        List of failure entry dicts. Each contains:
          _id, timestamp, nl_query, failed_sql, error, retries,
          corrected_sql (empty if uncorrected), source, corrected_at (if set).
    """
    if _BACKEND == "drive":
        raise NotImplementedError("Drive backend not yet implemented.")

    entries = _load_all_failures()

    if uncorrected_only:
        entries = [e for e in entries if not e.get("corrected_sql")]

    # Sort oldest first — natural review order (fix the oldest failures first)
    entries.sort(key=lambda e: e.get("timestamp", ""))

    # Remove the internal _file field (not useful to callers)
    return [
        {k: v for k, v in e.items() if k != "_file"}
        for e in entries[:limit]
    ]


@mcp.tool()
def get_failure(entry_id: str) -> dict[str, Any]:
    """
    Retrieve a single failure entry by its ID.

    Use this to inspect a specific failure in full before deciding on a
    correction. The full entry includes the original nl_query, the failed_sql,
    the exact error message, and any existing corrected_sql.

    Args:
        entry_id: The "_id" field from list_failures() output.
                  Also the filename stem: e.g. "20251201_143022_a3f8c1d2"
                  maps to failures/20251201_143022_a3f8c1d2.json

    Returns:
        Full failure entry dict, or {"error": "not_found", "id": entry_id}.
    """
    if _BACKEND == "drive":
        raise NotImplementedError("Drive backend not yet implemented.")

    path = _failure_dir() / f"{entry_id}.json"
    if not path.exists():
        return {"error": "not_found", "id": entry_id}

    return json.loads(path.read_text(encoding="utf-8"))


@mcp.tool()
def export_corpus(
    corrected_only: bool = True,
    output_path:    str  = "data/training/corpus_export.jsonl",
) -> dict[str, Any]:
    """
    Export training pairs from the corpus as a JSONL file for data_pipeline.py.

    Produces one JSON object per line in the format data_pipeline.py expects:
      {"nl_query": "...", "corrected_sql": "...", "failed_sql": "...",
       "error": "...", "timestamp": "...", "source": "..."}

    The failed_sql and error fields are included alongside corrected_sql so
    the training pipeline can use them for chain-of-thought reasoning traces
    (see training corpus strategy in the README).

    Typical workflow:
      1. Accumulate ≥200 corrected failures via CLI :correct
      2. Call export_corpus() → corpus_export.jsonl
      3. Run python fine_tuning/data_pipeline.py --input corpus_export.jsonl
      4. Run python fine_tuning/trainer.py

    Args:
        corrected_only: True (default) = only export entries that have a
                        corrected_sql value. These become training pairs.
                        False = export all entries including uncorrected ones
                        (useful for analysis, not for training).
        output_path:    Destination .jsonl file path, relative to project root.
                        Default: data/training/corpus_export.jsonl
                        Parent directories are created automatically.

    Returns:
        {"exported": N, "output_path": "<absolute-or-relative path>"}
    """
    if _BACKEND == "drive":
        raise NotImplementedError("Drive backend not yet implemented.")

    entries = _load_all_failures()

    if corrected_only:
        entries = [e for e in entries if e.get("corrected_sql")]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            record = {
                "nl_query":      e.get("nl_query",      ""),
                "corrected_sql": e.get("corrected_sql", ""),
                "failed_sql":    e.get("failed_sql",    ""),
                "error":         e.get("error",         ""),
                "timestamp":     e.get("timestamp",     ""),
                "source":        e.get("source",        ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        component   = "corpus_mcp",
        event       = "corpus_exported",
        count       = len(entries),
        output_path = str(out),
    )
    return {"exported": len(entries), "output_path": str(out)}


# ── Startup diagnostics ────────────────────────────────────────────────────────

def _log_startup_info() -> None:
    """
    Log corpus readiness stats at startup.

    Shows total entries, how many still need correction, and whether the
    corpus has reached the 200-pair minimum for Phase 2 fine-tuning.
    Check the logs at startup to know if you are ready to begin Phase 2.
    """
    failure_dir = _failure_dir()
    all_files   = list(failure_dir.glob("*.json"))

    # Count corrected entries — files that have corrected_sql filled in
    corrected_count = 0
    for p in all_files:
        try:
            if json.loads(p.read_text()).get("corrected_sql"):
                corrected_count += 1
        except Exception:
            pass   # skip unreadable files

    uncorrected_count = len(all_files) - corrected_count

    logger.info(
        component     = "corpus_mcp",
        event         = "startup_diagnostics",
        backend       = _BACKEND,
        failure_dir   = str(failure_dir),
        total_entries = len(all_files),
        corrected     = corrected_count,
        uncorrected   = uncorrected_count,
        # phase2_ready = True when enough corrected pairs for QLoRA training
        phase2_ready  = corrected_count >= 200,
        note          = (
            "Ready for Phase 2 QLoRA training."
            if corrected_count >= 200
            else f"Need {200 - corrected_count} more corrected pairs before Phase 2."
        ),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Warn if Drive backend requested but folder ID not set
    if _BACKEND == "drive" and not _DRIVE_FOLDER_ID:
        logger.warning(
            component = "corpus_mcp",
            event     = "drive_folder_id_missing",
            note      = (
                "MCP_CORPUS_BACKEND=drive but MCP_CORPUS_DRIVE_FOLDER_ID is empty. "
                "Set it in .env to enable the Drive backend. "
                "Falling back to local behaviour (tools will raise NotImplementedError)."
            ),
        )

    # Log corpus readiness before starting — visible in startup logs
    _log_startup_info()

    host = settings.mcp.corpus_host
    port = settings.mcp.corpus_port

    logger.info(
        component = "corpus_mcp",
        event     = "server_starting",
        host      = host,
        port      = port,
        backend   = _BACKEND,
    )

    # FastMCP 3.x calls uvicorn internally via run_http_async().
    # You do not call uvicorn.run() directly — FastMCP manages it.
    #
    # transport="http" = Streamable HTTP (MCP 2025-03-26 spec, recommended)
    # transport="sse"  = legacy SSE for older MCP clients
    mcp.run(
        transport      = "http",
        host           = host,
        port           = port,
        json_response  = True,   # return plain JSON instead of SSE stream
        stateless_http = True,   # no session handshake required per call
    )

    # If you ever need to tune uvicorn (e.g. log level, workers):
    # mcp.run(
    #     transport      = "http",
    #     host           = host,
    #     port           = port,
    #     uvicorn_config = {"log_level": "warning", "workers": 1},
    # )
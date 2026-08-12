"""
utils/logging_config.py
───────────────────────
Structured JSON logging using structlog.
Every pipeline stage logs a single JSON entry per request — all stages
are linked by request_id for end-to-end traceability.

CONSOLE VS FILE — DELIBERATELY DIFFERENT VERBOSITY
───────────────────────────────────────────────────
The file log (logs/nl_sql.jsonl) is always full detail — every INFO/DEBUG
event, for post-hoc debugging and the Observability workflows in the README.

The console is tuned for a human watching a live run. Default level is
WARNING, not INFO: at INFO, every retrieval call, every provider-selection
event, and — because `logging.basicConfig` attaches to the ROOT logger —
every third-party library's own internal logging (httpx's "HTTP Request:
POST ..." per call, sentence-transformers' model-load banners, OpenSearch's
request logs, sqlglot's DDL parse-fallback diagnostics) all land on stdout
too, interleaved with the app's own progress output. That's the wall of
`2026-...T... [info] ...` noise this module used to produce on every run —
not a bug in any one component, just every logger in the process sharing one
unfiltered console handler.

Call `set_console_level("INFO")` (wired to `--debug` / `:debug`) when you
actually want that trace back — nothing is lost, it's just off by default.
`sqlglot`'s own logger is pinned to ERROR unconditionally: its DDL
parse-fallback warnings are expected/benign noise (see ddl_parser.py) and
add nothing even in debug mode.

Usage:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("retrieval_complete", dense_hits=12, bm25_hits=8, tokens=847)
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import structlog

# Third-party loggers that are chatty at INFO but rarely diagnostic for this
# app's own console output. Silenced to WARNING so their routine chatter
# never reaches the console; real warnings/errors from these libraries
# still surface. Silencing happens at the *logger* level, so this also keeps
# them out of the file log — acceptable, since none of this is app logic.
_NOISY_LIBRARY_LOGGERS: tuple[str, ...] = (
    "httpx", "httpcore", "urllib3",
    "opensearch", "qdrant_client",
    "sentence_transformers", "transformers",
)

# Global handle to the console handler so set_console_level() can retarget it
# after configure_logging() has already run (main.py/batch_run.py configure
# logging at import time, before argparse has read --debug).
_console_handler: logging.Handler | None = None


def configure_logging(log_dir: str = "logs", level: str = "WARNING") -> None:
    """
    Configure structlog for structured JSON output.
    - Console: quiet by default (WARNING) — see module docstring
    - File: always full detail, newline-delimited JSON
    """
    global _console_handler

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "nl_sql.jsonl"

    # ── stdlib handler — JSON file ─────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # ── stdlib handler — console ──────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper(), logging.WARNING))
    _console_handler = console_handler

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        format="%(message)s",
    )

    # Silence known-noisy third-party loggers (see module docstring).
    for name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # sqlglot's DDL-parse-fallback diagnostics are expected/benign — always
    # off, not just at the default console level, since they add nothing
    # even when actively debugging (see ingestion/ddl_parser.py).
    logging.getLogger("sqlglot").setLevel(logging.ERROR)

    # Deprecation/user warnings from HF `transformers` (e.g. TRANSFORMERS_CACHE,
    # "Special tokens have been added...") are informational library noise,
    # not something this app's user can or needs to act on.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers.*")

    # ── structlog processors ──────────────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # File formatter — JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)

    # Console formatter — colour dev output
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)


def set_console_level(level: str) -> None:
    """
    Retarget the console handler's verbosity after configure_logging() has
    already run. Needed because main.py / batch_run.py call
    configure_logging() at import time, before argparse has read --debug —
    this lets --debug / :debug bump the console back up to INFO on demand
    without having to restructure the module-load-time logging setup.

    No-op (logs a warning to the file handler only) if called before
    configure_logging(); every entry point calls configure_logging() first,
    so this should not happen in practice.
    """
    if _console_handler is None:
        get_logger(__name__).warning(
            "set_console_level_before_configure",
            note="configure_logging() has not run yet; ignoring.",
        )
        return
    _console_handler.setLevel(getattr(logging, level.upper(), logging.WARNING))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)

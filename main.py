"""
main.py
────────
Entry point for the Querent CLI application.

Purpose:
- Initialize configuration and logging
- Load schema graph + pipeline runner
- Provide both interactive and single-query execution modes

Modes:
    1. Interactive CLI chat
    2. Single-query execution (--query)
    3. Dry-run / execution mode control

Example usage:
    python main.py
    python main.py --dry-run
    python main.py --exec
    python main.py --query "list pending evaluations"
"""

from __future__ import annotations

import argparse
import sys

from config.settings import settings
from utils.logging_config import configure_logging, get_logger
from pipeline.bootstrap import create_runner

# ─────────────────────────────────────────────────────────────
# Logging setup (must be initialized before any pipeline work)
# ─────────────────────────────────────────────────────────────
configure_logging(settings.log_dir)
logger = get_logger(__name__)


def main() -> None:
    """
    CLI entry function.

    Responsibilities:
    - Parse CLI arguments
    - Configure runtime settings
    - Initialize pipeline runner
    - Route execution to interactive or batch mode
    """

    # ─────────────────────────────────────────────────────────
    # CLI argument definitions
    # ─────────────────────────────────────────────────────────
    arg_parser = argparse.ArgumentParser(
        description="Querent — Natural Language to SQL Engine"
    )

    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run mode (no DB execution)"
    )

    arg_parser.add_argument(
        "--exec",
        action="store_true",
        help="Force execution mode (runs SQL on DB)"
    )

    arg_parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Run a single query and exit (non-interactive mode)"
    )

    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (shows retrieval + internal traces)"
    )

    arg_parser.add_argument(
        "--strict-version-check",
        action="store_true",
        help="Fail immediately on schema version mismatch"
    )

    args = arg_parser.parse_args()

    # ─────────────────────────────────────────────────────────
    # Runtime configuration overrides
    # ─────────────────────────────────────────────────────────
    settings.strict_version_check = args.strict_version_check
    settings.debug_mode = args.debug

    # Create pipeline runner (loads schema + retrieval + validation stack)
    runner = create_runner(strict_version_check=settings.strict_version_check)

    # Apply execution mode overrides
    if args.dry_run:
        settings.dry_run_default = True
    elif args.exec:
        settings.dry_run_default = False

    # ─────────────────────────────────────────────────────────
    # Single-query (non-interactive) mode
    # ─────────────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery:\n  {args.query}\n")

        result = runner.run(
            nl_query=args.query,
            user_context={"role": "evaluator"},
        )

        print("-" * 60)

        if result.success:
            print("SUCCESS")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Retries: {result.retries}")

            if result.sql:
                print("\nSQL:\n")
                print(result.sql)

            sys.exit(0)

        # Failure path
        print("FAILED")
        print(f"Error: {result.error}")

        if result.sql:
            print("\nLast SQL attempt:\n")
            print(result.sql)

        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # Interactive CLI mode
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Querent — Natural Language to SQL Engine".center(80))
    print("=" * 80)
    print("Enter queries below. Type 'exit' or 'quit' to stop.")
    print("=" * 80 + "\n")

    user_ctx = {"role": "evaluator"}

    while True:
        try:
            q = input("\nQuery: ").strip()

            if not q:
                continue

            if q.lower() in ("exit", "quit"):
                break

            result = runner.run(
                nl_query=q,
                user_context=user_ctx,
            )

            print("-" * 60)

            if result.success:
                print("SUCCESS")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Retries: {result.retries}")

                if result.sql:
                    print(result.sql)
            else:
                print("FAILED")
                print(f"Error: {result.error}")

                if result.sql:
                    print(result.sql)

        except KeyboardInterrupt:
            print("\nSession terminated.")
            break

        except Exception as e:
            logger.exception("cli_runtime_error")
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
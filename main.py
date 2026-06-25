"""
main.py
────────
Application entry point for the NL→SQL CLI.

Loads the FK graph and initialises the pipeline runner,
then hands off to the terminal chat interface.

Usage:
    python main.py
    python main.py --dry-run      # start in dry-run mode
    python main.py --query "..."  # single query, non-interactive
"""

from __future__ import annotations

import argparse
import sys

from config.settings import settings
from utils.logging_config import configure_logging, get_logger
from pipeline.bootstrap import create_runner

configure_logging(settings.log_dir)
logger = get_logger(__name__)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Digital Evaluation System — NL→SQL")
    arg_parser.add_argument("--dry-run", action="store_true",
                            help="Override default — start in dry-run mode")
    arg_parser.add_argument("--exec",    action="store_true",
                            help="Override default — start in execute mode")
    arg_parser.add_argument("--query",   type=str, default="",
                            help="Single non-interactive query")
    arg_parser.add_argument("--debug",   action="store_true",
                            help="Start with debug mode ON — show full retrieval chunks in console")
    arg_parser.add_argument("--strict-version-check", action="store_true",
                            help="Make schema version drift mismatch fatal")
    args = arg_parser.parse_args()

    if args.strict_version_check:
        settings.strict_version_check = True

    # Build pipeline runner (includes schema load & checks)
    runner = create_runner(strict_version_check=settings.strict_version_check)

    # Determine starting mode
    if args.dry_run:
        settings.dry_run_default = True
    elif args.exec:
        settings.dry_run_default = False

    if args.debug:
        settings.debug_mode = True

    # ─────────────────────────────────────────────────────────────────────────
    # Non-interactive mode
    # ─────────────────────────────────────────────────────────────────────────
    if args.query:
        print(f"\nEvaluating single query:\n  {args.query}\n")
        
        result = runner.run(
            nl_query=args.query,
            user_context={"role": "evaluator"},
        )
        print("—" * 60)
        
        if result.success:
            try:
                print("\n✅ GENERATION SUCCESSFUL")
            except UnicodeEncodeError:
                print("\n[OK] GENERATION SUCCESSFUL")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Retries used: {result.retries}")
            if result.sql:
                print("\nFINAL SQL:\n")
                print(result.sql)
            sys.exit(0)
        else:
            try:
                print("\n❌ PIPELINE FAILED")
            except UnicodeEncodeError:
                print("\n[X] PIPELINE FAILED")
            print(f"Reason: {result.error}")
            if result.sql:
                print("\nLAST ATTEMPTED SQL:\n")
                print(result.sql)
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Interactive chat mode
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("[*] Digital Evaluation AI -- NL->SQL Generator".center(80))
    print("="*80)
    print("Type a natural language question about the exam/evaluation schema.")
    print("Type 'exit' or 'quit' to terminate.")
    print("="*80 + "\n")

    user_ctx = {"role": "evaluator"}  # M1 fix: explicit role placeholder

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

            print("—" * 60)
            if result.success:
                try:
                    print("✅ GENERATION SUCCESSFUL")
                except UnicodeEncodeError:
                    print("[OK] GENERATION SUCCESSFUL")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Retries used: {result.retries}")
                if result.sql:
                    print("\nFINAL SQL:\n")
                    print(result.sql)
            else:
                try:
                    print("❌ PIPELINE FAILED")
                except UnicodeEncodeError:
                    print("[X] PIPELINE FAILED")
                print(f"Reason: {result.error}")
                if result.sql:
                    print("\nLAST ATTEMPTED SQL:\n")
                    print(result.sql)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[!] Unexpected Error: {e}")
            logger.exception("interactive_shell_error")


if __name__ == "__main__":
    main()
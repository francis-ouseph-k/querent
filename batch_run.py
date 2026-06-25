"""
batch_run.py
────────────
Batch runner for NL-to-SQL pipeline evaluation.

Reads questions from data/inputs/user-queries-batch.jsonl,
runs each through PipelineRunner sequentially,
writes results to data/output/batch-run-output-<timestamp>.jsonl.

Usage (from project root nl_to_sql/):
    python batch_run.py
    python batch_run.py --dry-run      # skip DB execution
    python batch_run.py --start 50     # resume from QNum 50

Output format (one JSON line per question):
    {
        "QNum": 1,
        "Question": "...",
        "type": "High",
        "Result": "Success" | "Error",
        "Error Message": "",
        "Generated query": "SELECT ...",
        "Elapsed ms": 421
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ── Bootstrap — identical to main.py ──────────────────────────────────────────

from config.settings import settings
from utils.logging_config import configure_logging, get_logger

configure_logging(settings.log_dir)
logger = get_logger(__name__)

INPUT_PATH = Path("data/inputs/user-queries-batch.jsonl")
OUTPUT_DIR = Path("data/output")

COMPONENT = "batch_run"


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable Xh Ym Zs string."""
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


from pipeline.bootstrap import create_runner


# ── Main batch logic ───────────────────────────────────────────────────────────

def run_batch(dry_run: bool = False, start_from: int = 1, strict_version_check: bool = False) -> None:

    if strict_version_check:
        settings.strict_version_check = True

    # Enforce strict version check if enabled
    from pipeline.bootstrap import check_schema_version
    check_schema_version(strict=settings.strict_version_check)

    # Validate input path
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    # Load questions — skip malformed/empty lines, never crash
    questions = []
    bad_lines = 0
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines += 1
                print(f"WARNING: skipping malformed line {line_num}: {exc}")
                logger.warning(
                    component=COMPONENT,
                    event="malformed_input_line",
                    line_num=line_num,
                    error=str(exc),
                )

    if bad_lines:
        print(f"Skipped {bad_lines} malformed line(s).")

    if not questions:
        print("ERROR: No valid questions found in input file.")
        sys.exit(1)

    # Apply --start filter
    if start_from > 1:
        questions = [q for q in questions if q.get("QNum", 0) >= start_from]
        print(f"Resuming from QNum {start_from} -- {len(questions)} questions to run.")

    total = len(questions)
    print(f"Loaded {total} questions from {INPUT_PATH}")

    # Output file — microseconds in timestamp prevents same-second collision
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"batch-run-output-{timestamp}.jsonl"
    print(f"Output -> {output_path}")

    logger.info(
        component=COMPONENT,
        event="batch_start",
        total=total,
        dry_run=dry_run,
        start_from=start_from,
        output=str(output_path),
    )

    # Initialise pipeline — guard so root cause is logged on startup failure
    try:
        print("Loading schema...")
        runner = create_runner()

        logger.info(
            component=COMPONENT,
            event="pipeline_ready",
            tables=len(runner.tables),
            graph_nodes=runner.fk_graph.number_of_nodes(),
        )
    except Exception as exc:
        try:
            logger.error(
                component=COMPONENT,
                event="init_failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        print(f"ERROR: Pipeline initialisation failed: {exc}")
        sys.exit(1)

    # Counters
    success_count  = 0
    error_count    = 0
    written_count  = 0
    write_failures = 0

    # Per-tier counters for summary
    tier_total   = {"High": 0, "Medium": 0, "Low": 0}
    tier_success = {"High": 0, "Medium": 0, "Low": 0}

    batch_start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, item in enumerate(questions, 1):
            qnum     = item.get("QNum", idx)
            question = item.get("Question", "")
            qtype    = item.get("type", "")

            # Track tier totals
            if qtype in tier_total:
                tier_total[qtype] += 1

            done      = idx - 1
            remaining = total - done
            print(f"\n[{idx}/{total}] Q{qnum} ({qtype}) | {done} done, {remaining} remaining")
            print(f"  {question[:100]}...")

            result_row = {
                "QNum":            qnum,
                "Question":        question,
                "type":            qtype,
                "Result":          "Error",
                "Error Message":   "",
                "Generated query": "",
                "Elapsed ms":      0,
                "Query Confidence": 0.0,
                "Retrieval ms":    0,
                "Generation ms":   0,
                "Prompt tokens":   0,
                "Completion tokens": 0,
            }

            # Validate question text — skip empty questions immediately
            if not question.strip():
                result_row["Error Message"] = "Empty question -- skipped"
                error_count += 1
                print(f"  SKIP: empty question")
                logger.warning(
                    component=COMPONENT,
                    event="empty_question_skipped",
                    qnum=qnum,
                )
            else:
                t0 = time.perf_counter()
                try:
                    result     = runner.run(question, dry_run=dry_run)
                    elapsed_ms = round((time.perf_counter() - t0) * 1000)
                    result_row["Elapsed ms"] = elapsed_ms
                    result_row["Query Confidence"] = getattr(result, "confidence", 0.0)
                    if hasattr(result, "latency_ms"):
                        result_row["Retrieval ms"] = result.latency_ms.get("retrieval_ms", 0)
                        result_row["Generation ms"] = result.latency_ms.get("generation_ms", 0)
                    if hasattr(result, "retrieval_meta"):
                        result_row["Prompt tokens"] = result.retrieval_meta.get("llm_prompt_tokens", 0)
                        result_row["Completion tokens"] = result.retrieval_meta.get("llm_completion_tokens", 0)

                    if result.success:
                        result_row["Result"]          = "Success"
                        result_row["Generated query"] = result.sql or ""
                        success_count += 1
                        if qtype in tier_success:
                            tier_success[qtype] += 1
                        print(f"  OK ({elapsed_ms}ms)")
                        logger.info(
                            component=COMPONENT,
                            event="question_success",
                            qnum=qnum,
                            qtype=qtype,
                            elapsed_ms=elapsed_ms,
                            intent=result.intent,
                            retries=result.retries,
                        )
                    else:
                        result_row["Result"]          = "Error"
                        result_row["Error Message"]   = result.error or "Unknown pipeline error"
                        result_row["Generated query"] = result.sql or ""
                        error_count += 1
                        print(f"  FAIL ({elapsed_ms}ms): {(result.error or '')[:120]}")
                        logger.warning(
                            component=COMPONENT,
                            event="question_failed",
                            qnum=qnum,
                            qtype=qtype,
                            elapsed_ms=elapsed_ms,
                            error=result.error,
                            retries=result.retries,
                        )

                except Exception as exc:
                    elapsed_ms = round((time.perf_counter() - t0) * 1000)
                    msg = f"{type(exc).__name__}: {exc}"
                    result_row["Error Message"] = msg
                    result_row["Elapsed ms"]    = elapsed_ms
                    error_count += 1
                    print(f"  EXCEPTION ({elapsed_ms}ms): {msg[:120]}")
                    try:
                        logger.error(
                            component=COMPONENT,
                            event="question_exception",
                            qnum=qnum,
                            qtype=qtype,
                            elapsed_ms=elapsed_ms,
                            error=msg,
                            traceback=traceback.format_exc(),
                        )
                    except Exception as log_exc:
                        print(f"  WARN logger failed: {log_exc}")

            # ── Progress indicator ─────────────────────────────────────────
            wall_elapsed   = time.perf_counter() - batch_start
            avg_per_q      = wall_elapsed / idx
            est_remaining  = avg_per_q * (total - idx)
            success_rate   = f"{success_count / idx * 100:.1f}%" if idx else "n/a"
            print(
                f"  Progress: {idx}/{total} | "
                f"Success: {success_count} | "
                f"Error: {error_count} | "
                f"Rate: {success_rate} | "
                f"Elapsed: {_fmt_duration(wall_elapsed)} | "
                f"ETA: {_fmt_duration(est_remaining)}"
            )

            # Write — protected so a disk error never kills the batch
            try:
                out_f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                out_f.flush()
                written_count += 1
            except Exception as write_exc:
                write_failures += 1
                print(f"  WARN write failed for Q{qnum}: {write_exc}")
                logger.error(
                    component=COMPONENT,
                    event="write_failed",
                    qnum=qnum,
                    error=str(write_exc),
                )

    # ── Final summary ──────────────────────────────────────────────────────
    wall_total = time.perf_counter() - batch_start
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE")
    print(f"  Total:      {total}")
    print(f"  Success:    {success_count}  ({success_count/total*100:.1f}%)" if total else "")
    print(f"  Error:      {error_count}")
    print(f"  Written:    {written_count}  (lost: {write_failures})")
    print(f"  Wall time:  {_fmt_duration(wall_total)}")
    print(f"  Avg/query:  {_fmt_duration(wall_total/total)}" if total else "")
    print("")
    print("By tier:")
    for tier in ("High", "Medium", "Low"):
        t   = tier_total[tier]
        s   = tier_success[tier]
        pct = f"{s/t*100:.1f}%" if t else "n/a"
        print(f"  {tier:<8} {s}/{t}  ({pct})")
    print(f"\nOutput: {output_path}")

    logger.info(
        component=COMPONENT,
        event="batch_complete",
        total=total,
        success=success_count,
        error=error_count,
        written=written_count,
        write_failures=write_failures,
        wall_seconds=round(wall_total),
        output=str(output_path),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL-to-SQL batch evaluator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip DB execution -- validate and generate SQL only",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        metavar="QNUM",
        help="Resume from this QNum (skip earlier questions)",
    )
    parser.add_argument(
        "--strict-version-check",
        action="store_true",
        help="Make schema version drift mismatch fatal",
    )
    args = parser.parse_args()

    run_batch(dry_run=args.dry_run, start_from=args.start, strict_version_check=args.strict_version_check)
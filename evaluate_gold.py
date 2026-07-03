"""
evaluate_gold.py — Execution-based accuracy for the NL→SQL pipeline.

WHY THIS EXISTS (Blocker #2)
---------------------------
The batch runner reports "Success" when a query passes the 12-step validation
pipeline and executes without error. That is NOT the same as "correct": a query
can be valid, run cleanly, and still return the wrong rows (wrong filter, wrong
grain, wrong join). This script turns "Success" into "correct" by executing the
generated SQL and the known-correct (gold) SQL against the real database and
comparing their result sets — the standard "execution accuracy" measure used in
text-to-SQL benchmarks.

It also surfaces the two numbers you actually care about before Phase 2:
  * true accuracy (generated result-set == gold result-set), by tier
  * "suspicious successes": validation passed but the answer is wrong

USAGE
-----
    python evaluate_gold.py \
        --batch  data/output/batch-run-output-YYYYMMDD_HHMMSS.jsonl \
        --gold   data/inputs/gold-sql.jsonl \
        [--dsn "postgresql://user:pass@host:5432/dbname"] \
        [--sorted-cells] [--round 4] [--timeout-ms 15000] [--limit 5000] \
        [--out data/output/gold-eval.jsonl]

GOLD FILE FORMAT (JSONL, one object per line). Keys are matched flexibly:
    {"QNum": 1, "sql": "SELECT ..."}                 # or "gold_sql" / "query"
    {"QNum": 2, "sql": "SELECT ...", "ordered": true}# force row-order comparison
If a gold row sets "ordered": true (or the gold SQL contains ORDER BY), row order
is enforced for that query; otherwise rows are compared as an order-insensitive
multiset.

COMPARISON SEMANTICS
--------------------
Both queries are executed read-only, in an aborting transaction, with a
statement_timeout and a safety LIMIT. Result sets are compared by VALUE, not by
column name (aliases differ between generated and gold SQL):
  * each cell is normalised (None→"∅", floats rounded to --round, everything else
    str()-ed and stripped);
  * each row becomes a tuple of normalised cells in SELECT order;
  * with --sorted-cells, cells within a row are also sorted, tolerating a
    different column order between generated and gold (looser; may accept a
    coincidental match — off by default);
  * rows are compared as a multiset (collections.Counter) unless ordered.

Empty-result handling: if the generated AND gold queries both return zero rows,
that is a real but DEGENERATE agreement (any two queries agree most easily on "no
rows"). It is reported as its own status `correct_empty` and a run-time WARNING, and
still counts toward accuracy — but flagged so you can confirm the correct answer is
genuinely empty rather than empty-for-the-wrong-reason (e.g. a bad enum filter).

This script never writes to the database. It is safe to run against a replica.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# psycopg2 is imported lazily inside the DB path so the pure comparison logic in
# this module can be imported and unit-tested without a database driver present.


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line {ln} in {path.name}: {e}", file=sys.stderr)
    return rows


def index_batch(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        q = _first(r, "QNum", "qnum", "id")
        if q is None:
            continue
        out[int(q)] = {
            "sql":    _first(r, "Generated query", "generated_query", "sql", default=""),
            "type":   _first(r, "type", "tier", default="?"),
            "result": _first(r, "Result", "result", default="?"),
            "question": _first(r, "Question", "question", default=""),
        }
    return out


def index_gold(rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in rows:
        q = _first(r, "QNum", "qnum", "id")
        if q is None:
            continue
        sql = _first(r, "sql", "gold_sql", "query", "Gold", default="")
        out[int(q)] = {"sql": sql, "ordered": bool(r.get("ordered", False))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Execution + comparison
# ─────────────────────────────────────────────────────────────────────────────
def normalise_cell(v, ndigits: int) -> str:
    if v is None:
        return "∅"
    if isinstance(v, float):
        return format(round(v, ndigits), f".{ndigits}f")
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


def run_query(conn, sql: str, timeout_ms: int, limit: int, ndigits: int, sorted_cells: bool):
    """Execute read-only; return (rows_as_normalised_tuples, error_or_None)."""
    with conn.cursor() as cur:
        try:
            cur.execute("BEGIN READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
            wrapped = f"SELECT * FROM (\n{sql.rstrip().rstrip(';')}\n) _gold_eval LIMIT {int(limit)}"
            cur.execute(wrapped)
            raw = cur.fetchall()
        except Exception as e:  # noqa: BLE001 — report any DB error verbatim
            conn.rollback()
            return None, str(e).strip().splitlines()[0][:200]
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
    rows = []
    for row in raw:
        cells = [normalise_cell(c, ndigits) for c in row]
        if sorted_cells:
            cells = sorted(cells)
        rows.append(tuple(cells))
    return rows, None


def compare(gen_rows, gold_rows, ordered: bool) -> bool:
    if ordered:
        return gen_rows == gold_rows
    return Counter(gen_rows) == Counter(gold_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def resolve_dsn(cli_dsn: str | None) -> str:
    if cli_dsn:
        return cli_dsn
    # Fall back to the project's own settings if available.
    try:
        from config.settings import settings
        pg = settings.postgres
        return (f"host={pg.host} port={pg.port} dbname={pg.database} "
                f"user={pg.user} password={pg.password}")
    except Exception as e:  # noqa: BLE001
        print(f"No --dsn given and could not read config.settings.postgres ({e}).", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-based accuracy vs gold SQL.")
    ap.add_argument("--batch", required=True, type=Path, help="batch_run output JSONL")
    ap.add_argument("--gold",  required=True, type=Path, help="gold SQL JSONL")
    ap.add_argument("--dsn", default=None, help="Postgres DSN/URI (else uses config.settings.postgres)")
    ap.add_argument("--sorted-cells", action="store_true",
                    help="sort cells within a row (tolerate column-order differences; looser)")
    ap.add_argument("--round", type=int, default=4, help="float rounding for comparison")
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--limit", type=int, default=5000, help="safety row cap per query")
    ap.add_argument("--out", type=Path, default=None, help="per-query results JSONL")
    args = ap.parse_args()

    batch = index_batch(load_jsonl(args.batch))
    gold  = index_gold(load_jsonl(args.gold))
    qnums = sorted(set(batch) & set(gold))
    if not qnums:
        print("No overlapping QNums between batch and gold files.", file=sys.stderr)
        sys.exit(2)
    missing_gold = sorted(set(batch) - set(gold))
    if missing_gold:
        print(f"  ! {len(missing_gold)} batch queries have no gold entry (skipped): {missing_gold[:10]}...")

    try:
        import psycopg2
    except ImportError:
        print("psycopg2 is required to execute queries: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(2)
    conn = psycopg2.connect(resolve_dsn(args.dsn))
    conn.autocommit = True  # we manage explicit BEGIN READ ONLY / ROLLBACK per query

    per_query = []
    tally = Counter()
    tier_tally = {}  # tier -> Counter of statuses

    for q in qnums:
        b, g = batch[q], gold[q]
        tier = b["type"]
        tier_tally.setdefault(tier, Counter())
        gen_sql, gold_sql = (b["sql"] or "").strip(), (g["sql"] or "").strip()
        ordered = g["ordered"] or ("order by" in gold_sql.lower())

        if not gen_sql:
            status = "no_generated_sql"
        else:
            gold_rows, gold_err = run_query(conn, gold_sql, args.timeout_ms, args.limit, args.round, args.sorted_cells)
            if gold_err:
                status = "gold_error"          # gold SQL itself failed — fix the gold file
            else:
                gen_rows, gen_err = run_query(conn, gen_sql, args.timeout_ms, args.limit, args.round, args.sorted_cells)
                if gen_err:
                    status = "generated_error"  # generated SQL failed to execute
                elif compare(gen_rows, gold_rows, ordered):
                    # Both-empty is a real agreement but a DEGENERATE one: two
                    # different queries most easily agree on "no rows". A wrong
                    # query that returns empty for the wrong reason (e.g. a bad
                    # enum filter) will match an empty gold and look correct. So
                    # we split it into its own status, warn at run time, and let
                    # the operator verify the correct answer is genuinely empty.
                    if not gen_rows and not gold_rows:
                        status = "correct_empty"
                        print(f"  WARNING Q{q}: generated AND gold both returned 0 rows "
                              f"— weak match; verify the correct answer is genuinely empty.")
                    else:
                        status = "correct"
                else:
                    status = "wrong_rows"       # ran fine, wrong answer

        # Flag the case the batch called Success but the answer is wrong.
        suspicious = (b["result"] == "Success" and status in ("wrong_rows", "generated_error"))
        tally[status] += 1
        tier_tally[tier][status] += 1
        if suspicious:
            tally["_suspicious_success"] += 1
        per_query.append({
            "QNum": q, "tier": tier, "batch_result": b["result"],
            "exec_status": status, "suspicious_success": suspicious,
            "ordered": ordered,
        })

    conn.close()

    # ── Report ──────────────────────────────────────────────────────────────
    total = len(qnums)
    # correct_empty counts as correct (it IS an agreement) but is flagged below
    # as a weak match the operator should eyeball.
    correct = tally["correct"] + tally["correct_empty"]
    print("\n" + "=" * 60)
    print("EXECUTION-BASED ACCURACY (vs gold SQL)")
    print("=" * 60)
    print(f"Compared:            {total}")
    print(f"Correct:             {correct} ({100*correct/total:.1f}%)   <-- TRUE accuracy")
    if tally["correct_empty"]:
        print(f"  of which empty-on-both: {tally['correct_empty']}  "
              f"(WEAK match — both returned 0 rows; verify gold is genuinely empty)")
    print(f"Wrong rows:          {tally['wrong_rows']}")
    print(f"Generated errored:   {tally['generated_error']}")
    print(f"Gold errored (fix):  {tally['gold_error']}")
    print(f"No generated SQL:    {tally['no_generated_sql']}")
    print(f"Suspicious Success:  {tally['_suspicious_success']}   <-- batch=Success but answer wrong/errored")
    print("\nBy tier (correct / compared):")
    for tier in sorted(tier_tally):
        c = tier_tally[tier]; n = sum(c.values())
        tc = c["correct"] + c["correct_empty"]
        print(f"  {tier:8} {tc}/{n} ({100*tc/n:.1f}%)")

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            for r in per_query:
                f.write(json.dumps(r) + "\n")
        print(f"\nPer-query detail -> {args.out}")


if __name__ == "__main__":
    main()
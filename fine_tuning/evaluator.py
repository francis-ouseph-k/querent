"""
fine_tuning/evaluator.py
═══════════════════════════════════════════════════════════════════════════════
Evaluation and Regression Guard
──────────────────────────────────────────
Runs the held-out eval set through the full Phase 1 pipeline using the
fine-tuned model (base + LoRA adapter via HuggingFace transformers) and
measures five metrics against the Phase 1 baseline.

This is the gate before export.py.  If any metric regresses below the
baseline, do NOT export — adjust training and retrain.

Metrics
───────
  syntax_pass_rate       SQL parses without error (sqlglot)
  hallucination_rate     SQL references only real schema tables
  execution_valid_rate   EXPLAIN runs without error (requires DB connection)
  semantic_correct_rate  Result rows match expected output (requires test data)
  p50_latency_ms         Median generation time

How it loads the model
──────────────────────
The evaluator loads the HuggingFace base model + LoRA adapter directly via
transformers and peft.  It does NOT use llama-server.  This means:
  - llama-server does not need to be running
  - The GPU is used for HuggingFace inference, not llama.cpp
  - Generation is slightly slower than llama-server (no GGUF optimisations)
  - This is intentional — eval happens before export, so the GGUF does not
    exist yet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLACKWELL GPU CONFLICT — READ BEFORE CHANGING MODEL LOADING CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Background
──────────
The original implementation loaded the model using BitsAndBytes NF4 4-bit
quantisation with device_map="auto".  This is the standard configuration for
loading large models on consumer GPUs and works on Ampere (sm_80) and older.

The Problem
───────────
On NVIDIA RTX 50-series "Blackwell" GPUs (sm_120 — e.g. RTX 5060 Ti,
RTX 5070, RTX 5080, RTX 5090), NF4 dequantisation kernels are not validated
for sm_120 and produce corrupt outputs.  When the evaluator loads the base
model using NF4, the adapter is applied on top of corrupted weights, making
every evaluation result unreliable regardless of actual adapter quality.
Additionally, device_map="auto" triggers an unstable dispatch path on
Blackwell even on single-GPU systems.

CRITICAL: The evaluator MUST load the model using the same configuration as
the trainer (trainer.py).  If the model was trained in bf16 with eager
attention and fixed device_map, evaluating it in NF4 introduces a dtype
mismatch that corrupts inference output and renders eval metrics meaningless.

The Workaround (active implementation in _load_model_and_tokenizer below)
──────────────────────────────────────────────────────────────────────────
  • Remove NF4 quantisation — load base model in bf16 (torch.bfloat16).
  • Force single-GPU: device_map={"": "cuda:0"}.
  • Disable flash-attention: attn_implementation="eager".
  Matches trainer.py exactly — training and evaluation see identical dtype.

Trade-offs
──────────
  ┌─────────────────────┬──────────────────────────┬──────────────────────────┐
  │                     │ Legacy (NF4/auto)         │ Blackwell-safe (bf16)    │
  ├─────────────────────┼──────────────────────────┼──────────────────────────┤
  │ VRAM (base model)   │ ~4–5 GB (4-bit)          │ ~6–7 GB (bf16)           │
  │ Eval throughput     │ Slightly faster (less I/O)│ Comparable on Blackwell  │
  │ Dtype consistency   │ Mismatch if trainer bf16  │ Matches trainer exactly  │
  │ Blackwell (sm_120)  │ ✗ Corrupt outputs        │ ✓ Validated              │
  │ Ampere (sm_80)      │ ✓ Validated              │ ✓ Compatible             │
  └─────────────────────┴──────────────────────────┴──────────────────────────┘

Reverting to the legacy implementation
───────────────────────────────────────
If you are on a non-Blackwell GPU AND the trainer uses NF4:
  In _load_model_and_tokenizer(), comment out the Blackwell-safe block and
  uncomment the Legacy block.  The two blocks are clearly marked [LEGACY]
  and [BLACKWELL-SAFE].  Also update requirements_fine_tuning.txt.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input
─────
  models/hf/Qwen2.5-Coder-3B-Instruct/   ← base model
  models/adapters/fine_tuning-v{N}/            ← LoRA adapter from trainer.py
  data/fine_tuning_eval.jsonl                  ← held-out eval set
  data/eval_baseline.json                 ← Phase 1 baseline metrics

Output
──────
  data/eval_results_fine_tuning-v{N}.json      ← full metric report
  data/eval_baseline.json                 ← updated with baseline on first run

Usage
─────
  python fine_tuning/evaluator.py
  python fine_tuning/evaluator.py --version v2
  python fine_tuning/evaluator.py --skip-execution   # skip EXPLAIN (no DB needed)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import sqlglot

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
HF_MODEL_DIR  = Path(settings.fine_tuning.hf_model_dir)   # FT_HF_MODEL_DIR (.env)
ADAPTER_DIR   = Path(settings.fine_tuning.adapter_dir)    # FT_ADAPTER_DIR  (.env)
EVAL_DATA     = Path(settings.fine_tuning.eval_data)      # FT_EVAL_DATA    (.env)
BASELINE_PATH = Path(settings.fine_tuning.baseline_path)  # FT_BASELINE_PATH(.env)
RESULTS_DIR   = Path("data")

# ── Regression thresholds ─────────────────────────────────────────────────────
# If a metric is worse than baseline by more than this tolerance,
# the evaluator flags a regression and recommends not exporting.
_REGRESSION_TOLERANCE = 0.03   # 3 percentage points


def _load_eval_pairs(eval_path: Path) -> list[dict[str, Any]]:
    pairs = []
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    logger.info(component="evaluator", event="eval_pairs_loaded", count=len(pairs))
    return pairs


def _load_baseline(baseline_path: Path) -> dict[str, Any]:
    if not baseline_path.exists():
        return {}
    try:
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_model_and_tokenizer(adapter_path: Path):
    """
    Load HuggingFace base model + LoRA adapter for evaluation inference.

    IMPORTANT: The model must be loaded using the same dtype and device
    configuration as trainer.py.  A mismatch (e.g. training in bf16 but
    evaluating in NF4) corrupts inference output and makes eval metrics
    meaningless.  Both files currently use the Blackwell-safe bf16 path.

    See the Blackwell GPU Conflict section in the module docstring for the
    full rationale, trade-off table, and reversion instructions.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # ── [LEGACY] NF4 import — non-Blackwell GPUs only ────────────────────
        # Uncomment when reverting to NF4 on Ampere or older.  Also uncomment
        # the BitsAndBytesConfig block below and update requirements_phase2.txt.
        # from transformers import BitsAndBytesConfig  # [LEGACY]
        #
        # ── [BLACKWELL-SAFE] BitsAndBytesConfig not imported ──────────────────
        # NF4 dequantisation kernels produce corrupt outputs on sm_120.
        # See Blackwell GPU Conflict section in module docstring.
        from peft import PeftModel
    except ImportError as exc:
        print(f"\nERROR: Phase 2 dependencies not installed: {exc}")
        print("  Run:  pip install -r requirements_phase2.txt")
        raise SystemExit(1)

    print(f"Loading tokeniser from {HF_MODEL_DIR}…")
    tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model ───────────────────────────────────────────────────────
    #
    # Two implementations are provided below.  The Blackwell-safe block is
    # active.  The legacy block is commented out immediately above it.
    #
    # ── [LEGACY] NF4 QLoRA — Ampere (sm_80) / Turing (sm_75) and older ───────
    # Works on most consumer GPUs released before 2024.  Produces corrupt
    # outputs on Blackwell (sm_120) — do NOT use on RTX 50-series GPUs.
    # Also produces a dtype mismatch if trainer.py was run in bf16 mode.
    #
    # Pros:  Lower VRAM (~4–5 GB base model after quantisation).
    #        Faster to load.
    # Cons:  Incompatible with Blackwell (sm_120).
    #        Dtype mismatch if trainer used bf16 (invalidates eval metrics).
    #        Requires bitsandbytes in requirements_phase2.txt.
    #
    # To restore: uncomment this block and comment out the Blackwell-safe block.
    #             Also revert trainer.py to NF4 so dtypes are consistent.
    #
    # print("Loading base model in 4-bit NF4 for evaluation…")           # [LEGACY]
    # bnb_config = BitsAndBytesConfig(                                    # [LEGACY]
    #     load_in_4bit              = True,                               # [LEGACY]
    #     bnb_4bit_quant_type       = "nf4",                             # [LEGACY]
    #     bnb_4bit_compute_dtype    = torch.float16,                     # [LEGACY]
    #     bnb_4bit_use_double_quant = True,                              # [LEGACY]
    # )                                                                   # [LEGACY]
    # base_model = AutoModelForCausalLM.from_pretrained(                 # [LEGACY]
    #     str(HF_MODEL_DIR),                                             # [LEGACY]
    #     quantization_config = bnb_config,  # 4-bit NF4                # [LEGACY]
    #     device_map          = "auto",      # auto multi-device map     # [LEGACY]
    #     trust_remote_code   = True,                                    # [LEGACY]
    # )                                                                   # [LEGACY]
    #
    # ── [BLACKWELL-SAFE] bf16 standard LoRA — all GPUs including sm_120 ───────
    # Matches trainer.py dtype exactly — training and evaluation see the same
    # model weights in the same precision.  Safe on Ampere and older too.
    #
    # Pros:  Dtype consistent with bf16 trainer — eval metrics are valid.
    #        Numerically stable on all CUDA architectures.
    #        No bitsandbytes dependency.
    # Cons:  Higher VRAM (~6–7 GB vs ~4–5 GB NF4).
    #        Slightly slower to load.
    #
    print("Loading base model in bf16 (Blackwell-safe, matches trainer dtype)…")  # [BLACKWELL-SAFE]
    base_model = AutoModelForCausalLM.from_pretrained(                             # [BLACKWELL-SAFE]
        str(HF_MODEL_DIR),                                                          # [BLACKWELL-SAFE]
        torch_dtype         = torch.bfloat16,  # matches trainer.py dtype          # [BLACKWELL-SAFE]
        device_map          = {"": "cuda:0"},  # force single-GPU; avoids          # [BLACKWELL-SAFE]
                                               # auto-dispatch NaN on Blackwell    # [BLACKWELL-SAFE]
        attn_implementation = "eager",         # disable flash-attention;          # [BLACKWELL-SAFE]
                                               # incompatible with sm_120 on      # [BLACKWELL-SAFE]
                                               # some driver combinations          # [BLACKWELL-SAFE]
        trust_remote_code   = True,                                                 # [BLACKWELL-SAFE]
    )                                                                               # [BLACKWELL-SAFE]

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    return model, tokenizer


def _generate_sql(
    model,
    tokenizer,
    instruction:   str,
    system_prompt: str,
    max_new_tokens: int = 512,
    # REVIEW FIX (#13): was 256. The model must emit a full JSON object
    # (sql, tables_used, confidence, explanation) — for complex multi-table
    # joins the SQL text alone can exceed 200 tokens, leaving no room for
    # the rest of the JSON contract. 256 risked truncating the JSON mid-field,
    # which would then fail the JSON-parse path in _extract_sql() below and
    # fall through to the regex fallback unnecessarily. 512 matches the
    # budget used elsewhere for generation at inference time
    # (generation/prompt_builder.py / sql_generator.py).
    temperature: float | None = None,
    # Generation temperature. None -> read settings.llm.temperature (the single
    # source of truth used across the codebase). 0 (the recommended default for
    # evaluation) means greedy/deterministic decoding so fine-tuning eval numbers
    # are reproducible run-to-run. Was previously hard-coded to 0.1.
) -> tuple[str, float]:
    """
    Generate SQL for a single NL question using the fine-tuned model.
    Returns (raw_output, latency_ms).
    """
    import torch

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Resolve temperature from the single source of truth when not given.
    eval_temp = settings.llm.temperature if temperature is None else temperature

    # HuggingFace only applies `temperature` when `do_sample=True`; under greedy
    # decoding it is ignored (and warns). So switch modes on the value: temp 0
    # -> deterministic greedy (reproducible), temp > 0 -> sampling at that temp.
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens = max_new_tokens,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    if eval_temp and eval_temp > 0:
        gen_kwargs.update(do_sample=True, temperature=eval_temp)
    else:
        gen_kwargs.update(do_sample=False)  # greedy — reproducible evaluation

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return raw, latency_ms


def _extract_sql(raw_output: str) -> str | None:
    """
    Extract SQL from model output.
    Tries JSON parse first (correct output), then regex fallback.
    """
    import re

    # Try JSON
    try:
        # REVIEW FIX (#6): lstrip("```json") / rstrip("```") strip any
        # leading/trailing characters found in the *set* {`, j, s, o, n} —
        # not the literal substrings "```json" / "```". This could eat real
        # JSON content (e.g. a leading '"' or trailing 's' in a string value)
        # or fail to strip the fence at all if the model's whitespace
        # differs from what was expected. removeprefix/removesuffix match
        # the literal string only.
        cleaned = raw_output.strip()
        cleaned = cleaned.removeprefix("```json").strip()
        cleaned = cleaned.removeprefix("```").strip()  # bare ``` fence, no language tag
        cleaned = cleaned.removesuffix("```").strip()
        data    = json.loads(cleaned)
        return data.get("sql", "").strip() or None
    except (json.JSONDecodeError, AttributeError):
        pass

    # Regex fallback
    match = re.search(r"(SELECT\s.+?)(?:;|$)", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _check_syntax(sql: str) -> bool:
    """Return True if sqlglot can parse the SQL without error."""
    try:
        sqlglot.parse(sql, dialect="postgres")
        return True
    except sqlglot.errors.ParseError:
        return False


def _check_hallucination(sql: str, schema_map: dict) -> bool:
    """
    Return True if ALL table AND column references in the SQL exist in the schema.

    FIX-P2: Extended from table-only to column-level hallucination detection.
    A model could reference a real table but invent a column name
    (e.g. answer_script.evaluator_magic_score).  The original check passed
    these through silently.

    Column check: for each qualified column reference (table.column or
    alias.column) where the table alias can be resolved to a known table,
    verify the column exists in that table's inventory.  Unqualified columns
    and unresolvable aliases are skipped conservatively.
    """
    try:
        stmts = sqlglot.parse(sql, dialect="postgres")
        for stmt in stmts:
            if stmt is None:
                continue

            # Collect CTE aliases
            cte_names = {c.alias.lower() for c in stmt.find_all(sqlglot.exp.CTE) if c.alias}

            # Build alias → table mapping from FROM / JOIN clauses
            alias_to_table: dict[str, str] = {}
            for tbl in stmt.find_all(sqlglot.exp.Table):
                tbl_name = tbl.name.lower() if tbl.name else ""
                alias    = tbl.alias.lower() if tbl.alias else tbl_name
                if tbl_name and tbl_name not in cte_names:
                    alias_to_table[alias] = tbl_name

            # ── Table hallucination check ─────────────────────────────────
            for tbl in stmt.find_all(sqlglot.exp.Table):
                name = tbl.name.lower() if tbl.name else ""
                if name and name not in schema_map and name not in cte_names:
                    return False

            # ── Column hallucination check ────────────────────────────────
            for col in stmt.find_all(sqlglot.exp.Column):
                col_name   = col.name.lower()  if col.name   else ""
                table_ref  = col.table.lower() if col.table  else ""

                if not col_name or not table_ref:
                    # Unqualified column — cannot validate without type inference
                    continue

                # Resolve alias to real table name
                real_table = alias_to_table.get(table_ref, table_ref)
                if real_table not in schema_map:
                    # Could be a CTE alias or subquery alias — skip
                    continue

                inv = schema_map[real_table]
                if hasattr(inv, "columns") and col_name not in inv.columns:
                    logger.debug(
                        component="evaluator",
                        event="column_hallucination_detected",
                        table=real_table,
                        column=col_name,
                    )
                    return False

    except Exception:
        return False
    return True


def _open_eval_connection():
    """
    H4 fix: open a single reusable connection for the entire eval loop instead
    of creating one per pair.  The original _check_execution() called
    psycopg2.connect() on every evaluation pair — up to 200 sequential TCP
    handshakes per eval run.  Returns None if DB is not configured.
    """
    pg = settings.postgres
    if not pg.host:   # C6-aligned: guard on host only, not password
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(
            host     = pg.host,
            port     = pg.port,
            dbname   = pg.database,
            user     = pg.user,
            password = pg.password,
            options  = f"-c statement_timeout={pg.statement_timeout_ms}",
        )
        conn.set_session(readonly=True)
        return conn
    except Exception as exc:
        logger.warning(component="evaluator", event="db_connect_failed", error=str(exc),
                       note="EXPLAIN checks will be skipped for this eval run")
        return None


def _check_execution(sql: str, conn=None) -> bool:
    """
    Return True if EXPLAIN runs without error.
    H4 fix: accepts a pre-opened connection (conn) instead of opening one per call.
    Falls through gracefully if conn is None (DB not configured or connect failed).
    """
    if conn is None:
        return True   # skip gracefully — no DB connection available

    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {sql}")
        cur.close()
        return True
    except Exception:
        # Roll back the failed transaction so the connection remains usable
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def _compare_to_baseline(
    metrics:  dict[str, float],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare current metrics against the stored Phase 1 baseline.
    Returns a dict of {metric: {current, baseline, delta, regression}}.
    """
    baseline_metrics = baseline.get("metrics", {})
    comparison: dict[str, Any] = {}

    for key, current_val in metrics.items():
        base_val = baseline_metrics.get(key)
        if base_val is None:
            comparison[key] = {
                "current":    round(current_val, 4),
                "baseline":   None,
                "delta":      None,
                "regression": False,
                "note":       "No baseline — this run will become the baseline",
            }
            continue

        delta      = current_val - base_val
        regression = delta < -_REGRESSION_TOLERANCE

        comparison[key] = {
            "current":    round(current_val, 4),
            "baseline":   round(base_val, 4),
            "delta":      round(delta, 4),
            "regression": regression,
        }

    return comparison


def _print_report(
    version:    str,
    metrics:    dict[str, float],
    comparison: dict[str, Any],
    n_pairs:    int,
) -> bool:
    """
    Print a human-readable evaluation report.
    Returns True if all metrics pass (no regressions), False otherwise.
    """
    print(f"\n{'═' * 60}")
    print(f"  Evaluation Report — adapter: fine_tuning-{version}")
    print(f"  Eval pairs: {n_pairs}")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<30} {'Current':>8} {'Baseline':>9} {'Delta':>7} {'Status':>10}")
    print(f"  {'─' * 56}")

    any_regression = False
    for metric, info in comparison.items():
        current  = f"{info['current']:.1%}"
        baseline = f"{info['baseline']:.1%}" if info["baseline"] is not None else "  —"
        delta    = f"{info['delta']:+.1%}"   if info["delta"]    is not None else "  —"

        if info["regression"]:
            status = "⚠ REGRESSION"
            any_regression = True
        elif info["baseline"] is None:
            status = "NEW BASELINE"
        elif info["delta"] > 0:
            status = "✓ IMPROVED"
        else:
            status = "✓ OK"

        print(f"  {metric:<30} {current:>8} {baseline:>9} {delta:>7} {status:>10}")

    print(f"{'─' * 60}")

    if any_regression:
        print(
            "\n  ⚠  REGRESSIONS DETECTED — do not export this adapter.\n"
            "     Adjust training (more data, more epochs, lower LR)\n"
            "     and retrain before exporting.\n"
        )
    else:
        print(
            "\n  ✓  All metrics pass.\n"
            f"     Next step: python fine_tuning/export.py --version {version}\n"
        )

    print(f"{'═' * 60}\n")
    return not any_regression


def evaluate(
    version:          str  = "v1",
    skip_execution:   bool = False,
    write_baseline:   bool = False,
) -> bool:
    """
    Run evaluation and return True if all metrics pass.
    """
    adapter_path  = ADAPTER_DIR / f"fine_tuning-{version}"
    results_path  = RESULTS_DIR / f"eval_results_fine_tuning-{version}.json"

    # ── Prerequisite checks ───────────────────────────────────────────────────
    errors = []
    if not HF_MODEL_DIR.exists():
        errors.append(f"Base model not found at {HF_MODEL_DIR}")
    if not adapter_path.exists():
        errors.append(f"Adapter not found at {adapter_path} — run trainer.py first")
    if not EVAL_DATA.exists():
        errors.append(f"Eval data not found at {EVAL_DATA} — run data_pipeline.py first")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)

    # ── Load schema map for hallucination check ───────────────────────────────
    from ingestion.ddl_parser import DDLParser
    parser     = DDLParser()
    tables     = parser.parse_file(Path(settings.ddl_path))
    schema_map = {name.lower(): inv for name, inv in tables.items()}

    # ── Load eval pairs ───────────────────────────────────────────────────────
    eval_pairs = _load_eval_pairs(EVAL_DATA)
    if not eval_pairs:
        print("ERROR: Eval set is empty. Run fine_tuning/data_pipeline.py first.")
        raise SystemExit(1)

    baseline = _load_baseline(BASELINE_PATH)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = _load_model_and_tokenizer(adapter_path)

    # ── Run evaluation ────────────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    # H4 fix: open one DB connection for the entire eval loop.
    # The original _check_execution() opened a fresh psycopg2 connection per
    # pair — up to 200 sequential TCP handshakes per eval run.  One connection
    # opened here is passed to every _check_execution() call and closed after
    # the loop completes.
    eval_conn = None if skip_execution else _open_eval_connection()
    if eval_conn is not None:
        print("✓  DB connection open for EXPLAIN checks.\n")
    elif not skip_execution:
        print("⚠  No DB connection — EXPLAIN checks will be skipped.\n")

    # ── GPU warmup — prevents first-query latency spike skewing p50 ──────────
    # FIX-P2: run 2 throwaway generations before the timed loop so CUDA
    # kernels are warm and latency measurements are representative.
    if eval_pairs:
        print("Warming up GPU (2 throwaway generations)…")
        warmup_pair = eval_pairs[0]
        for _ in range(2):
            _generate_sql(
                model, tokenizer,
                warmup_pair["instruction"],
                warmup_pair["input"],
            )
        print("✓  GPU warm.\n")

    print(f"Evaluating {len(eval_pairs)} pairs…")

    for i, pair in enumerate(eval_pairs):
        instruction   = pair["instruction"]
        system_prompt = pair["input"]
        expected_sql  = pair["output"]

        raw_output, latency_ms = _generate_sql(model, tokenizer, instruction, system_prompt)
        generated_sql          = _extract_sql(raw_output)
        latencies.append(latency_ms)

        syntax_ok   = _check_syntax(generated_sql) if generated_sql else False
        no_halluc   = _check_hallucination(generated_sql, schema_map) if generated_sql and syntax_ok else False
        # H4 fix: pass the shared eval_conn (opened once before this loop)
        exec_ok     = (_check_execution(generated_sql, eval_conn) if generated_sql and syntax_ok and not skip_execution else True)

        # FIX-P1: renamed from semantic_ok to exact_match_ok to accurately
        # reflect what is being measured.  True semantic correctness requires
        # executing both queries against a test DB and comparing result sets —
        # that is a future enhancement requiring a populated test database.
        # Exact SQL match is a conservative lower bound: it will under-report
        # correctness (different-but-equivalent SQL counts as wrong) but never
        # over-report it.  The metric key in the report is also renamed so
        # regression comparison is consistent across versions.
        exact_match_ok = bool(
            generated_sql and
            generated_sql.strip().lower() == expected_sql.strip().lower()
        )

        results.append({
            "instruction":    instruction,
            "expected_sql":   expected_sql,
            "generated_sql":  generated_sql,
            "raw_output":     raw_output,
            "syntax_pass":    syntax_ok,
            "no_halluc":      no_halluc,
            "exec_valid":     exec_ok,
            "exact_match":    exact_match_ok,
            "latency_ms":     round(latency_ms, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(eval_pairs)} evaluated…")

    # H4 fix: close the shared DB connection after the loop
    if eval_conn is not None:
        try:
            eval_conn.close()
        except Exception:
            pass

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = len(results)
    metrics = {
        "syntax_pass_rate":      sum(r["syntax_pass"]  for r in results) / n,
        "no_hallucination_rate": sum(r["no_halluc"]    for r in results) / n,
        "execution_valid_rate":  sum(r["exec_valid"]   for r in results) / n,
        # FIX-P1: renamed from semantic_correct_rate to exact_match_rate
        # to accurately reflect that this is exact SQL string comparison,
        # not semantic / result-set comparison.
        "exact_match_rate":      sum(r["exact_match"]  for r in results) / n,
        "p50_latency_ms":        statistics.median(latencies),
    }

    # ── Compare to baseline ───────────────────────────────────────────────────
    comparison = _compare_to_baseline(metrics, baseline)
    passed     = _print_report(version, metrics, comparison, n)

    # ── Write results ─────────────────────────────────────────────────────────
    import os
    report = {
        "version":    version,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs":    n,
        "metrics":    {k: round(v, 4) for k, v in metrics.items()},
        "comparison": comparison,
        "passed":     passed,
        "results":    results,
    }
    tmp = results_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, results_path)
    print(f"Full results written to: {results_path}")

    # ── Update baseline on first run or when explicitly requested ─────────────
    baseline_is_stub = not baseline.get("metrics")
    if baseline_is_stub or write_baseline:
        updated_baseline = {
            "version":    version,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note":       "Phase 1 baseline — generated by fine_tuning/evaluator.py",
            "metrics":    {k: round(v, 4) for k, v in metrics.items()},
        }
        tmp = BASELINE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(updated_baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, BASELINE_PATH)
        print(f"Baseline written to: {BASELINE_PATH}")

    logger.info(
        component="evaluator",
        event="evaluation_complete",
        version=version,
        passed=passed,
        metrics=metrics,
    )

    # REVIEW FIX (#12): explicitly release the model and clear the CUDA
    # cache before returning. Without this, the ~6-7 GB bf16 base model +
    # adapter stays resident until the Python process exits — relevant if
    # evaluate() is ever called from a longer-lived process (e.g. a future
    # batch-comparison script that evaluates several adapter versions in
    # one run) rather than only as a one-shot CLI invocation.
    import torch
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning Evaluator")
    parser.add_argument("--version",         type=str,  default="v1",
                        help="Adapter version to evaluate (default: v1)")
    parser.add_argument("--skip-execution",  action="store_true",
                        help="Skip EXPLAIN check (use when no DB connection available)")
    parser.add_argument("--write-baseline",  action="store_true",
                        help="Overwrite eval_baseline.json with current results")
    args = parser.parse_args()

    passed = evaluate(
        version        = args.version,
        skip_execution = args.skip_execution,
        write_baseline = args.write_baseline,
    )
    raise SystemExit(0 if passed else 1)
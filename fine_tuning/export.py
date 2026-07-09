"""
fine_tuning/export.py
═══════════════════════════════════════════════════════════════════════════════
GGUF EXPORT PIPELINE  —  turn a trained LoRA adapter into a servable model file.

WHAT THIS SCRIPT DOES (in plain terms)
──────────────────────────────────────
trainer.py produced a small LoRA "adapter" (~30–100 MB of weight *deltas*).
That adapter is useless on its own — it only makes sense sitting on top of the
full base model. This script combines the two and packages the result into a
single quantised .gguf file that llama-server can load directly, exactly like
the Phase-1 inference model. Three steps:

  Step 1 — MERGE:    base model (bf16) + LoRA adapter  → one merged HF model
  Step 2 — CONVERT:  merged HF model                   → F16 .gguf   (llama.cpp)
  Step 3 — QUANTISE: F16 .gguf                          → Q4_K_M .gguf (llama.cpp)

The Q4_K_M file is the deliverable; the merged model and F16 file are large
temporaries that are deleted at the end unless you pass --keep-merged / --keep-f16.

INPUTS
──────
  <base HF model>                         ← FT_HF_MODEL_DIR (full-precision base)
  models/adapters/fine_tuning-v{N}/       ← LoRA adapter written by trainer.py

OUTPUTS
───────
  models/merged/fine_tuning-v{N}/                              ~12 GB (temp)
  models/qwen/qwen2.5-coder-3b-finetuned-v{N}-f16.gguf         ~6 GB  (temp)
  models/qwen/qwen2.5-coder-3b-finetuned-v{N}-q4_k_m.gguf      ~2.4 GB (KEEP)

Peak disk needed during a run: ~26 GB free. Keep the OLD gguf until you have
verified the new one on a few queries, then delete it.

⚠ VRAM NOTE (8 GB card): Step 1 loads the base model onto the GPU by default
  (device_map cuda:0, ~6 GB bf16). STOP llama-server before running, or the two
  together will exceed 8 GB. If the merge OOMs, set FT_MERGE_DEVICE=cpu to merge
  in system RAM instead (slower, but no VRAM pressure, and llama-server can stay up).

TOOL PATHS (override via environment variables, no code edit needed)
───────────────────────────────────────────────────────────────────
  LLAMA_CPP_SOURCE    — dir containing convert_hf_to_gguf.py   (default D:\\llama.cpp)
  LLAMA_PRECOMPILED   — dir containing llama-quantize.exe       (default …\\llama-precompiled)
  LLAMA_QUANTIZE_BIN  — override just the quantiser binary NAME (default llama-quantize.exe)
  LLAMA_SERVER_BIN    — override just the server binary NAME    (default llama-server.exe)
  FT_MERGE_DEVICE     — where Step-1 merge runs: "cuda:0" (default) or "cpu"

MODEL PATHS come from config/settings.py (env prefix FT_): FT_HF_MODEL_DIR,
FT_ADAPTER_DIR, FT_MERGED_DIR, FT_GGUF_OUTPUT_DIR. These MUST match what
trainer.py used, or export will look in the wrong place for the adapter.

Note: "CONFIDENTAIL" in the default path below is the real (misspelled) folder
name on this machine — deliberately not corrected.

USAGE
─────
  python -m fine_tuning.export                     # export v1 (default)
  python -m fine_tuning.export --version v2
  python -m fine_tuning.export --keep-merged --keep-f16   # keep temporaries
  python -m fine_tuning.export --skip-merge        # re-quantise an existing merge
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────
# REVIEW FIX (#4): the prerequisite/skip-merge failures used to call
# `raise SystemExit(1)`, which kills the whole Python interpreter. That is fine
# when run as a script, but fatal if export() is ever called from a notebook,
# a test, or a web handler. We now raise this domain-specific exception instead
# and let the __main__ block below translate it into a process exit code. Code
# that imports export() can catch ExportError and stay alive.
class ExportError(RuntimeError):
    """Raised when the export pipeline cannot proceed (missing files, bad args)."""


# ── Version-label safety ──────────────────────────────────────────────────────
# REVIEW FIX (#1): `version` is interpolated straight into filesystem paths
# (e.g. models/adapters/fine_tuning-{version}). Without validation, a value like
# "../../etc/evil" could escape the intended directory tree. Restrict it to a
# safe character set. This is defence-in-depth even on a single-user box.
_VALID_VERSION = re.compile(r"^[A-Za-z0-9._-]+$")

def _validate_version(version: str) -> str:
    if not _VALID_VERSION.match(version):
        raise ExportError(
            f"Invalid --version {version!r}. Allowed characters: letters, digits, "
            "dot, underscore, hyphen (e.g. v1, v2, exp-2026-01)."
        )
    return version


# ── Default tool paths (llama.cpp) ────────────────────────────────────────────
# These point at the llama.cpp checkout and the precompiled Windows binaries.
# Override via env var without editing source.
_LLAMA_CPP_SOURCE = Path(os.environ.get("LLAMA_CPP_SOURCE", r"D:\llama.cpp"))
_LLAMA_PRECOMPILED = Path(
    os.environ.get(
        "LLAMA_PRECOMPILED",
        r"D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\llama-precompiled",
    )
)

# REVIEW FIX (#6): binary NAMES are overridable so the script is not hard-wired
# to ".exe". On WSL2/Linux you would set LLAMA_QUANTIZE_BIN=llama-quantize etc.
_QUANTIZE_BIN_NAME = os.environ.get("LLAMA_QUANTIZE_BIN", "llama-quantize.exe")
_SERVER_BIN_NAME   = os.environ.get("LLAMA_SERVER_BIN",   "llama-server.exe")

# ── Model paths (env-overridable via config/settings.py → .env, prefix FT_) ───
# Sourced from settings so export reads/writes the SAME locations trainer.py used.
_HF_MODEL_DIR = Path(settings.fine_tuning.hf_model_dir)        # FT_HF_MODEL_DIR
_ADAPTER_DIR  = Path(settings.fine_tuning.adapter_dir)         # FT_ADAPTER_DIR
_MERGED_DIR   = Path(settings.fine_tuning.merged_dir)          # FT_MERGED_DIR
_OUTPUT_DIR   = Path(settings.fine_tuning.gguf_output_dir)     # FT_GGUF_OUTPUT_DIR

_CONVERT_SCRIPT = _LLAMA_CPP_SOURCE  / "convert_hf_to_gguf.py"
_QUANTIZE_BIN   = _LLAMA_PRECOMPILED / _QUANTIZE_BIN_NAME

# REVIEW FIX (#5): subprocess steps get a timeout so a deadlocked binary cannot
# hang the pipeline forever. Convert is the slower step (loads the whole model),
# so it gets the larger budget.
_CONVERT_TIMEOUT_S  = 3600   # 1 hour
_QUANTIZE_TIMEOUT_S = 1800   # 30 minutes

# Where the Step-1 merge runs. GPU by default (fast), CPU as an 8 GB escape hatch.
_MERGE_DEVICE = os.environ.get("FT_MERGE_DEVICE", "cuda:0")


# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite check — fail fast, BEFORE importing torch (which is slow to load)
# ─────────────────────────────────────────────────────────────────────────────
def _check_prerequisites(version: str, skip_merge: bool) -> None:
    """
    Verify every required file/binary exists and surface ALL problems at once
    (rather than dying on the first). Runs before the heavy ML imports so a
    typo in a path fails in milliseconds, not after a 10-second torch import.
    """
    errors: list[str] = []

    # The adapter trainer.py produced. Without it there is nothing to export.
    adapter_path = _ADAPTER_DIR / f"fine_tuning-{version}"
    if not adapter_path.exists():
        errors.append(
            f"LoRA adapter not found: {adapter_path}\n"
            f"  Run:  python -m fine_tuning.trainer --version {version}"
        )

    # The full-precision base is only needed if we are actually merging.
    if not skip_merge and not _HF_MODEL_DIR.exists():
        # REVIEW FIX (#9): print the path via as_posix() so Windows backslashes
        # don't turn into broken escape sequences in the copy-paste command.
        errors.append(
            f"HuggingFace base model not found: {_HF_MODEL_DIR}\n"
            "  Run:  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', "
            f"local_dir='{_HF_MODEL_DIR.as_posix()}')\""
        )

    # llama.cpp conversion script (Step 2).
    if not _CONVERT_SCRIPT.exists():
        errors.append(
            f"convert_hf_to_gguf.py not found: {_CONVERT_SCRIPT}\n"
            f"  Override: set env var LLAMA_CPP_SOURCE to the directory containing it.\n"
            f"  Current:  LLAMA_CPP_SOURCE={_LLAMA_CPP_SOURCE}"
        )

    # llama.cpp quantiser binary (Step 3).
    if not _QUANTIZE_BIN.exists():
        errors.append(
            f"quantiser binary not found: {_QUANTIZE_BIN}\n"
            f"  Override: set LLAMA_PRECOMPILED (directory) or LLAMA_QUANTIZE_BIN (name).\n"
            f"  Current:  LLAMA_PRECOMPILED={_LLAMA_PRECOMPILED}  bin={_QUANTIZE_BIN_NAME}"
        )

    if errors:
        # One combined message, then stop. Raise ExportError (not SystemExit) so
        # a programmatic caller can catch this; __main__ converts it to exit 1.
        raise ExportError("\n\n".join(f"ERROR: {e}" for e in errors))

    # ── Advisory disk-space check (warning only, never blocks) ────────────────
    # REVIEW FIX (#3): the old code checked the CURRENT directory's drive. But
    # the big temporary files land under _MERGED_DIR / _OUTPUT_DIR, which may be
    # on a DIFFERENT drive. Checking the wrong drive could pass while the real
    # target drive is full. `.anchor` is the drive root (e.g. "D:\\") and works
    # even if the directory doesn't exist yet.
    try:
        target_drive = (_MERGED_DIR.anchor or _OUTPUT_DIR.anchor or Path(".").anchor)
        free_gb = shutil.disk_usage(target_drive).free / (1024 ** 3)
        if free_gb < 26:
            print(
                f"\n⚠  WARNING: Only {free_gb:.1f} GB free on {target_drive}.\n"
                "   Peak disk usage during export is ~26 GB:\n"
                "     merged model  ~12 GB (deleted after export unless --keep-merged)\n"
                "     F16 GGUF       ~6 GB (deleted after quantisation unless --keep-f16)\n"
                "     Q4_K_M GGUF  ~2.4 GB (the final deliverable)\n"
                "   Free space before proceeding to avoid a mid-run failure.\n"
            )
    except Exception:
        # A failed disk probe must never stop an otherwise-valid export.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — MERGE adapter into base model
# ─────────────────────────────────────────────────────────────────────────────
def _step_merge(version: str) -> Path:
    """
    Load the full base model, apply the LoRA adapter, fold the adapter weights
    permanently into the base (merge_and_unload), and save the result as a normal
    HuggingFace model directory.

    WHY bf16 (not fp16): this box is an RTX 5060 Ti (Blackwell, sm_120). fp16
    produces NaN logits on sm_120 — the same failure seen in the Gemma runs.
    bf16 is the correct, validated compute dtype for Blackwell.

    WHY the imports are INSIDE this function (deliberate — see review #10): torch/
    transformers/peft take seconds to import and pull in CUDA. Keeping them here
    means `--help` and the fast prerequisite check don't pay that cost, and a
    missing-dependency error surfaces with a clear pip hint instead of a stack
    trace at module load. It is a one-shot function, so the "repeated import"
    cost the linters warn about never actually happens.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise ExportError(
            f"Fine-tuning dependencies not installed: {exc}\n"
            "  Run:  pip install -r requirements_phase2.txt"
        )

    adapter_path = _ADAPTER_DIR / f"fine_tuning-{version}"
    merged_path  = _MERGED_DIR  / f"fine_tuning-{version}"

    print(f"\n{'─' * 60}")
    print(f"  Step 1 — Merge adapter into base model")
    print(f"  Base model : {_HF_MODEL_DIR}")
    print(f"  Adapter    : {adapter_path}")
    print(f"  Output     : {merged_path}")
    print(f"  Merge on   : {_MERGE_DEVICE}")
    print(f"{'─' * 60}")

    merged_path.mkdir(parents=True, exist_ok=True)

    print(
        "\nLoading base model in bf16 for merge…\n"
        "(Full precision required — a quantised model cannot be merged.)\n"
    )

    # device_map controls WHERE the base loads:
    #   "cuda:0" → ~6 GB VRAM. Fast. STOP llama-server first (8 GB card).
    #   "cpu"    → uses system RAM (~12 GB). Slower, but zero VRAM pressure —
    #              use this if the GPU merge OOMs (set FT_MERGE_DEVICE=cpu).
    # attn_implementation="eager" avoids flash-attn issues on sm_120 (harmless on CPU).
    if _MERGE_DEVICE == "cpu":
        device_map = {"": "cpu"}
    else:
        device_map = {"": _MERGE_DEVICE}

    base_model = AutoModelForCausalLM.from_pretrained(
        str(_HF_MODEL_DIR),
        torch_dtype         = torch.bfloat16,
        device_map          = device_map,
        # SECURITY NOTE (review #2): trust_remote_code=True EXECUTES Python that
        # ships inside the model repo. It is required for Qwen's custom modelling
        # code. Safe here because the base is a known, locally-pinned snapshot —
        # but if you ever point FT_HF_MODEL_DIR at an untrusted model, this is the
        # line that would run its code. Keep the base source trusted.
        trust_remote_code   = True,
        attn_implementation = "eager",
    )

    print(f"Loading LoRA adapter from {adapter_path}…")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # merge_and_unload() adds the low-rank deltas (B·A·scaling) back into the
    # original weight matrices and drops the adapter wrappers, leaving a plain
    # model that behaves as if it were trained fully — no adapter needed at serve.
    print("Merging adapter weights into base model (in-place)…")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}…")
    model.save_pretrained(str(merged_path))

    # Save the tokeniser alongside so the merged dir is self-contained for Step 2.
    tokenizer = AutoTokenizer.from_pretrained(str(_HF_MODEL_DIR), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))

    # Report on-disk size (simple recursive sum; fine for a one-shot operation).
    size_gb = sum(f.stat().st_size for f in merged_path.rglob("*") if f.is_file()) / (1024 ** 3)
    print(f"✓  Merge complete. Merged model: {merged_path} ({size_gb:.1f} GB)\n")

    logger.info(
        component="export", event="merge_complete",
        adapter=str(adapter_path), merged=str(merged_path), size_gb=round(size_gb, 2),
    )
    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — CONVERT merged HF model → F16 GGUF
# ─────────────────────────────────────────────────────────────────────────────
def _step_convert(merged_path: Path, version: str) -> Path:
    """
    Run llama.cpp's convert_hf_to_gguf.py to turn the merged HF model into an
    F16 .gguf. We invoke it as a SUBPROCESS (not an import) because that script
    has its own dependency set that may not match this venv.
    """
    f16_gguf = _OUTPUT_DIR / f"qwen2.5-coder-3b-finetuned-{version}-f16.gguf"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'─' * 60}")
    print(f"  Step 2 — Convert HF model → F16 GGUF")
    print(f"  Script : {_CONVERT_SCRIPT}")
    print(f"  Input  : {merged_path}")
    print(f"  Output : {f16_gguf}")
    print(f"{'─' * 60}\n")

    cmd = [
        sys.executable,               # same Python running this script
        str(_CONVERT_SCRIPT),
        str(merged_path),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]

    logger.info(component="export", event="convert_start", cmd=" ".join(map(str, cmd)))

    # capture_output=True (review #10 origin): keep stdout/stderr so a failure's
    #   real error is logged even when run headless/CI (not just "exit code 1").
    # encoding="utf-8" + errors="replace" (REVIEW FIX #12): text=True otherwise
    #   defaults to the Windows locale (often cp1252), which crashes the moment
    #   the subprocess prints a UTF-8/CJK char or an emoji. Force UTF-8.
    # timeout (REVIEW FIX #5): don't hang forever on a wedged converter.
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=_CONVERT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"convert_hf_to_gguf.py exceeded {_CONVERT_TIMEOUT_S}s and was killed.\n"
            "The merged model may be corrupt, or the machine is heavily loaded."
        )

    if result.returncode != 0:
        logger.error(
            component="export", event="convert_failed", returncode=result.returncode,
            stdout=(result.stdout or "")[-4000:], stderr=(result.stderr or "")[-4000:],
        )
        print(result.stdout); print(result.stderr)
        raise RuntimeError(
            f"convert_hf_to_gguf.py exited with code {result.returncode}. "
            "See output above."
        )

    # Exit 0 but no file = the script wrote somewhere else. Fail loudly.
    if not f16_gguf.exists():
        raise RuntimeError(
            f"Conversion reported success but output not found:\n  {f16_gguf}\n"
            "Check the script output for the actual path."
        )

    size_gb = f16_gguf.stat().st_size / (1024 ** 3)
    print(f"\n✓  Conversion complete. F16 GGUF: {f16_gguf} ({size_gb:.1f} GB)\n")
    logger.info(component="export", event="convert_complete",
                output=str(f16_gguf), size_gb=round(size_gb, 2))
    return f16_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — QUANTISE F16 GGUF → Q4_K_M
# ─────────────────────────────────────────────────────────────────────────────
def _step_quantize(f16_gguf: Path, version: str) -> Path:
    """
    Compress the F16 .gguf down to Q4_K_M (4-bit) with llama-quantize. Q4_K_M is
    the SAME level the Phase-1 inference model uses, so the output is a drop-in
    replacement — no other component changes.
    """
    q4_gguf = _OUTPUT_DIR / f"qwen2.5-coder-3b-finetuned-{version}-q4_k_m.gguf"

    print(f"{'─' * 60}")
    print(f"  Step 3 — Quantise F16 GGUF → Q4_K_M")
    print(f"  Binary : {_QUANTIZE_BIN}")
    print(f"  Input  : {f16_gguf}")
    print(f"  Output : {q4_gguf}")
    print(f"{'─' * 60}\n")

    cmd = [str(_QUANTIZE_BIN), str(f16_gguf), str(q4_gguf), "Q4_K_M"]

    logger.info(component="export", event="quantize_start", cmd=" ".join(map(str, cmd)))

    # Same subprocess hardening as Step 2: capture output, force UTF-8, timeout.
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=_QUANTIZE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"llama-quantize exceeded {_QUANTIZE_TIMEOUT_S}s and was killed."
        )

    if result.returncode != 0:
        logger.error(
            component="export", event="quantize_failed", returncode=result.returncode,
            stdout=(result.stdout or "")[-4000:], stderr=(result.stderr or "")[-4000:],
        )
        print(result.stdout); print(result.stderr)
        raise RuntimeError(
            f"llama-quantize exited with code {result.returncode}. See output above."
        )

    if not q4_gguf.exists():
        raise RuntimeError(
            f"Quantisation reported success but output not found:\n  {q4_gguf}"
        )

    size_gb = q4_gguf.stat().st_size / (1024 ** 3)
    print(f"\n✓  Quantisation complete. Q4_K_M GGUF: {q4_gguf} ({size_gb:.1f} GB)\n")
    logger.info(component="export", event="quantize_complete",
                output=str(q4_gguf), size_gb=round(size_gb, 2))
    return q4_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — run the three steps, then clean up temporaries
# ─────────────────────────────────────────────────────────────────────────────
def export(
    version:     str  = "v1",
    keep_merged: bool = False,
    keep_f16:    bool = False,
    skip_merge:  bool = False,
) -> Path:
    """
    Run the full merge → convert → quantise pipeline and return the final
    Q4_K_M .gguf path.

    Args:
        version:     Adapter label — MUST match trainer.py --version.
        keep_merged: Keep the ~12 GB merged HF model (re-quantise later w/ --skip-merge).
        keep_f16:    Keep the ~6 GB F16 GGUF (re-quantise without re-converting).
        skip_merge:  Reuse an already-merged model instead of running Step 1.

    Raises:
        ExportError:  a prerequisite is missing or an argument is invalid.
        RuntimeError: a subprocess step failed or produced no output.
    """
    version = _validate_version(version)          # REVIEW FIX (#1)
    _check_prerequisites(version, skip_merge)

    print(f"\n{'═' * 60}")
    print(f"  GGUF Export — adapter: fine_tuning-{version}")
    print(f"{'═' * 60}")
    logger.info(component="export", event="export_start", version=version,
                skip_merge=skip_merge, keep_merged=keep_merged, keep_f16=keep_f16)

    # ── Step 1: Merge (or reuse an existing merged model) ─────────────────────
    merged_path = _MERGED_DIR / f"fine_tuning-{version}"
    if skip_merge:
        if not merged_path.exists():
            raise ExportError(
                f"--skip-merge requested but merged model not found:\n  {merged_path}\n"
                "Remove --skip-merge to run the merge step first."
            )
        print(f"\n  Skipping merge — using existing model at {merged_path}")
    else:
        merged_path = _step_merge(version)

    # ── Step 2: Convert ───────────────────────────────────────────────────────
    f16_gguf = _step_convert(merged_path, version)

    # ── Step 3: Quantise ──────────────────────────────────────────────────────
    q4_gguf = _step_quantize(f16_gguf, version)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    # NOTE (review #8): we intentionally only clean up on SUCCESS. If a step
    # above raised, the intermediates are left on disk ON PURPOSE so you can
    # inspect them / retry with --skip-merge. On a space-tight machine, a failed
    # run may need manual cleanup of models/merged/ and the *-f16.gguf.
    if not keep_f16 and f16_gguf.exists():
        freed = f16_gguf.stat().st_size / (1024 ** 3)
        print(f"Removing F16 GGUF ({freed:.1f} GB freed)…")
        f16_gguf.unlink()
        print("  Removed.\n")

    if not keep_merged and merged_path.exists():
        freed = sum(f.stat().st_size for f in merged_path.rglob("*") if f.is_file()) / (1024 ** 3)
        print(f"Removing merged HF model ({freed:.1f} GB freed)…")
        shutil.rmtree(merged_path)
        print("  Removed.\n")

    # ── Final summary + ready-to-run serve command ────────────────────────────
    q4_size_gb = q4_gguf.stat().st_size / (1024 ** 3)
    print(f"{'═' * 60}")
    print(f"  ✓  Export complete.")
    print(f"     GGUF : {q4_gguf}")
    print(f"     Size : {q4_size_gb:.1f} GB")
    print(f"{'═' * 60}\n")

    # Printed deploy command. FIX: was `-ngl -1` (ambiguous across llama.cpp
    # builds) with no context size. Use `-ngl 99` to offload all layers to the
    # GPU and `-c 8192` to match the Phase-1 context window.
    server_bin = _LLAMA_PRECOMPILED / _SERVER_BIN_NAME
    print(
        "  To deploy, stop the running llama-server, then start it on the new GGUF:\n\n"
        f"    {server_bin} \\\n"
        f"      -m {q4_gguf} \\\n"
        # f"      -c 8192 -ngl 99 --port 8080\n\n"
        f"      -c 32768 -ngl 99 --port 8080 --chat-template chatml\n\n"
        "  Phase-1 system resumes with zero code changes.\n"
        "  Keep the OLD gguf until you have verified a sample of queries.\n"
    )

    logger.info(component="export", event="export_complete",
                version=version, q4_gguf=str(q4_gguf), size_gb=round(q4_size_gb, 2))
    return q4_gguf


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GGUF Export Pipeline (LoRA adapter → merged → Q4_K_M .gguf)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # REVIEW FIX (#7): document the env vars the code ACTUALLY reads. Model
        # paths come from settings (FT_* prefix); tool paths are read here directly.
        epilog=(
            "Environment overrides (no code change needed):\n"
            "  Tool paths : LLAMA_CPP_SOURCE, LLAMA_PRECOMPILED,\n"
            "               LLAMA_QUANTIZE_BIN, LLAMA_SERVER_BIN\n"
            "  Merge dev  : FT_MERGE_DEVICE  (cuda:0 [default] | cpu)\n"
            "  Model paths: FT_HF_MODEL_DIR, FT_ADAPTER_DIR, FT_MERGED_DIR,\n"
            "               FT_GGUF_OUTPUT_DIR  (via config/settings.py / .env)\n"
        ),
    )
    parser.add_argument("--version", type=str, default="v1",
                        help="Adapter version label — must match trainer.py --version (default: v1)")
    parser.add_argument("--keep-merged", action="store_true",
                        help="Keep the ~12 GB merged HuggingFace model after export")
    parser.add_argument("--keep-f16", action="store_true",
                        help="Keep the ~6 GB F16 GGUF after quantisation")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge — reuse existing models/merged/fine_tuning-v{N}/")
    args = parser.parse_args()

    # Translate our domain exception into a clean process exit. Programmatic
    # callers of export() catch ExportError themselves and are unaffected.
    try:
        export(
            version     = args.version,
            keep_merged = args.keep_merged,
            keep_f16    = args.keep_f16,
            skip_merge  = args.skip_merge,
        )
    except ExportError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
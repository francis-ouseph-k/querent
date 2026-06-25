"""
fine_tuning/export.py
═══════════════════════════════════════════════════════════════════════════════
GGUF Export Pipeline
────────────────────────────────
Merges the LoRA adapter into the base model and exports a production-ready
Q4_K_M GGUF file that is a drop-in replacement for the Phase 1 inference model.

Three-step pipeline
───────────────────
  Step 1 — Merge:    base model + LoRA adapter → merged HuggingFace model
  Step 2 — Convert:  merged HF model → F16 GGUF  (convert_hf_to_gguf.py)
  Step 3 — Quantise: F16 GGUF → Q4_K_M GGUF     (llama-quantize.exe)

Input
──────
  models/hf/Qwen2.5-Coder-3B-Instruct/        ← HuggingFace base model (~6 GB)
  models/adapters/fine_tuning-v{N}/                 ← LoRA adapter from trainer.py

Output
──────
  models/merged/fine_tuning-v{N}/                                    ← merged HF model (~12 GB, temporary)
  models/qwen/qwen2.5-coder-3b-finetuned-v{N}-f16.gguf         ← F16 GGUF   (~6 GB, temporary)
  models/qwen/qwen2.5-coder-3b-finetuned-v{N}-q4_k_m.gguf      ← Q4_K_M GGUF (~2.4 GB, KEEP)

Disk requirement (peak during export): ~26 GB free.
Keep the old GGUF until the new model is verified, then delete to reclaim ~2.4 GB.

Tool paths
──────────
Default paths match this machine's filesystem.  Override via env vars without
touching source code:
  LLAMA_CPP_SOURCE   — directory containing convert_hf_to_gguf.py
  LLAMA_PRECOMPILED  — directory containing llama-quantize.exe
  HF_MODEL_DIR       — HuggingFace base model directory

Note: CONFIDENTAIL in the default LLAMA_PRECOMPILED path below matches the
actual filesystem spelling on this machine — not a typo to correct.

Usage
─────
  python fine_tuning/export.py
  python fine_tuning/export.py --version v2
  python fine_tuning/export.py --keep-merged --keep-f16
  python fine_tuning/export.py --skip-merge   # re-quantise from an existing merged model
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Default tool paths ────────────────────────────────────────────────────────
# Override any of these via environment variable without editing source code.
_LLAMA_CPP_SOURCE = Path(
    os.environ.get("LLAMA_CPP_SOURCE", r"D:\llama.cpp")
)
_LLAMA_PRECOMPILED = Path(
    os.environ.get(
        "LLAMA_PRECOMPILED",
        r"D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\llama-precompiled",
    )
)
_HF_MODEL_DIR = Path(
    os.environ.get("HF_MODEL_DIR", "models/hf/Qwen2.5-Coder-3B-Instruct")
)

_ADAPTER_DIR    = Path("models/adapters")
_MERGED_DIR     = Path("models/merged")
_OUTPUT_DIR     = Path("models/qwen")

_CONVERT_SCRIPT = _LLAMA_CPP_SOURCE  / "convert_hf_to_gguf.py"
_QUANTIZE_BIN   = _LLAMA_PRECOMPILED / "llama-quantize.exe"


# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite check
# ─────────────────────────────────────────────────────────────────────────────

def _check_prerequisites(version: str, skip_merge: bool) -> None:
    """
    Fail fast with clear messages if any required path or binary is absent.
    Called before importing torch/transformers so errors surface immediately.
    """
    errors: list[str] = []

    adapter_path = _ADAPTER_DIR / f"fine_tuning-{version}"
    if not adapter_path.exists():
        errors.append(
            f"LoRA adapter not found: {adapter_path}\n"
            f"  Run:  python fine_tuning/trainer.py --version {version}"
        )

    if not skip_merge and not _HF_MODEL_DIR.exists():
        errors.append(
            f"HuggingFace base model not found: {_HF_MODEL_DIR}\n"
            "  Run:  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', local_dir='{_HF_MODEL_DIR}')\""
        )

    if not _CONVERT_SCRIPT.exists():
        errors.append(
            f"convert_hf_to_gguf.py not found: {_CONVERT_SCRIPT}\n"
            f"  Override: set env var LLAMA_CPP_SOURCE to the directory containing it.\n"
            f"  Current:  LLAMA_CPP_SOURCE={_LLAMA_CPP_SOURCE}"
        )

    if not _QUANTIZE_BIN.exists():
        errors.append(
            f"llama-quantize.exe not found: {_QUANTIZE_BIN}\n"
            f"  Override: set env var LLAMA_PRECOMPILED to the directory containing it.\n"
            f"  Current:  LLAMA_PRECOMPILED={_LLAMA_PRECOMPILED}"
        )

    if errors:
        print("\n" + "\n\n".join(f"ERROR: {e}" for e in errors))
        raise SystemExit(1)

    # Advisory disk check — not a hard block, just an early warning
    try:
        free_gb = shutil.disk_usage(Path(".")).free / (1024 ** 3)
        if free_gb < 26:
            print(
                f"\n⚠  WARNING: Only {free_gb:.1f} GB free on current drive.\n"
                "   Peak disk usage during export is ~26 GB:\n"
                "     merged model  ~12 GB (deleted after export unless --keep-merged)\n"
                "     F16 GGUF       ~6 GB (deleted after quantisation unless --keep-f16)\n"
                "     Q4_K_M GGUF  ~2.4 GB (the final deliverable)\n"
                "   Free space before proceeding to avoid a mid-run failure.\n"
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Merge
# ─────────────────────────────────────────────────────────────────────────────

def _step_merge(version: str) -> Path:
    """
    Merge LoRA adapter into the base model and save the merged HuggingFace model.

    Uses bfloat16 (not float16) because this machine has an RTX 5060 Ti
    (Blackwell sm_120).  float16 causes NaN logits on this architecture —
    the same issue documented in the Gemma training runs.  bfloat16 is the
    correct compute dtype for Blackwell.

    Returns the path to the saved merged model directory.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        print(f"\nERROR: Fine-tuning dependencies not installed: {exc}")
        print("  Run:  pip install -r requirements_phase2.txt")
        raise SystemExit(1)

    adapter_path = _ADAPTER_DIR / f"fine_tuning-{version}"
    merged_path  = _MERGED_DIR  / f"fine_tuning-{version}"

    print(f"\n{'─' * 60}")
    print(f"  Step 1 — Merge adapter into base model")
    print(f"  Base model : {_HF_MODEL_DIR}")
    print(f"  Adapter    : {adapter_path}")
    print(f"  Output     : {merged_path}")
    print(f"{'─' * 60}")

    merged_path.mkdir(parents=True, exist_ok=True)

    print(
        "\nLoading base model in bf16 for merge…\n"
        "(Full precision required — quantised models cannot be merged.\n"
        " ~6 GB VRAM. Stop llama-server first if it is running.)\n"
    )

    # bf16 + device_map to single GPU — the validated Blackwell-safe load path.
    # attn_implementation="eager" avoids flash-attn compatibility issues on sm_120.
    base_model = AutoModelForCausalLM.from_pretrained(
        str(_HF_MODEL_DIR),
        torch_dtype           = torch.bfloat16,
        device_map            = {"": "cuda:0"},
        trust_remote_code     = True,
        attn_implementation   = "eager",
    )

    print(f"Loading LoRA adapter from {adapter_path}…")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter weights into base model (in-place)…")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}…")
    model.save_pretrained(str(merged_path))

    tokenizer = AutoTokenizer.from_pretrained(
        str(_HF_MODEL_DIR), trust_remote_code=True
    )
    tokenizer.save_pretrained(str(merged_path))

    size_gb = sum(
        f.stat().st_size for f in merged_path.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    print(f"✓  Merge complete. Merged model: {merged_path} ({size_gb:.1f} GB)\n")

    logger.info(
        component="export", event="merge_complete",
        adapter=str(adapter_path), merged=str(merged_path), size_gb=round(size_gb, 2),
    )
    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Convert HF → F16 GGUF
# ─────────────────────────────────────────────────────────────────────────────

def _step_convert(merged_path: Path, version: str) -> Path:
    """
    Convert the merged HuggingFace model to a F16 GGUF using convert_hf_to_gguf.py.

    Invokes the script via subprocess so it runs in its own Python environment
    (the llama.cpp convert script has its own dependency requirements that may
    differ from this venv).

    Returns the path to the produced F16 GGUF file.
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
        sys.executable,
        str(_CONVERT_SCRIPT),
        str(merged_path),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]

    logger.info(component="export", event="convert_start", cmd=" ".join(str(c) for c in cmd))
    # REVIEW FIX (#10): capture_output=True so stdout/stderr are available for
    # logging on failure. The previous check=False with no capture meant a
    # failure's actual error message only ever reached an interactive
    # terminal — anyone running this from a script, CI job, or with output
    # redirected to a file would see only "exited with code 1" and nothing
    # about why. text=True decodes to str instead of bytes.
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(
            component="export", event="convert_failed",
            returncode=result.returncode,
            stdout=result.stdout[-4000:] if result.stdout else "",
            stderr=result.stderr[-4000:] if result.stderr else "",
        )
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"convert_hf_to_gguf.py exited with code {result.returncode}.\n"
            "Check the output above for details."
        )

    if not f16_gguf.exists():
        raise RuntimeError(
            f"Conversion appeared to succeed (exit code 0) but output not found:\n"
            f"  {f16_gguf}\n"
            "Check script output for the actual output path and set it explicitly."
        )

    size_gb = f16_gguf.stat().st_size / (1024 ** 3)
    print(f"\n✓  Conversion complete. F16 GGUF: {f16_gguf} ({size_gb:.1f} GB)\n")

    logger.info(
        component="export", event="convert_complete",
        output=str(f16_gguf), size_gb=round(size_gb, 2),
    )
    return f16_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Quantise F16 GGUF → Q4_K_M
# ─────────────────────────────────────────────────────────────────────────────

def _step_quantize(f16_gguf: Path, version: str) -> Path:
    """
    Quantise the F16 GGUF to Q4_K_M using llama-quantize.exe.

    Q4_K_M is the same quantisation level as the original inference model —
    the output is a drop-in replacement with no other component changes needed.

    Returns the path to the final Q4_K_M GGUF file.
    """
    q4_gguf = _OUTPUT_DIR / f"qwen2.5-coder-3b-finetuned-{version}-q4_k_m.gguf"

    print(f"{'─' * 60}")
    print(f"  Step 3 — Quantise F16 GGUF → Q4_K_M")
    print(f"  Binary : {_QUANTIZE_BIN}")
    print(f"  Input  : {f16_gguf}")
    print(f"  Output : {q4_gguf}")
    print(f"{'─' * 60}\n")

    cmd = [str(_QUANTIZE_BIN), str(f16_gguf), str(q4_gguf), "Q4_K_M"]

    logger.info(component="export", event="quantize_start", cmd=" ".join(str(c) for c in cmd))
    # REVIEW FIX (#10): same fix as _step_convert above — capture output so
    # a failure's actual error is logged and printed, not just the exit code.
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(
            component="export", event="quantize_failed",
            returncode=result.returncode,
            stdout=result.stdout[-4000:] if result.stdout else "",
            stderr=result.stderr[-4000:] if result.stderr else "",
        )
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"llama-quantize.exe exited with code {result.returncode}.\n"
            "Check the output above for details."
        )

    if not q4_gguf.exists():
        raise RuntimeError(
            f"Quantisation appeared to succeed (exit code 0) but output not found:\n"
            f"  {q4_gguf}"
        )

    size_gb = q4_gguf.stat().st_size / (1024 ** 3)
    print(f"\n✓  Quantisation complete. Q4_K_M GGUF: {q4_gguf} ({size_gb:.1f} GB)\n")

    logger.info(
        component="export", event="quantize_complete",
        output=str(q4_gguf), size_gb=round(size_gb, 2),
    )
    return q4_gguf


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def export(
    version:     str  = "v1",
    keep_merged: bool = False,
    keep_f16:    bool = False,
    skip_merge:  bool = False,
) -> Path:
    """
    Run the full three-step export pipeline.

    Returns the path to the final Q4_K_M GGUF file.

    Args:
        version:     Adapter version label — must match trainer.py --version.
        keep_merged: Retain the ~12 GB merged HF model after export.
                     Useful for re-quantising at a different quantisation level
                     without re-merging (use --skip-merge on next run).
        keep_f16:    Retain the ~6 GB F16 GGUF after quantisation.
                     Useful for re-quantising without repeating the convert step.
        skip_merge:  Skip Step 1 — use an already-merged model from a prior run.
                     The merged model must exist at models/merged/fine_tuning-v{N}/.
    """
    _check_prerequisites(version, skip_merge)

    print(f"\n{'═' * 60}")
    print(f"  GGUF Export — adapter: fine_tuning-{version}")
    print(f"{'═' * 60}")

    logger.info(
        component="export", event="export_start", version=version,
        skip_merge=skip_merge, keep_merged=keep_merged, keep_f16=keep_f16,
    )

    # ── Step 1: Merge ─────────────────────────────────────────────────────────
    merged_path = _MERGED_DIR / f"fine_tuning-{version}"
    if skip_merge:
        if not merged_path.exists():
            print(
                f"\nERROR: --skip-merge requested but merged model not found:\n"
                f"  {merged_path}\n"
                "Remove --skip-merge to run the merge step first."
            )
            raise SystemExit(1)
        print(f"\n  Skipping merge — using existing model at {merged_path}")
    else:
        merged_path = _step_merge(version)

    # ── Step 2: Convert ───────────────────────────────────────────────────────
    f16_gguf = _step_convert(merged_path, version)

    # ── Step 3: Quantise ──────────────────────────────────────────────────────
    q4_gguf = _step_quantize(f16_gguf, version)

    # ── Cleanup: remove large intermediate files unless explicitly retained ────
    if not keep_f16 and f16_gguf.exists():
        f16_size_gb = f16_gguf.stat().st_size / (1024 ** 3)
        print(f"Removing F16 GGUF ({f16_size_gb:.1f} GB freed)…")
        f16_gguf.unlink()
        print("  Removed.\n")

    if not keep_merged and merged_path.exists():
        merged_size_gb = sum(
            f.stat().st_size for f in merged_path.rglob("*") if f.is_file()
        ) / (1024 ** 3)
        print(f"Removing merged HF model ({merged_size_gb:.1f} GB freed)…")
        shutil.rmtree(merged_path)
        print("  Removed.\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    q4_size_gb = q4_gguf.stat().st_size / (1024 ** 3)

    print(f"{'═' * 60}")
    print(f"  ✓  Export complete.")
    print(f"     GGUF : {q4_gguf}")
    print(f"     Size : {q4_size_gb:.1f} GB")
    print(f"{'═' * 60}\n")

    server_bin = _LLAMA_PRECOMPILED / "llama-server.exe"
    print(
        "  To deploy, stop llama-server, then restart pointing at the new GGUF:\n\n"
        f"    {server_bin} \\\n"
        f"      -m {q4_gguf} \\\n"
        f"      -ngl -1 --port 8080\n\n"
        "  Phase 1 system resumes — zero code changes required.\n"
        "  Keep the old GGUF until you have verified a sample of queries.\n"
    )

    logger.info(
        component="export", event="export_complete",
        version=version, q4_gguf=str(q4_gguf), size_gb=round(q4_size_gb, 2),
    )
    return q4_gguf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GGUF Export Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tool path env var overrides (no code change needed):\n"
            "  LLAMA_CPP_SOURCE   — directory containing convert_hf_to_gguf.py\n"
            "  LLAMA_PRECOMPILED  — directory containing llama-quantize.exe\n"
            "  HF_MODEL_DIR       — HuggingFace base model directory\n"
        ),
    )
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Adapter version label — must match trainer.py --version (default: v1)",
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Keep the ~12 GB merged HuggingFace model after export",
    )
    parser.add_argument(
        "--keep-f16", action="store_true",
        help="Keep the ~6 GB F16 GGUF after quantisation",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip merge step — use existing model at models/merged/fine_tuning-v{N}/",
    )
    args = parser.parse_args()

    export(
        version     = args.version,
        keep_merged = args.keep_merged,
        keep_f16    = args.keep_f16,
        skip_merge  = args.skip_merge,
    )
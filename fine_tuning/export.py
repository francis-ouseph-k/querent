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
  LLAMA_CPP_SOURCE    — REQUIRED. Dir containing convert_hf_to_gguf.py. This is a
                        shared external llama.cpp checkout (outside this repo), so
                        there is no baked default — set it in .env / the environment.
  LLAMA_PRECOMPILED   — Dir containing the quantiser/server binaries. Default is the
                        repo-relative "../llama-precompiled" (a sibling of the project
                        inside the workspace); override for any other location.
  LLAMA_QUANTIZE_BIN  — override just the quantiser binary NAME (default llama-quantize.exe)
  LLAMA_SERVER_BIN    — override just the server binary NAME    (default llama-server.exe)
  FT_MERGE_DEVICE     — where Step-1 merge runs: "cuda:0" (default) or "cpu"

MODEL PATHS come from config/settings.py (env prefix FT_): FT_HF_MODEL_DIR,
FT_ADAPTER_DIR, FT_MERGED_DIR, FT_GGUF_OUTPUT_DIR. These MUST match what
trainer.py used, or export will look in the wrong place for the adapter.

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
from utils.executables import probe_executable

logger = get_logger(__name__)


class ExportError(RuntimeError):
    """Raised when the export pipeline cannot proceed (missing files, bad args)."""


_VALID_VERSION = re.compile(r"^[A-Za-z0-9._-]+$")

def _validate_version(version: str) -> str:
    if not _VALID_VERSION.match(version):
        raise ExportError(
            f"Invalid --version {version!r}. Allowed characters: letters, digits, "
            "dot, underscore, hyphen (e.g. v1, v2, exp-2026-01)."
        )
    return version


_HF_MODEL_DIR = Path(settings.fine_tuning.hf_model_dir)
_ADAPTER_DIR  = Path(settings.fine_tuning.adapter_dir)
_MERGED_DIR   = Path(settings.fine_tuning.merged_dir)
_OUTPUT_DIR   = Path(settings.fine_tuning.gguf_output_dir)

_CONVERT_TIMEOUT_S  = 3600
_QUANTIZE_TIMEOUT_S = 1800


# ── Tool paths / merge device (ALL sourced from settings → .env) ─────────────
# export.py reads its config the SAME way main.py / batch_run.py do: through
# pydantic settings, which loads the one project .env. Nothing here reads
# os.environ directly, so a value in .env is honoured without exporting it into
# the shell first. Relative values (defaults or .env overrides) anchor to the
# WORKSPACE (project's parent) — NOT the shell CWD — so they resolve the same
# wherever export is launched from. Layout this assumes:
#     <workspace>/                  ← _WORKSPACE  (project's parent)
#     ├── <project>/                ← _PROJECT_ROOT (this file: project/fine_tuning/)
#     └── llama-precompiled/        ← quantiser/server binaries, sibling of project
#     <external>/llama.cpp/         ← convert_hf_to_gguf.py (LLAMA_CPP_SOURCE, required)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/<project>
_WORKSPACE    = _PROJECT_ROOT.parent                  # …/<workspace>

_ft = settings.fine_tuning


def _anchor(raw: str, default_rel: str) -> Path:
    """
    Resolve a directory that HAS a sensible default. Uses `raw` if non-empty,
    else `default_rel`; a relative result anchors to the WORKSPACE (not CWD) and
    is always resolved (absolute inputs are normalised too).
    """
    p = Path(raw or default_rel).expanduser()
    if not p.is_absolute():
        p = _WORKSPACE / p
    return p.resolve()


def _optional(raw: str) -> Path | None:
    """
    Resolve a path that has NO baked default (lives outside the repo, differs per
    machine). Returns None when `raw` is empty so `--help` / import still work;
    the missing-value error is raised later in _check_prerequisites.
    """
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


_MERGE_DEVICE      = _ft.merge_device
_QUANTIZE_BIN_NAME = _ft.llama_quantize_bin
_SERVER_BIN_NAME   = _ft.llama_server_bin

# llama.cpp SOURCE checkout (convert_hf_to_gguf.py). Required, no default → None
# until provided; guarded in _check_prerequisites before Step 2 runs.
_LLAMA_CPP_SOURCE: Path | None = _optional(_ft.llama_cpp_source)

# Precompiled binaries dir. Empty in .env → repo-relative default beside project.
_LLAMA_PRECOMPILED = _anchor(_ft.llama_precompiled, "llama-precompiled")

_CONVERT_SCRIPT: Path | None = (
    (_LLAMA_CPP_SOURCE / "convert_hf_to_gguf.py") if _LLAMA_CPP_SOURCE else None
)
_QUANTIZE_BIN = _LLAMA_PRECOMPILED / _QUANTIZE_BIN_NAME


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
        errors.append(
            f"HuggingFace base model not found: {_HF_MODEL_DIR}\n"
            "  Run:  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', "
            f"local_dir='{_HF_MODEL_DIR.as_posix()}')\""
        )

    # llama.cpp conversion script (Step 2). LLAMA_CPP_SOURCE has no default
    # because it is an external, shared checkout — so first confirm it was set,
    # then confirm the script is actually there.
    if _LLAMA_CPP_SOURCE is None:
        errors.append(
            "LLAMA_CPP_SOURCE is not set. It must point at your llama.cpp checkout "
            "(the directory that contains convert_hf_to_gguf.py).\n"
            "  Windows   : $env:LLAMA_CPP_SOURCE='D:\\llama.cpp'\n"
            "  Linux/WSL : export LLAMA_CPP_SOURCE=/path/to/llama.cpp\n"
            "  Or set it in .env (see .env.example)."
        )
    elif not _CONVERT_SCRIPT.exists():
        errors.append(
            f"convert_hf_to_gguf.py not found: {_CONVERT_SCRIPT}\n"
            f"  LLAMA_CPP_SOURCE is set to: {_LLAMA_CPP_SOURCE}\n"
            f"  Check that this directory is a llama.cpp checkout."
        )

    # llama.cpp precompiled binaries (Step 3). Validate in order of specificity
    # so the error names the ROOT problem, not a symptom:
    #   1. the directory itself (a wrong LLAMA_PRECOMPILED gives a misleading
    #      "quantise.exe not found" when the real fault is the folder),
    #   2. the quantiser is PRESENT *and* LAUNCHABLE (probe_executable catches
    #      wrong-arch / missing-DLL/.so / (Linux) no execute-bit — failures that
    #      .exists() passes but that would otherwise crash Step 3 AFTER the
    #      ~12 GB merge).
    # The server binary is only used to PRINT a deploy command at the end, so it
    # is advisory (a warning further down), never a hard error here.
    if not _LLAMA_PRECOMPILED.exists():
        errors.append(
            f"llama precompiled directory not found: {_LLAMA_PRECOMPILED}\n"
            f"  Override: set env var LLAMA_PRECOMPILED to the folder holding "
            f"{_QUANTIZE_BIN_NAME} / {_SERVER_BIN_NAME}\n"
            f"  (default is the repo-relative 'llama-precompiled' beside the project)."
        )
    elif not _LLAMA_PRECOMPILED.is_dir():
        errors.append(
            f"LLAMA_PRECOMPILED is not a directory: {_LLAMA_PRECOMPILED}"
        )
    else:
        quant_err = probe_executable(_QUANTIZE_BIN, "quantiser binary")
        if quant_err:
            errors.append(
                quant_err + "\n"
                f"  Override: set LLAMA_PRECOMPILED (directory) or "
                f"LLAMA_QUANTIZE_BIN (name).\n"
                f"  Current:  LLAMA_PRECOMPILED={_LLAMA_PRECOMPILED}  "
                f"bin={_QUANTIZE_BIN_NAME}"
            )

    if errors:
        raise ExportError("\n\n".join(f"ERROR: {e}" for e in errors))

    # ── Advisory: server binary (deploy command only, never blocks) ──────
    _server_bin = _LLAMA_PRECOMPILED / _SERVER_BIN_NAME
    if _LLAMA_PRECOMPILED.exists() and probe_executable(_server_bin, "server binary"):
        print(
            f"\n⚠  NOTE: server binary not found or not runnable: {_server_bin}\n"
            f"   Export will still succeed; only the printed deploy command at the\n"
            f"   end will be wrong. Set LLAMA_SERVER_BIN if your binary has a\n"
            f"   different name (e.g. 'llama-server' without .exe on Linux/WSL).\n"
        )

    # ── Advisory disk-space check (warning only, never blocks) ────────────────
    # The big temporaries land under _MERGED_DIR / _OUTPUT_DIR, which may be on a
    # DIFFERENT drive than CWD. `.anchor` is the drive root (e.g. "D:\\") and
    # works even if the directory doesn't exist yet.
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
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — MERGE adapter into base model
# ─────────────────────────────────────────────────────────────────────────────
def _step_merge(version: str) -> Path:
    """
    Load the full base model, apply the LoRA adapter, fold the adapter weights
    permanently into the base (merge_and_unload), and save the result as a normal
    HuggingFace model directory.

    WHY bf16 (not fp16): Blackwell (sm_120) produces NaN logits under fp16 for
    this workload. bf16 is the correct, validated compute dtype for sm_120. This
    is a property of the GPU + kernels, not the OS.

    WHY the imports are INSIDE this function: torch/transformers/peft take seconds
    to import and pull in CUDA. Keeping them here means `--help` and the fast
    prerequisite check don't pay that cost, and a missing-dependency error
    surfaces with a clear pip hint instead of a stack trace at module load.
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
        # SECURITY NOTE: trust_remote_code=True EXECUTES Python that ships inside
        # the model repo. Required for Qwen's custom modelling code. Safe here
        # because the base is a known, locally-pinned snapshot — but if you ever
        # point FT_HF_MODEL_DIR at an untrusted model, this is the line that would
        # run its code. Keep the base source trusted.
        trust_remote_code   = True,
        attn_implementation = "eager",
    )

    print(f"Loading LoRA adapter from {adapter_path}…")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # merge_and_unload() adds the low-rank deltas (B·A·scaling) back into the
    # original weight matrices and drops the adapter wrappers, leaving a plain
    # model that behaves as if trained fully — no adapter needed at serve.
    print("Merging adapter weights into base model (in-place)…")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}…")
    model.save_pretrained(str(merged_path))

    # Save the tokeniser alongside so the merged dir is self-contained for Step 2.
    tokenizer = AutoTokenizer.from_pretrained(str(_HF_MODEL_DIR), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))

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
        sys.executable,
        str(_CONVERT_SCRIPT),
        str(merged_path),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]

    logger.info(component="export", event="convert_start", cmd=" ".join(map(str, cmd)))

    # capture_output: keep stdout/stderr so a failure's real error is logged even
    #   when run headless (not just "exit code 1").
    # encoding="utf-8" + errors="replace": text=True otherwise defaults to the
    #   Windows locale (often cp1252), which crashes the moment the subprocess
    #   prints a UTF-8/CJK char or emoji. Force UTF-8.
    # timeout: don't hang forever on a wedged converter.
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
    version = _validate_version(version)
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

    # ── Cleanup (only on SUCCESS — failed runs leave intermediates for retry) ─
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

    server_bin = _LLAMA_PRECOMPILED / _SERVER_BIN_NAME
    print(
        "  To deploy, stop the running llama-server, then start it on the new GGUF:\n\n"
        f"    {server_bin} \\\n"
        f"      -m {q4_gguf} \\\n"
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
        epilog=(
            "Environment overrides (no code change needed):\n"
            "  Tool paths : LLAMA_CPP_SOURCE (REQUIRED — external llama.cpp checkout),\n"
            "               LLAMA_PRECOMPILED (default: ../llama-precompiled beside project),\n"
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
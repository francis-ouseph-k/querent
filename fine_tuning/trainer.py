"""
fine_tuning/trainer.py
═══════════════════════════════════════════════════════════════════════════════
Fine-Tuning Trainer
──────────────────────────────
Fine-tunes Qwen2.5-Coder-3B-Instruct on the training pairs produced by
fine_tuning/data_pipeline.py using parameter-efficient LoRA adapters.  The base
model weights are never modified; only the adapter (~50–100 MB) is trained.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLACKWELL GPU CONFLICT — READ BEFORE CHANGING MODEL LOADING CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Background
──────────
The original implementation used BitsAndBytes NF4 4-bit quantisation (QLoRA)
with fp16 precision and device_map="auto".  This is the standard, widely
documented configuration for fine-tuning large models on consumer GPUs and
works correctly on Ampere (sm_80) and earlier architectures.

The Problem
───────────
On NVIDIA RTX 50-series "Blackwell" GPUs (compute capability sm_120 — e.g.
RTX 5060 Ti, RTX 5070, RTX 5080, RTX 5090), this configuration produces
all-NaN logits silently from the very first training step.  Loss reports NaN,
gradients are undefined, and the resulting adapter is unusable.  The failure
is silent — no exception is raised; the NaN propagates through every weight
update and the model degrades rather than learns.

Root causes identified across 4 training runs on RTX 5060 Ti (sm_120):
  (a) BitsAndBytes NF4 dequantisation kernels have not been validated for
      sm_120 and produce numerically corrupt outputs on that architecture.
  (b) fp16 mixed precision is more prone to underflow/overflow than bf16
      on Blackwell.  Blackwell's Tensor Cores are optimised for bf16 and
      tf32, not fp16.
  (c) device_map="auto" triggers a multi-device dispatch path in
      transformers even on single-GPU systems, which interacts with the
      Blackwell kernel issues to amplify the NaN propagation.
  (d) Flash-attention (the default attention implementation in recent
      transformers) is incompatible with sm_120 on some driver +
      transformers combinations, compounding the instability.

The Workaround (active implementation below)
────────────────────────────────────────────
  • Remove BitsAndBytes quantisation entirely — load the base model in
    full bf16 precision (torch_dtype=torch.bfloat16).
  • Force single-GPU device placement: device_map={"": "cuda:0"}.
  • Disable flash-attention: attn_implementation="eager".
  • Use bf16=True / fp16=False in SFTConfig.
  • Standard LoRA adapters replace QLoRA adapters (rank and alpha unchanged).

Validated: 4 training runs, RTX 5060 Ti (8 GB VRAM, sm_120, CUDA 13.1,
driver 591.74), all producing stable loss curves and non-NaN logits.

Trade-offs
──────────
  ┌─────────────────────┬──────────────────────────┬──────────────────────────┐
  │                     │ Legacy (NF4/fp16/auto)   │ Blackwell-safe (bf16)    │
  ├─────────────────────┼──────────────────────────┼──────────────────────────┤
  │ VRAM (base model)   │ ~4–5 GB (4-bit)          │ ~6–7 GB (bf16)           │
  │ Adapter size        │ ~50 MB                   │ ~100 MB                  │
  │ Training throughput │ Slightly faster (less I/O)│ Comparable on Blackwell  │
  │ Numerical stability │ Lower (fp16, NF4)        │ Higher (bf16 native)     │
  │ Blackwell (sm_120)  │ ✗ NaN logits             │ ✓ Validated              │
  │ Ampere (sm_80)      │ ✓ Validated              │ ✓ Compatible             │
  │ Turing (sm_75)      │ ✓ Validated              │ ✓ Compatible (8+ GB)     │
  │ bitsandbytes req.   │ Yes                      │ No                       │
  └─────────────────────┴──────────────────────────┴──────────────────────────┘

Reverting to the legacy implementation
───────────────────────────────────────
If you are on a non-Blackwell GPU and want to restore NF4 QLoRA:
  1. In the imports block (step 2 inside train()), uncomment the
     BitsAndBytesConfig import and comment out the Blackwell note.
  2. In step 2 (model load), comment out the Blackwell-safe block and
     uncomment the Legacy block.
  3. In step 6 (SFTConfig), set fp16=True, bf16=False.
  4. In requirements_phase2.txt, uncomment the bitsandbytes line.
Each site is clearly marked with [LEGACY] and [BLACKWELL-SAFE] labels.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hardware requirement
────────────────────
  8 GB VRAM minimum (bf16 path).  Stop llama-server before running — the GPU
  must be fully free.  Training and inference cannot share the GPU on 8 GB.

  Estimated training time (RTX 5060 Ti, 8 GB VRAM, bf16 LoRA):
    200 pairs  →  ~1.5 hours
    500 pairs  →  ~3–4 hours
  1,000 pairs  →  ~6–8 hours

  Estimated training time (Ampere/Turing, 8 GB VRAM, NF4 QLoRA legacy):
    200 pairs  →  ~1 hour
    500 pairs  →  ~2–3 hours
  1,000 pairs  →  ~4–6 hours

Input
─────
  models/hf/Qwen2.5-Coder-3B-Instruct/   ← HuggingFace base model (~6 GB)
  data/fine_tuning_train.jsonl                  ← produced by data_pipeline.py

Output
──────
  models/adapters/fine_tuning-v{N}/
    adapter_config.json
    adapter_model.safetensors   ← LoRA adapter weights only; base untouched

CRITICAL CONSTRAINT
───────────────────
The prompt format in _build_prompt() MUST match generation/prompt_builder.py
exactly.  Fine-tuning on a different format than inference degrades the model.

Usage
─────
  python fine_tuning/trainer.py
  python fine_tuning/trainer.py --version v2 --epochs 5
  python fine_tuning/trainer.py --lora-rank 32 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.logging_config import get_logger
from config.settings import settings
from fine_tuning.preprocess.pipeline import run as run_preprocess, PreprocessConfig

logger = get_logger(__name__)

# REVIEW FIX (#9): fixed seed for reproducibility — data shuffling, LoRA
# initialisation, and dropout were previously non-deterministic across runs,
# making it impossible to tell whether a metric change came from a real
# improvement or run-to-run noise.
DEFAULT_SEED = 42

# ── Paths (env-overridable via config/settings.py → .env, prefix FT_) ─────────
# Defaults live in FineTuningSettings. Override in <project root>/.env:
#   FT_ADAPTER_DIR=../models/adapters        (README sibling layout)
#   FT_HF_MODEL_DIR=../models/hf/Qwen2.5-Coder-3B-Instruct
#   FT_TRAIN_DATA=data/fine_tuning_train.jsonl
HF_MODEL_DIR  = Path(settings.fine_tuning.hf_model_dir)
ADAPTER_DIR   = Path(settings.fine_tuning.adapter_dir)
TRAIN_DATA    = Path(settings.fine_tuning.train_data)

# ── Default hyperparameters ───────────────────────────────────────────────────
# These are tuned for an 8 GB GPU with gradient checkpointing enabled.
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 1 * 16 = 16
# These values are shared between the legacy and Blackwell-safe paths.
LORA_RANK          = 16
LORA_ALPHA         = 32     # [SUPERSEDED — FIX-F8] alpha is now computed as 2×rank inside
                            # train(); this constant is kept only for the docstring table.
LORA_DROPOUT       = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE      = 2e-4
NUM_EPOCHS         = 3
BATCH_SIZE         = 1      # per-device batch size (8 GB: batch 1 is the safe default)
GRAD_ACCUM_STEPS   = 16     # effective batch = 16 (was 2*8; now 1*16)
MAX_SEQ_LENGTH     = settings.fine_tuning.max_seq   # token ceiling per training example.
                            # SINGLE SOURCE OF TRUTH: comes from .env MAX_SEQ_LENGTH
                            # (settings.fine_tuning.max_seq, default 2048; legacy alias
                            # FT_MAX_SEQ still accepted). The SAME value
                            # drives the preprocessor's fit_rows budget, so the corpus and
                            # the training ceiling can never diverge. Do NOT hardcode a
                            # different number here — change MAX_SEQ_LENGTH in .env instead, then
                            # re-run preprocessing so fit.jsonl is refitted to the new budget.
                            # 8 GB VRAM note: seq 2048 spills to shared RAM (~130 s/step) but
                            # trains correctly; 1024 truncates past the assistant turn on this
                            # corpus (reserve alone reaches ~1112 tok) → masked → zero loss.
WARMUP_RATIO       = 0.03
LR_SCHEDULER       = "cosine"
SAVE_STEPS         = 50     # save checkpoint every N steps
LOGGING_STEPS      = 10


def _check_prerequisites() -> None:
    """
    Fail fast with clear messages if prerequisites are not met.
    Better to catch this before importing torch/transformers.
    """
    errors = []

    if not HF_MODEL_DIR.exists():
        errors.append(
            f"HuggingFace base model not found at {HF_MODEL_DIR}\n"
            "  Download it once with:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', "
            f"local_dir='{HF_MODEL_DIR}')\""
        )

    if not TRAIN_DATA.exists():
        errors.append(
            f"Training data not found at {TRAIN_DATA}\n"
            "  Preprocessing does NOT run automatically before training.\n"
            "  Build the cleaned corpus first, either:\n"
            "    python -m fine_tuning.preprocess.pipeline --force        (standalone build)\n"
            "    python -m fine_tuning.trainer --version vN --preprocess  (build + train)\n"
            "  then re-run training."
        )

    if errors:
        print("\n" + "\n\n".join(f"ERROR: {e}" for e in errors))
        raise SystemExit(1)

    # Count training pairs
    n_pairs = sum(1 for line in TRAIN_DATA.read_text(encoding="utf-8").splitlines() if line.strip())
    if n_pairs < 50:
        print(
            f"\n⚠  WARNING: Only {n_pairs} training pairs in {TRAIN_DATA}.\n"
            "   Fine-tuning on fewer than 50 pairs may degrade the model.\n"
            "   Proceeding anyway — monitor eval metrics carefully.\n"
        )
    else:
        print(f"\n✓  {n_pairs} training pairs found.")


def _build_prompt(instruction: str, system_prompt: str, output: str = "") -> str:
    """
    Build the Qwen2.5 instruct chat template prompt.

    CRITICAL: This format must match generation/prompt_builder.py exactly.
    Qwen2.5-Instruct uses the ChatML format:
      <|im_start|>system
      {system}<|im_end|>
      <|im_start|>user
      {instruction}<|im_end|>
      <|im_start|>assistant
      {output}<|im_end|>

    At inference time (Phase 1), llama-server applies this template
    automatically from the model's chat_template metadata.  During training
    we apply it explicitly so the model sees the same format.
    """
    if output:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
    # Inference-only format (no output label)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _load_dataset(train_path: Path) -> "Dataset":
    """
    Load training pairs from JSONL and return a HuggingFace Dataset.

    FIX-P1: Removed unused `tokenizer` and `max_seq_length` parameters.
    The SFTTrainer handles tokenisation internally via `dataset_text_field`
    and `max_seq_length` in SFTConfig.  Accepting them here implied explicit
    tokenisation was happening, which it was not.

    Each record must have a 'text' field containing the fully-formatted
    ChatML prompt (built by data_pipeline.py with schema context included).
    """
    from datasets import Dataset

    records = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt = _build_prompt(
            instruction   = item["instruction"],
            system_prompt = item["input"],
            output        = item["output"],
        )
        records.append({"text": prompt})

    logger.info(component="trainer", event="dataset_loaded", records=len(records))
    return Dataset.from_list(records)


def _split_train_dev(dataset: "Dataset", dev_fraction: float = 0.10, seed: int = DEFAULT_SEED):
    """
    REVIEW FIX (#8): hold out a small dev split from the training data so
    SFTTrainer can report eval_loss during training.  This is separate from
    the held-out set evaluator.py uses for the 5 production metrics — this
    split exists purely to catch overfitting (rising eval_loss while train
    loss keeps falling) while the run is still in progress, before GPU hours
    are spent on a model nobody can use.

    10% is a default, not a tuned value — on a 200-pair corpus this holds out
    ~20 examples, the practical minimum to get a non-noisy eval_loss signal.
    """
    split = dataset.train_test_split(test_size=dev_fraction, seed=seed)
    return split["train"], split["test"]


def train(
    version:              str   = "v1",
    lora_rank:            int   = LORA_RANK,
    epochs:               int   = NUM_EPOCHS,
    batch_size:           int   = BATCH_SIZE,
    lr:                   float = LEARNING_RATE,
    resume_from_checkpoint: str | None = None,
    seed:                  int   = DEFAULT_SEED,
) -> Path:
    """
    Run LoRA fine-tuning and save the adapter.

    Returns the path to the saved adapter directory.
    """
    # REVIEW FIX (#9): set_seed before any data shuffling, model init, or
    # LoRA layer initialisation so runs are reproducible. Must be called
    # before _load_dataset() and get_peft_model() to take effect.
    from transformers import set_seed
    set_seed(seed)

    # ── Lazy imports — only needed during training ────────────────────────────
    try:
        import torch
        # TF32: harmless on Blackwell. Speeds any residual fp32 matmul paths.
        # Small effect here (training is bf16, so most matmuls are already bf16),
        # but zero downside to training quality.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            # M3 fix: TrainingArguments removed — SFTConfig from trl is used instead.
            # Keeping TrainingArguments in the import was a dead import from an
            # earlier version; it added confusion about which config class is active.
            #
            # ── [LEGACY] NF4 import — non-Blackwell GPUs only ─────────────────
            # Uncomment the line below when reverting to NF4 QLoRA on Ampere or
            # older.  Also uncomment the BitsAndBytesConfig block in step 2 and
            # set fp16=True / bf16=False in step 6.  Requires bitsandbytes in
            # requirements_phase2.txt.
            # BitsAndBytesConfig,       # [LEGACY] NF4 quantisation config
            #
            # ── [BLACKWELL-SAFE] NF4 not imported ─────────────────────────────
            # BitsAndBytesConfig is intentionally absent.  NF4 dequantisation
            # kernels produce all-NaN logits on sm_120 (RTX 50-series).
            # See the Blackwell GPU Conflict section at the top of this file.
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        print(
            f"\nERROR: Fine-tuning dependencies not installed: {exc}\n"
            "  Run:  pip install -r requirements_phase2.txt\n"
        )
        raise SystemExit(1)

    adapter_path = ADAPTER_DIR / f"fine_tuning-{version}"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # ── Verify GPU ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print(
            "\n⚠  WARNING: No CUDA GPU detected. Training on CPU will be extremely slow.\n"
            "   If you have a GPU, ensure CUDA drivers and torch+cu* are installed.\n"
            "   Stop llama-server to free the GPU if it is running.\n"
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        # Report FREE VRAM, not total. total_memory ignores whatever is already
        # resident — a running llama-server can hold 4-6 GB, so a card that
        # reports "8 GB total" may have only 2 GB free and will OOM on load.
        # mem_get_info() returns (free, total) in bytes.
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        vram_gb      = total_bytes / 1e9
        free_vram_gb = free_bytes  / 1e9
        print(f"\n✓  GPU: {gpu_name} ({vram_gb:.1f} GB total, {free_vram_gb:.1f} GB free)")
        if free_vram_gb < 7.5:
            print(
                f"⚠  WARNING: Only {free_vram_gb:.1f} GB VRAM free.\n"
                # [BLACKWELL-SAFE] bf16 full-precision base model footprint:
                "   bf16 LoRA requires ~7–8 GB. Training may OOM.\n"
                "   Stop llama-server (or any other GPU process) to free VRAM.\n"
                # [LEGACY] NF4 footprint was lower (~4–5 GB for the base model),
                # so this threshold was less likely to trigger on Ampere/Turing.
                "   Try reducing --batch-size to 1.\n"
            )

    logger.info(
        component="trainer",
        event="training_start",
        version=version,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        adapter_path=str(adapter_path),
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # ── 1. Load tokeniser ─────────────────────────────────────────────────────
    print(f"\nLoading tokeniser from {HF_MODEL_DIR}…")
    tokenizer = AutoTokenizer.from_pretrained(
        str(HF_MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load base model ────────────────────────────────────────────────────
    #
    # Two implementations are provided below.  The Blackwell-safe block is
    # active.  The legacy block is commented out immediately above it.
    # See the header comment for the full rationale and trade-off table.
    #
    # ── [LEGACY] NF4 QLoRA — Ampere (sm_80) / Turing (sm_75) and older ───────
    # Works on most consumer GPUs released before 2024.  Produces all-NaN
    # logits on Blackwell (sm_120) — do NOT use on RTX 50-series GPUs.
    #
    # Pros:  Lower VRAM (~4–5 GB base model footprint after quantisation).
    #        ~50 MB adapter output.
    #        Faster loading (less data to transfer to GPU).
    # Cons:  Incompatible with Blackwell (sm_120).
    #        NF4 dequantisation adds per-layer overhead at each forward pass.
    #        fp16 is less numerically stable than bf16 on modern Tensor Cores.
    #
    # To restore: uncomment this block and comment out the Blackwell-safe block.
    #
    # print("Loading base model in 4-bit NF4 (QLoRA)…")                # [LEGACY]
    # bnb_config = BitsAndBytesConfig(                                  # [LEGACY]
    #     load_in_4bit              = True,                              # [LEGACY]
    #     bnb_4bit_quant_type       = "nf4",                            # [LEGACY]
    #     bnb_4bit_compute_dtype    = torch.float16,                    # [LEGACY]
    #     bnb_4bit_use_double_quant = True,                             # [LEGACY]
    # )                                                                  # [LEGACY]
    # model = AutoModelForCausalLM.from_pretrained(                     # [LEGACY]
    #     str(HF_MODEL_DIR),                                            # [LEGACY]
    #     quantization_config = bnb_config,   # 4-bit NF4               # [LEGACY]
    #     device_map          = "auto",       # auto multi-device map    # [LEGACY]
    #     trust_remote_code   = True,                                    # [LEGACY]
    # )                                                                  # [LEGACY]
    # model.config.use_cache = False                                     # [LEGACY]
    #
    # ── [BLACKWELL-SAFE] bf16 standard LoRA — all GPUs including sm_120 ───────
    # Validated on RTX 5060 Ti (sm_120, 8 GB VRAM, CUDA 13.1, driver 591.74).
    # Produces stable loss curves and non-NaN logits on Blackwell.
    # Also runs correctly on Ampere and older — safe for all platforms.
    #
    # Pros:  Numerically stable on all CUDA architectures.
    #        No bitsandbytes dependency.
    #        bf16 is the native Tensor Core dtype on Blackwell and Ampere.
    #        Simpler load path — fewer failure modes.
    # Cons:  Higher VRAM (~6–7 GB base model in bf16 vs ~4–5 GB NF4).
    #        Adapter output ~100 MB vs ~50 MB for NF4.
    #        Slightly slower to load (more bytes transferred to GPU).
    #
    print("Loading base model in bf16 (Blackwell-safe, no NF4 quantisation)…")  # [BLACKWELL-SAFE]
    model = AutoModelForCausalLM.from_pretrained(                                # [BLACKWELL-SAFE]
        str(HF_MODEL_DIR),                                                        # [BLACKWELL-SAFE]
        torch_dtype         = torch.bfloat16,  # native bf16 — no quantisation   # [BLACKWELL-SAFE]
        device_map          = {"": "cuda:0"},  # force single-GPU, avoids auto   # [BLACKWELL-SAFE]
                                               # dispatch NaN on Blackwell        # [BLACKWELL-SAFE]
        attn_implementation = "eager",         # disable flash-attention;         # [BLACKWELL-SAFE]
                                               # incompatible with sm_120 on     # [BLACKWELL-SAFE]
                                               # some driver combinations         # [BLACKWELL-SAFE]
        trust_remote_code   = True,                                               # [BLACKWELL-SAFE]
    )                                                                             # [BLACKWELL-SAFE]
    model.config.use_cache = False                                                # [BLACKWELL-SAFE]

    # ── 3. Validate LoRA target modules exist in model ────────────────────────
    # FIX-P2: verify target modules are present before applying LoRA so the
    # error is clear rather than a silent wrong-layer application.
    actual_module_names = [name for name, _ in model.named_modules()]
    missing_modules = [
        m for m in LORA_TARGET_MODULES
        if not any(m in a for a in actual_module_names)
    ]
    if missing_modules:
        raise ValueError(
            f"LoRA target modules not found in model: {missing_modules}\n"
            f"Update LORA_TARGET_MODULES at the top of trainer.py to match "
            f"the actual module names in {HF_MODEL_DIR}."
        )
    logger.info(component="trainer", event="lora_modules_validated", modules=LORA_TARGET_MODULES)

    # ── 4. Apply LoRA ─────────────────────────────────────────────────────────
    # LoRA rank, alpha, dropout, and target modules are identical between the
    # legacy and Blackwell-safe paths.  Only the base model precision changes.
    # FIX-F8: alpha follows rank. LORA_ALPHA was a constant 32 designed for
    # rank 16 (effective scale alpha/r = 2). The v4 run used --lora-rank 8
    # with alpha still 32 → scale 4, doubling how hard the adapter's learned
    # behaviour is imprinted at merge time. alpha = 2×rank keeps the scale
    # constant no matter what rank the CLI picks.
    lora_alpha = 2 * lora_rank
    print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})…")
    lora_config = LoraConfig(
        r                = lora_rank,
        lora_alpha       = lora_alpha,
        lora_dropout     = LORA_DROPOUT,
        target_modules   = LORA_TARGET_MODULES,
        bias             = "none",
        task_type        = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # LoRA + gradient_checkpointing: the checkpointed forward runs on input
    # embeddings that don't require grad, so autograd finds no path to the LoRA
    # params and backward dies with "element 0 ... does not require grad".
    # This hook re-enables grad flow through the frozen embeddings.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── 5. Load dataset ───────────────────────────────────────────────────────
    # FIX-P1: _load_dataset no longer accepts tokenizer/max_seq_length —
    # SFTTrainer handles tokenisation internally via SFTConfig.
    full_dataset = _load_dataset(TRAIN_DATA)

    # REVIEW FIX (#8): hold out a dev split so SFTTrainer reports eval_loss
    # during training. Without this there was no signal to catch overfitting
    # until evaluator.py ran after training finished — by then GPU hours were
    # already spent. This dev split is separate from evaluator.py's held-out
    # set; it exists only to monitor the training run itself.
    # FIX-F9: in-loop eval is hard-disabled on 8 GB (eval_strategy="no" below),
    # so holding out 10% only shrinks an already tiny corpus (~38 rows wasted
    # on v4's 378). Train on 100%; re-enable the split together with
    # eval_strategy="steps" on a bigger GPU.
    _EVAL_ENABLED = False   # keep in sync with eval_strategy in SFTConfig below
    if _EVAL_ENABLED:
        dataset, dev_dataset = _split_train_dev(full_dataset, dev_fraction=0.10, seed=seed)
        print(f"Training on {len(dataset)} examples, {len(dev_dataset)} held out for eval_loss tracking…")
    else:
        dataset, dev_dataset = full_dataset, None
        print(f"Training on {len(dataset)} examples (no dev split — in-loop eval disabled on 8 GB)…")

    # ── 6. Training arguments ─────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir                  = str(adapter_path / "checkpoints"),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        # Non-reentrant checkpointing plays correctly with PEFT/LoRA grad flow;
        # the default reentrant path is what triggers the no-grad backward error.
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        #
        # ── Optimizer — must match the active model-load path ────────────────
        #
        # REVIEW FIX (#2): paged_adamw_8bit is a bitsandbytes optimizer. It
        # requires the bitsandbytes package to be installed. The Blackwell-safe
        # path deliberately removes bitsandbytes (NF4 kernels are unstable on
        # sm_120) — keeping this optimizer here meant training would either
        # crash with an ImportError (bitsandbytes absent) or silently
        # re-introduce the exact library the Blackwell fix was designed to
        # avoid (bitsandbytes present for some other reason).
        #
        # optim = "paged_adamw_8bit",  # [LEGACY] — bitsandbytes optimizer,
        #                              # restore only alongside the [LEGACY]
        #                              # NF4 model-load block in step 2 and
        #                              # the bitsandbytes line in
        #                              # requirements_phase2.txt
        optim = "adamw_torch",        # [BLACKWELL-SAFE] standard PyTorch AdamW,
                                       # no bitsandbytes dependency
        #
        learning_rate               = lr,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        seed                        = seed,  # REVIEW FIX (#9): reproducible shuffling
        # Explicit gradient clipping. TRL defaults this to 1.0, but LoRA on a
        # small corpus is prone to occasional gradient spikes — pinning it makes
        # the run reproducible regardless of the TRL version's default.
        max_grad_norm               = 1.0,
        # Checkpoint selection. With load_best_model_at_end=False (current), the
        # run keeps the LAST checkpoint. EarlyStoppingCallback (added at trainer
        # construction below) still halts the run when eval_loss stops improving —
        # it just doesn't roll back to the best-eval_loss checkpoint afterwards.
        # To instead restore the best checkpoint, set this True (requires the
        # eval/save strategies below to both be "steps", which they are).
        # metric_for_best_model / greater_is_better only take effect when this
        # is True, but are left set so flipping the flag needs no other change.
        #
        # ── [8GB-DISABLED] in-loop eval + best-checkpoint restore ────────────
        # WHY DISABLED: in-loop eval runs a full forward whose LM loss
        # materialises logits [1, seq, 151936] upcast to fp32 — a ~9 GB spike
        # even at seq 2048 on top of the 6 GB bf16 base. On an 8 GB card this
        # OOMs, and because eval fires BEFORE the checkpoint save it takes the
        # step down with it (no resume point written). Confirmed: v1 run died
        # at step 50 in evaluation_loop → shift_logits.contiguous().
        # These three lines (load_best + metric + greater_is_better) require
        # in-loop eval, so they are off together with eval_strategy below.
        # TO RESTORE (GPU with >=16 GB, or if you shrink eval to a few short
        # rows): uncomment the block, set eval_strategy="steps", and re-add the
        # EarlyStoppingCallback at trainer construction (step 7).
        # load_best_model_at_end      = True,          # [8GB-DISABLED]
        # metric_for_best_model       = "eval_loss",   # [8GB-DISABLED]
        # greater_is_better           = False,         # [8GB-DISABLED]
        #
        # [8GB-SAFE] keep the LAST checkpoint; validate post-hoc via
        # evaluator.py / batch_run (RESULT-2) instead of in-loop eval_loss.
        load_best_model_at_end      = False,
        save_strategy               = "steps",
        #
        # REVIEW FIX (#5): TRL's SFTTrainer can default to packing=True in
        # some versions when dataset_text_field is set, concatenating
        # multiple examples into one sequence to maximise GPU utilisation.
        # Each of our examples is a complete, deliberately-formatted ChatML
        # prompt (system + schema context + question + SQL) — packing can
        # slice through the boundary between two examples, so the model
        # would see "assistant\n<sql>...<|im_end|><|im_start|>system\n..."
        # mid-sequence, training on malformed prompt boundaries.
        packing                      = False,
        #
        # REVIEW FIX (#8): eval_loss reporting against the dev split held out
        # above. ── [8GB-DISABLED] ── see the load_best_model_at_end note above:
        # in-loop eval OOMs on 8 GB (9 GB fp32-logit spike). Kept off; the
        # dev_dataset is still built and passed to SFTTrainer so re-enabling is
        # a one-line flip back to "steps".
        # eval_strategy              = "steps",   # [8GB-DISABLED] OOMs on 8 GB
        eval_strategy                = "no",     # [8GB-SAFE]
        eval_steps                   = SAVE_STEPS,
        #
        # ── Precision flags — must match the model load dtype above ──────────
        #
        # [LEGACY]         fp16=True,  bf16=False
        #   Use with the NF4 QLoRA block in step 2.
        #   fp16 was the original choice; it is less stable on Blackwell.
        #   On Ampere and Turing fp16 is fine and gives slightly faster training
        #   than bf16 because fp16 Tensor Core throughput is higher on those
        #   architectures.
        # fp16 = True,   # [LEGACY] — restore with NF4 block in step 2
        # bf16 = False,  # [LEGACY] — restore with NF4 block in step 2
        #
        # [BLACKWELL-SAFE] fp16=False, bf16=True
        #   bf16 is the native precision for Blackwell Tensor Cores.
        #   fp16 overflows/underflows on sm_120 with the loaded bf16 base model,
        #   producing NaN gradients.  bf16 throughout eliminates this.
        #   Also safe on Ampere and Turing; no functional difference vs fp16
        #   for this workload.
        fp16 = False,  # [BLACKWELL-SAFE] fp16 causes NaN on sm_120
        bf16 = True,   # [BLACKWELL-SAFE] native Blackwell Tensor Core dtype
        #
        max_seq_length              = MAX_SEQ_LENGTH,
        dataset_text_field          = "text",
        dataloader_pin_memory       = True,   # default under CUDA; explicit for version-stability
        # dataloader_num_workers left at 0: Windows spawn overhead + 452 tiny
        # pre-tokenised rows means workers>0 would slow startup, not speed it.
        report_to                   = "none",
        run_name                    = f"fine_tuning-{version}",
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    # Early stopping: halt when eval_loss (on the dev split held out above) has
    # not improved for `patience` consecutive evals — saves GPU hours once the
    # model stops improving. With load_best_model_at_end=False (see SFTConfig)
    # the LAST checkpoint is kept; set that flag True to restore the best-eval_loss
    # checkpoint instead. EarlyStoppingCallback lives in transformers; SFTTrainer
    # (from trl) subclasses the HF Trainer, so the callback machinery is shared.
    from transformers import EarlyStoppingCallback

    # ── Completion-only masking ───────────────────────────────────────────────
    # Without this, SFTTrainer computes loss over the ENTIRE sequence — the
    # ~2k-token schema head + system prompt included — so the SQL/JSON label
    # (a few hundred tokens) contributes almost none of the gradient and the
    # model is trained to regurgitate schema dumps. DataCollatorForCompletionOnlyLM
    # masks every token before the assistant turn, so loss lands ONLY on the JSON
    # the model must actually produce at serve time.
    #
    # The response template must match how ChatML frames the assistant turn in
    # _build_prompt: "<|im_start|>assistant\n". We pass it as TOKEN IDS (not a
    # string) because <|im_start|> is a special token; id-matching avoids the
    # mid-sequence whitespace/merge mismatch that makes the string form silently
    # fail to find the template (which would mask the WHOLE sequence → zero loss).
    from trl import DataCollatorForCompletionOnlyLM

    # ── VRAM telemetry ────────────────────────────────────────────────────────
    # Logs GPU memory each logging step and around evaluation. On an 8 GB card
    # this is the difference between "it crashed" and "it crashed allocating
    # 9.2 GB at the first eval" — the peak counters survive the step that OOMs,
    # so the last printed line points straight at the culprit.
    from transformers import TrainerCallback

    class VRAMMonitorCallback(TrainerCallback):
        _GB = 1024 ** 3

        def _log(self, tag: str):
            if not torch.cuda.is_available():
                return
            a  = torch.cuda.memory_allocated()     / self._GB   # live tensors
            r  = torch.cuda.memory_reserved()      / self._GB   # cached by allocator
            pa = torch.cuda.max_memory_allocated() / self._GB   # peak live (since reset)
            pr = torch.cuda.max_memory_reserved()  / self._GB   # peak reserved
            free, total = torch.cuda.mem_get_info(0)
            print(
                f"[VRAM {tag}] alloc {a:.2f} / reserved {r:.2f} | "
                f"peak_alloc {pa:.2f} / peak_reserved {pr:.2f} | "
                f"free {free/self._GB:.2f} of {total/self._GB:.2f} GB"
            )

        def on_step_end(self, args, state, control, **kwargs):
            # Only print on logging steps to avoid flooding stdout.
            if state.global_step % args.logging_steps == 0:
                self._log(f"step {state.global_step}")
            # Reset peak each step so peak_alloc reflects THIS step's high-water
            # mark, not the whole run — makes a single spiking step obvious.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_evaluate(self, args, state, control, **kwargs):
            self._log(f"eval @ step {state.global_step}")

    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    completion_collator = DataCollatorForCompletionOnlyLM(
        response_template = response_template_ids,
        tokenizer         = tokenizer,
    )

    # ── GUARD: response template must survive truncation ──────────────────────
    # If MAX_SEQ_LENGTH cuts the sequence before "<|im_start|>assistant\n", the
    # collator finds no template → labels all -100 → NaN/zero loss → dead adapter.
    # This is exactly the 1024-vs-2048 mismatch that silently wasted GPU hours.
    # Fail LOUD here (cheap CPU tokenisation of a few rows) instead of discovering
    # it hours later. Probe the LONGEST rows — they are the ones that truncate.
    _rt = response_template_ids
    _probe_rows = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["text"]),
                         reverse=True)[:min(8, len(dataset))]
    for _i in _probe_rows:
        _ids = tokenizer(dataset[_i]["text"], truncation=True,
                         max_length=MAX_SEQ_LENGTH)["input_ids"]
        _found = any(_ids[j:j + len(_rt)] == _rt
                     for j in range(len(_ids) - len(_rt) + 1))
        if not _found:
            raise SystemExit(
                f"FATAL: assistant response template not found within the first "
                f"{MAX_SEQ_LENGTH} tokens after truncation (row {_i}). The label is "
                f"being cut off → completion-only masking will zero the loss.\n"
                f"  Fix: raise MAX_SEQ_LENGTH in .env AND re-run preprocessing so "
                f"fit.jsonl is refitted to the same budget:\n"
                f"    python -m fine_tuning.preprocess.pipeline --force\n"
                f"  (current MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH})"
            )
    logger.info(component="trainer", event="response_template_guard_passed",
                max_seq_length=MAX_SEQ_LENGTH, probed=len(_probe_rows))

    trainer = SFTTrainer(
        model          = model,
        train_dataset  = dataset,
        eval_dataset   = dev_dataset,  # REVIEW FIX (#8)
        args           = training_args,
        tokenizer      = tokenizer,
        data_collator  = completion_collator,  # completion-only loss masking
        # ── [8GB-DISABLED] EarlyStoppingCallback ─────────────────────────────
        # WHY DISABLED: EarlyStopping halts on eval_loss, which requires in-loop
        # eval — off on 8 GB (see SFTConfig note). With load_best_model_at_end
        # also False, the TRL/HF assertion
        #   "EarlyStoppingCallback requires load_best_model_at_end = True"
        # would fire on train start anyway. Restore both together on a bigger GPU.
        # callbacks    = [EarlyStoppingCallback(early_stopping_patience=3)],  # [8GB-DISABLED]
        callbacks      = [VRAMMonitorCallback()],   # per-step GPU memory telemetry
    )

    print(f"\nStarting training — {epochs} epochs…")
    print("This will take 1–6 hours depending on corpus size and GPU.\n")

    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}\n")

    # FIX-P2: pass resume_from_checkpoint to trainer.train() so interrupted
    # runs can be resumed without losing progress.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── 8. Save adapter ───────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter to {adapter_path}…")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    logger.info(
        component="trainer",
        event="training_complete",
        adapter_path=str(adapter_path),
        note="Run fine_tuning/evaluator.py next to check metrics before exporting",
    )

    print(
        f"\n✓  Training complete.\n"
        f"   Adapter saved to: {adapter_path}\n"
        f"\nNext step: python fine_tuning/evaluator.py --version {version}\n"
    )

    return adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning LoRA Trainer")
    parser.add_argument("--version",    type=str,   default="v1",
                        help="Adapter version label (default: v1)")
    parser.add_argument("--lora-rank",  type=int,   default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS,
                        help=f"Training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,
                        help=f"Per-device batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume interrupted training "
                             "(e.g. models/adapters/fine_tuning-v1/checkpoints/checkpoint-150)")
    parser.add_argument("--seed",       type=int,   default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--preprocess", action="store_true",
                        help="Build the cleaned training corpus BEFORE training. Runs retrieval "
                             "(needs Qdrant + OpenSearch up) and loads the tokenizer. DEFAULT: OFF "
                             "— the trainer loads the existing artifact as-is and does NO "
                             "preprocessing, so it never competes with training for RAM/VRAM.")
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_LENGTH,
                        help="Token budget used when fitting rows during --preprocess "
                             "(must match training; default MAX_SEQ_LENGTH).")
    args = parser.parse_args()

    # Preprocessing is OFF by default. It runs ONLY with --preprocess; otherwise the trainer
    # loads the existing artifact (fit.jsonl) unchanged. No implicit/auto rebuild exists in
    # this path — this is deliberate so preprocessing never contends with fine-tuning for the
    # 8 GB GPU / RAM (Qdrant, OpenSearch, llama-server can all be shut down before training).
    if args.preprocess:
        run_preprocess(PreprocessConfig(
            force   = True,          # explicit request → always (re)build the artifact
            max_seq = args.max_seq,
        ))
        # Preprocessing just held the retriever stack (Qdrant/OpenSearch) + tokenizer in memory.
        # Pause so the user can free that memory before the GPU-heavy training begins.
        print(
            "\n...fit.jsonl has been generated successfully.\n"
            "Please shut down the Llama inference server, Qdrant, OpenSearch, and any other "
            "unnecessary processes to free maximum RAM and GPU memory.\n"
            "Continue with fine-tuning? (Y/N): ",
            end="",
        )
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("\nAborting before fine-tuning — no 'Y' received. "
                  "fit.jsonl is ready; re-run WITHOUT --preprocess to train.")
            raise SystemExit(0)

    _check_prerequisites()
    train(
        version               = args.version,
        lora_rank             = args.lora_rank,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        lr                    = args.lr,
        resume_from_checkpoint = args.resume_from_checkpoint,
        seed                  = args.seed,
    )
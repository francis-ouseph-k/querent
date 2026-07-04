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
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 2 * 8 = 16
# These values are shared between the legacy and Blackwell-safe paths.
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE      = 2e-4
NUM_EPOCHS         = 3
BATCH_SIZE         = 2      # per-device batch size
GRAD_ACCUM_STEPS   = 8      # effective batch = 16
MAX_SEQ_LENGTH     = 4096   # token ceiling per training example
                            # REVIEW FIX (#4): was 1024 — Phase 1 inference prompts run
                            # 2,000-4,000+ tokens once schema/workflow/glossary/join/
                            # few-shot context is included (budget ~10,200 effective
                            # tokens). 1024 silently truncated every training example,
                            # cutting off schema context or even the SQL output —
                            # a severe train/inference distribution mismatch.
                            # REVISED: 2048 still truncated the long tail — the same
                            # schema-context prompts that hit 2,000-4,000+ tokens at
                            # INFERENCE were being cut at 2,048 during TRAINING, so the
                            # model never saw the full context it must condition on at
                            # serve time. Raised to 4096 to cover the realistic upper
                            # band. This raises VRAM/step time; gradient_checkpointing is
                            # already enabled (SFTConfig) to offset it, and BATCH_SIZE is
                            # small (2). If OOM on 8 GB, drop BATCH_SIZE to 1 and raise
                            # GRAD_ACCUM_STEPS to 16 to hold the effective batch at 16.
                            # 4096 is a budget, not a verified ceiling for this GPU.
                            # Watch VRAM after the first few steps (nvidia-smi or
                            # torch.cuda.memory_summary()); reduce --batch-size to 1
                            # first if you OOM, since dropping below the inference-side
                            # prompt length reintroduces truncation. If your enriched
                            # corpus routinely exceeds 4096, cap schema_context in
                            # data_pipeline.py to this budget rather than raising it
                            # further — keeps the training and inference budgets aligned.
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
            "  Run:  python fine_tuning/data_pipeline.py"
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
    print(f"Applying LoRA (rank={lora_rank}, alpha={LORA_ALPHA})…")
    lora_config = LoraConfig(
        r                = lora_rank,
        lora_alpha       = LORA_ALPHA,
        lora_dropout     = LORA_DROPOUT,
        target_modules   = LORA_TARGET_MODULES,
        bias             = "none",
        task_type        = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
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
    dataset, dev_dataset = _split_train_dev(full_dataset, dev_fraction=0.10, seed=seed)
    print(f"Training on {len(dataset)} examples, {len(dev_dataset)} held out for eval_loss tracking…")

    # ── 6. Training arguments ─────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir                  = str(adapter_path / "checkpoints"),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
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
        load_best_model_at_end      = False,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
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
        # REVIEW FIX (#8): enable periodic eval_loss reporting against the
        # dev split held out above, so overfitting is visible during the run
        # rather than only after evaluator.py finishes post-hoc.
        eval_strategy                = "steps",
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

    trainer = SFTTrainer(
        model          = model,
        train_dataset  = dataset,
        eval_dataset   = dev_dataset,  # REVIEW FIX (#8)
        args           = training_args,
        tokenizer      = tokenizer,
        callbacks      = [EarlyStoppingCallback(early_stopping_patience=3)],
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
    args = parser.parse_args()

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
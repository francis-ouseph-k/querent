"""
config/model_profile.py
────────────────────────
SINGLE SOURCE OF TRUTH for the model ↔ prompt-profile contract.

WHY THIS MODULE EXISTS (FIX-R1b — centralised from batch_run.py):
The profile↔model coupling is a property of the DEPLOYMENT, not of any one
entry point. The original FIX-R1 guard lived inline in batch_run.run_batch(),
which left main.py (interactive/production serving) with ZERO protection — it
would happily serve a fine-tuned GGUF with the `full` profile, silently
reproducing the v5/v6 out-of-distribution failure in production, which is
worse than in a benchmark. This module states the rule ONCE; every entry
point calls it at startup, right after probing the served model's identity.

THE CONTRACT:
  fine-tuned GGUF  →  LLM_PROMPT_PROFILE=ft    (training-parity prompt)
  base GGUF        →  LLM_PROMPT_PROFILE=full  (rich serve prompt)
Anything else measures nothing you can act on (benchmark) or silently
degrades accuracy below the base model (production).

CONTROL MODEL — validate, do NOT derive:
The profile is controlled SOLELY by the environment flag LLM_PROMPT_PROFILE
(settings.llm.prompt_profile). The served model's identity is used only to
VALIDATE the coupling, never to choose the profile. Rationale: the
fine-tuned-ness detection below is a filename heuristic
(`finetuned|adapter|merged`) — good enough to catch mistakes loudly, too
brittle to silently drive behaviour (a fine-tuned model exported as
`qwen-custom.gguf` would be misclassified). When export-time GGUF metadata
stamping lands (R5), detection can switch to reading the stamp and
derive-by-default becomes safe to revisit.

Public API:
  probe_model_identity()                      → str   (llama-server /props or LLM_MODEL_PATH)
  looks_finetuned(model_id)                   → bool  (filename heuristic — single definition)
  validate_profile(model_id, profile, ...)    → str   (raises ProfileMismatchError on violation)
  resolve_profile(model_id=None, ...)         → tuple[str, str]  (probe + read env + validate)
"""

from __future__ import annotations

import re
from pathlib import Path

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Filename heuristic for "this GGUF is a fine-tune". Matches the export
# pipeline's naming convention (export.py writes *finetuned-vN* GGUFs;
# adapters/merged dirs carry those words). Keep the convention or upgrade
# to GGUF metadata stamping (R5) — do NOT weaken this regex.
_FINETUNED_NAME_RE = re.compile(r"finetuned|adapter|merged", re.IGNORECASE)

VALID_PROFILES = ("full", "ft")


class ProfileMismatchError(RuntimeError):
    """Served model and LLM_PROMPT_PROFILE violate the training-parity contract."""


def probe_model_identity() -> str:
    """
    Identify the model actually serving this process (FIX-F6 provenance).

    HTTP mode: GET llama-server /props → model file path (+ file size when the
    path is locally resolvable — a cheap integrity proxy; SHA256 of a 2.4 GB
    GGUF at every start is not worth the wall-clock).
    In-process mode: LLM_MODEL_PATH from settings.
    Never raises — neither a benchmark nor a production CLI should die on a
    telemetry probe.
    """
    # PROVIDER SWITCH: a hosted provider (Mistral / Gemini) has no GGUF to
    # fingerprint and no llama-server /props endpoint. Its identity is simply
    # provider:model, which is stable, secret-free, and enough for provenance.
    if not settings.llm.is_local_provider:
        from generation.llm import describe_active_llm
        info = describe_active_llm()
        return f"{settings.llm.provider}:{info.model_id}"

    base_url = settings.llm.base_url
    if base_url:
        try:
            import httpx
            root = base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url
            r = httpx.get(f"{root}/props", timeout=5.0)
            r.raise_for_status()
            props = r.json()
            path = (props.get("model_path")
                    or props.get("default_generation_settings", {}).get("model")
                    or props.get("model", ""))
            if path:
                try:
                    size = Path(path).stat().st_size
                    return f"{path} ({size:,} bytes)"
                except OSError:
                    return str(path)
            return f"llama-server@{root} (path not reported)"
        except Exception as exc:
            return f"llama-server@{base_url} (probe failed: {exc})"
    p = Path(settings.llm.model_path)
    try:
        return f"{p} ({p.stat().st_size:,} bytes, in-process)"
    except OSError:
        return f"{p} (in-process, size unavailable)"


def looks_finetuned(model_id: str) -> bool:
    """Filename heuristic — the ONE definition every caller shares."""
    return bool(_FINETUNED_NAME_RE.search(str(model_id)))


def validate_profile(
    model_id: str,
    profile: str,
    *,
    allow_mismatch: bool = False,
) -> str:
    """
    Enforce the model↔profile contract. Returns the validated profile.

    Raises ProfileMismatchError on violation unless allow_mismatch=True
    (deliberate OOD experiments only — the result is NOT a valid A/B
    measurement and NOT a supported production configuration).
    """
    if profile not in VALID_PROFILES:
        raise ProfileMismatchError(
            f"Unknown LLM_PROMPT_PROFILE={profile!r}. "
            f"Valid values: {', '.join(VALID_PROFILES)}."
        )

    # PROVIDER SWITCH: the contract below is about OUR LoRA adapter — it only
    # has meaning for the locally served GGUF. A hosted model is never a
    # fine-tune of our corpus, so the filename heuristic would be nonsense
    # applied to "gemini-2.0-flash". LLMSettings.normalise_provider() has
    # already pinned prompt_profile to "full" for hosted providers; here we
    # simply skip the guard rather than fail a legitimate configuration.
    if not settings.llm.is_local_provider:
        return profile

    finetuned = looks_finetuned(model_id)

    if allow_mismatch:
        if (finetuned and profile != "ft") or (not finetuned and profile == "ft"):
            logger.warning(
                component="model_profile",
                event="profile_mismatch_overridden",
                model=model_id,
                profile=profile,
                note="allow_mismatch=True — deliberate OOD run; "
                     "NOT a valid A/B measurement.",
            )
        return profile

    if finetuned and profile != "ft":
        raise ProfileMismatchError(
            f"The served model looks fine-tuned ({model_id}) but "
            f"LLM_PROMPT_PROFILE={profile!r}. Fine-tuned models MUST use the "
            f"training-parity profile: set LLM_PROMPT_PROFILE=ft in .env "
            f"(or pass allow_mismatch for a deliberate OOD experiment — the "
            f"result will NOT be a valid A/B measurement)."
        )
    if not finetuned and profile == "ft":
        raise ProfileMismatchError(
            f"The served model looks like a BASE model ({model_id}) but "
            f"LLM_PROMPT_PROFILE=ft. Base models MUST use "
            f"LLM_PROMPT_PROFILE=full (or pass allow_mismatch)."
        )
    return profile


def resolve_profile(
    model_id: str | None = None,
    *,
    allow_mismatch: bool = False,
) -> tuple[str, str]:
    """
    One-call startup helper for entry points.

    Probes the served model (unless model_id is given), reads the profile
    from the environment flag (settings.llm.prompt_profile — the SOLE
    control), validates the coupling, logs provenance, and returns
    (model_id, profile).
    """
    if model_id is None:
        model_id = probe_model_identity()
    profile = validate_profile(
        model_id, settings.llm.prompt_profile, allow_mismatch=allow_mismatch
    )
    logger.info(
        component="model_profile",
        event="model_provenance",
        model=model_id,
        prompt_profile=profile,
        temperature=settings.llm.temperature,
    )
    return model_id, profile
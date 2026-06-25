"""
utils/tokenizer.py
──────────────────
Shared offline Qwen tokenizer loader for accurate token counting.
"""

from pathlib import Path
from transformers import AutoTokenizer
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    from config.settings import settings
    
    # Check custom HF cache path from settings
    hf_home = settings.hf_home
    if hf_home:
        snapshots_dir = Path(hf_home) / "hub" / "models--Qwen--Qwen2.5-Coder-3B-Instruct" / "snapshots"
        if snapshots_dir.exists():
            subdirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if subdirs:
                try:
                    _TOKENIZER = AutoTokenizer.from_pretrained(str(subdirs[0]))
                    logger.info(component="tokenizer", event="loaded_from_snapshot", path=str(subdirs[0]))
                    return _TOKENIZER
                except Exception as e:
                    logger.warning(
                        component="tokenizer",
                        event="failed_to_load_from_snapshot",
                        path=str(subdirs[0]),
                        error=str(e),
                    )

    try:
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
        logger.info(component="tokenizer", event="loaded_via_local_files_only")
        return _TOKENIZER
    except Exception as e:
        logger.warning(component="tokenizer", event="local_files_only_failed", error=str(e))

    # Fallback to tiktoken
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        logger.info(component="tokenizer", event="loaded_fallback_tiktoken")
        return _TOKENIZER
    except Exception:
        logger.warning(component="tokenizer", event="tiktoken_fallback_failed")
        _TOKENIZER = None
        return _TOKENIZER


def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    if tokenizer is not None:
        try:
            # Check if it's a transformers tokenizer or tiktoken encoder
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text) // 4

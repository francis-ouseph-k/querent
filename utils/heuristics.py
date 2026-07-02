"""
utils/heuristics.py
───────────────────
HEURISTICS — the tunable knobs for the semantic checks, loaded once at import
from config/heuristics.yaml.

Keeps thresholds, keyword lists, and synonym maps used by the SemanticValidator
out of the code and in a config file the team can edit without touching logic. On
a load failure it degrades to an empty dict rather than crashing the pipeline, so
a malformed YAML disables the tuned checks instead of taking the system down.
"""

import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_HEURISTICS_PATH = Path(__file__).parents[1] / 'config' / 'heuristics.yaml'

try:
    with _HEURISTICS_PATH.open('r', encoding='utf-8') as f:
        HEURISTICS = yaml.safe_load(f) or {}
except Exception as e:
    logger.error(f"Failed to load heuristics from {_HEURISTICS_PATH}: {e}")
    HEURISTICS = {}


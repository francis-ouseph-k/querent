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


# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess/manifest.py
==================================
Dataset fingerprint + freshness check.

Every generated training artifact (`<train_data>`) gets a sibling manifest
(`<train_data>.manifest.json`) recording exactly how it was produced. The
trainer uses `is_fresh()` to decide whether the cached artifact can be reused
or must be rebuilt — so preprocessing runs *only* when the source or the
preprocessing configuration actually changed. Months later the manifest tells
you which corpus + config trained a given adapter.

The freshness contract is a pair of hashes:
  source_hash  = sha256 of the curated source file (+ benchmark file, since
                 de-leaking depends on it)
  config_hash  = sha256 of the preprocessing knobs that change the output
                 (jaccard, max_seq, ddl path, generator version, …)

If both match the manifest, the artifact is fresh. Anything else → stale.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bump when the preprocessing LOGIC changes in a way that alters output even
# though inputs/config are unchanged (e.g. a new gate rule). This forces a
# rebuild for everyone without them having to touch config.
GENERATOR_VERSION = "3.0.0"   # 3.0.0: gold-table pinning + section-aware fit with gold-ctx hard gate
                              #        + placeholder substitution (strip is fallback) + reasoning scrub
                              #        + off-task category whitelist. Shared renderer with
                              #        prompt_builder (train/serve parity via build_ft / LLM_PROMPT_PROFILE=ft).
                              # 2.1.0: training uses short _TRAIN_SYSTEM_PROMPT (was full serve prompt)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(source_path: Path, benchmark_path: Path) -> str:
    """Fingerprint the inputs whose CONTENT changes the corpus."""
    parts = [_sha256_file(source_path)]
    if benchmark_path.exists():
        parts.append(_sha256_file(benchmark_path))
    return _sha256_obj(parts)


def config_hash(cfg: dict[str, Any]) -> str:
    """Fingerprint the preprocessing knobs + generator version."""
    return _sha256_obj({"generator_version": GENERATOR_VERSION, **cfg})


@dataclass
class Manifest:
    generator_version: str
    source_hash: str
    config_hash: str
    config: dict[str, Any]
    row_counts: dict[str, int]                 # {input, after_deleak, after_gate, final}
    rejected: dict[str, int]                   # by reason, e.g. {placeholder_reject: 0, garbage_join: 10, ...}
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def path_for(artifact: Path) -> Path:
        return artifact.with_suffix(artifact.suffix + ".manifest.json")

    def write(self, artifact: Path) -> Path:
        p = self.path_for(artifact)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(p)
        return p

    @staticmethod
    def read(artifact: Path) -> "Manifest | None":
        p = Manifest.path_for(artifact)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return Manifest(**d)
        except Exception:
            return None  # unreadable/old-format manifest → treat as stale


def is_fresh(artifact: Path, want_source_hash: str, want_config_hash: str) -> bool:
    """True iff the artifact exists AND its manifest matches both hashes."""
    if not artifact.exists():
        return False
    m = Manifest.read(artifact)
    if m is None:
        return False
    return (m.source_hash == want_source_hash
            and m.config_hash == want_config_hash
            and m.generator_version == GENERATOR_VERSION)
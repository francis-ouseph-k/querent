# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess/pipeline.py
==================================
Orchestrator. Chains the stages and owns ALL file I/O + freshness caching:

    sources.load_curated
        → quality.deleak          (drop benchmark leaks)
        → quality.gate            (strip placeholders / reject garbage+bad rows)
        → build.format_pairs      (retrieval → ChatML; train/serve parity)
        → build.wrap_rows         (JSON serve envelope)
        → build.fit_rows          (token-budget to max_seq)
        → write <artifact> + <artifact>.manifest.json

Public API
----------
  run(cfg)          — force a full rebuild, write artifact + manifest, return path.
  ensure_fresh(cfg) — the trainer hook. Rebuild ONLY if the artifact is stale
                      (source or config hash changed); otherwise reuse the cache.
                      cfg.no_preprocess=True turns "stale" into a hard error
                      (CI / strict reproducibility) instead of an auto-rebuild.

Because build.format_pairs stands up the live retriever (Qdrant + OpenSearch +
FK graph) and runs it per row, a rebuild is expensive — which is exactly why the
manifest freshness check exists: unchanged inputs ⇒ zero retrieval, instant reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logging_config import get_logger

from . import build, manifest, quality, sources

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    # inputs
    curated_source: Path = field(default_factory=lambda: Path("data/training/nl-sql_traning-dataset.jsonl"))
    benchmark:      Path = field(default_factory=lambda: Path("data/inputs/benchmark-test-set.jsonl"))
    ddl_path:       Path = field(default_factory=lambda: Path(settings.ddl_path))
    model_dir:      str  = field(default_factory=lambda: settings.fine_tuning.hf_model_dir)
    # output artifact (what trainer reads)
    artifact:       Path = field(default_factory=lambda: Path(settings.fine_tuning.train_data))
    # knobs
    jaccard:        float = 0.85
    max_seq:        int   = 2048
    skip_retrieval: bool  = False        # throwaway smoke test only — degrades data
    # freshness behaviour
    force:          bool  = False        # rebuild even if fresh
    no_preprocess:  bool  = False        # stale ⇒ raise instead of rebuild (CI)

    def config_fingerprint(self) -> dict[str, Any]:
        """The knobs that change the OUTPUT — hashed into the manifest config_hash."""
        return {
            "jaccard": self.jaccard,
            "max_seq": self.max_seq,
            "ddl_path": str(self.ddl_path),
            "curated_source": str(self.curated_source),
            "benchmark": str(self.benchmark),
            "skip_retrieval": self.skip_retrieval,
        }


def _hashes(cfg: PreprocessConfig) -> tuple[str, str]:
    sh = manifest.source_hash(cfg.curated_source, cfg.benchmark)
    ch = manifest.config_hash(cfg.config_fingerprint())
    return sh, ch


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json, os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def run(cfg: PreprocessConfig) -> Path:
    """Full rebuild → artifact + manifest. Returns the artifact path."""
    for p in (cfg.curated_source, cfg.benchmark):
        if not p.exists():
            raise SystemExit(f"ERROR: preprocess input not found: {p}")

    # 1. load
    pairs = sources.load_curated(cfg.curated_source)
    n_in = len(pairs)

    # 2. de-leak (train/test hygiene)
    benchq = sources.load_benchmark_questions(cfg.benchmark)
    pairs, removed = quality.deleak(pairs, benchq, cfg.jaccard)
    n_deleak = len(pairs)

    # 3. quality gate (placeholder strip / garbage+bad reject)
    pairs, rej = quality.gate(pairs)
    n_gate = len(pairs)

    # 4. format (retrieval → ChatML) — the expensive stage
    if cfg.skip_retrieval:
        logger.warning(component="preprocess.pipeline", event="skip_retrieval",
                       note="EMPTY schema context — train/serve mismatch; smoke test only")
        retriever = qu = None
    else:
        retriever, qu = build.init_retriever()
    rows, empty_ctx = build.format_pairs(pairs, retriever, qu)

    # 5. wrap (JSON serve envelope) — backstop drop of any non-SQL survivors
    rows, wrap_dropped = build.wrap_rows(rows, cfg.ddl_path)

    # 6. fit (token budget)
    rows, trimmed, still_over = build.fit_rows(rows, cfg.model_dir, cfg.max_seq)
    n_final = len(rows)

    # 7. write artifact + manifest
    _write_jsonl(cfg.artifact, rows)
    sh, ch = _hashes(cfg)
    reasons = {k: v for k, v in rej.items() if k != "_stripped"}
    reasons.update({"wrap_dropped_non_sql": wrap_dropped})
    man = manifest.Manifest(
        generator_version=manifest.GENERATOR_VERSION,
        source_hash=sh, config_hash=ch,
        config=cfg.config_fingerprint(),
        row_counts={
            "input": n_in, "after_deleak": n_deleak, "after_gate": n_gate,
            "final": n_final, "placeholder_stripped": rej.get("_stripped", 0),
            "leaks_removed": len(removed), "schema_trimmed": trimmed,
            "empty_context": len(empty_ctx), "still_over_max_seq": still_over,
        },
        rejected=reasons,
    )
    man_path = man.write(cfg.artifact)

    logger.info(component="preprocess.pipeline", event="rebuild_complete",
                artifact=str(cfg.artifact), manifest=str(man_path),
                final_rows=n_final, rejected=reasons)
    print(f"\n[preprocess] {n_in} → {n_final} rows "
          f"(deleak-removed {len(removed)}, gate-rejected {n_gate and n_deleak - n_gate}, "
          f"placeholder-stripped {rej.get('_stripped', 0)})")
    print(f"[preprocess] artifact: {cfg.artifact}")
    print(f"[preprocess] manifest: {man_path}")
    if still_over:
        print(f"[preprocess] WARNING: {still_over} rows still exceed max_seq={cfg.max_seq}")
    return cfg.artifact


def ensure_fresh(cfg: PreprocessConfig) -> Path:
    """
    Trainer hook. Rebuild only if stale; otherwise reuse the cached artifact.
      cfg.force         → always rebuild
      cfg.no_preprocess → stale is a hard error (never auto-rebuild)
    """
    sh, ch = _hashes(cfg)
    fresh = manifest.is_fresh(cfg.artifact, sh, ch)

    if fresh and not cfg.force:
        logger.info(component="preprocess.pipeline", event="cache_hit",
                    artifact=str(cfg.artifact))
        print(f"[preprocess] cache fresh → reusing {cfg.artifact}")
        return cfg.artifact

    if cfg.no_preprocess:
        raise SystemExit(
            f"ERROR: training artifact is STALE and --no-preprocess is set.\n"
            f"  artifact: {cfg.artifact}\n"
            f"  Run preprocessing first:  python -m fine_tuning.preprocess\n"
        )

    reason = "forced" if cfg.force else ("missing" if not cfg.artifact.exists() else "stale")
    logger.info(component="preprocess.pipeline", event="rebuild_triggered", reason=reason)
    print(f"[preprocess] {reason} → rebuilding corpus (retrieval will run)…")
    return run(cfg)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Clean the corpus → training artifact + manifest.")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--benchmark", type=Path)
    ap.add_argument("--out", type=Path, help="Artifact path (default: settings.fine_tuning.train_data)")
    ap.add_argument("--ddl", type=Path)
    ap.add_argument("--model", type=str, help="Qwen snapshot dir for tokenisation")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--jaccard", type=float, default=0.85)
    ap.add_argument("--force", action="store_true", help="Rebuild even if fresh")
    ap.add_argument("--skip-retrieval", action="store_true", help="Smoke test only — degrades data")
    args = ap.parse_args()

    cfg = PreprocessConfig(force=True, max_seq=args.max_seq, jaccard=args.jaccard,
                           skip_retrieval=args.skip_retrieval)
    if args.source:    cfg.curated_source = args.source
    if args.benchmark: cfg.benchmark = args.benchmark
    if args.out:       cfg.artifact = args.out
    if args.ddl:       cfg.ddl_path = args.ddl
    if args.model:     cfg.model_dir = args.model
    run(cfg)


if __name__ == "__main__":
    main()
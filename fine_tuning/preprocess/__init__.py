# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess
======================
Corpus cleaning + training-artifact build, run automatically before the trainer.

    from fine_tuning.preprocess import ensure_fresh, PreprocessConfig
    ensure_fresh(PreprocessConfig())      # rebuild iff stale, else reuse cache

Flow:  sources -> quality (deleak + gate) -> build (format -> wrap -> fit)
       -> <artifact> + <artifact>.manifest.json

Replaces the manual chain deleak_train -> build_train_from_curated ->
wrap_outputs_json -> fit_context, and dissolves data_pipeline.py.
"""
from .pipeline import PreprocessConfig, ensure_fresh, run

__all__ = ["PreprocessConfig", "ensure_fresh", "run"]
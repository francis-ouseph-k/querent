"""
tests/test_retrieval_orchestrator.py
────────────────────────────────────────
retrieval/orchestrator.py::_fetch_mandatory_chunks_for — mandatory schema
chunks are fetched CONCURRENTLY across tables (not sequentially), and one
failed fetch must not lose the results of the others.

CONSOLIDATED FROM: test_security_hardening.py ("A1: retrieval
parallelization"). Extracts and execs the exact shipped method body (located
dynamically by marker, not by hardcoded line numbers) rather than importing
the module directly, since the module's real import chain
(sentence-transformers, qdrant-client) is not needed to prove this method's
concurrency behaviour is correct.
"""

from __future__ import annotations

import time as _time


def test_a1_mandatory_chunk_fetch_is_concurrent_and_correct():
    import enum
    import textwrap
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src_path = "retrieval/orchestrator.py"
    full = open(src_path, encoding="utf-8").read()
    marker = "    def _fetch_mandatory_chunks_for("
    start = full.index(marker)
    after = full[start + len(marker):]
    end_rel = after.find("\n# " + "─" * 10)
    if end_rel == -1:
        end_rel = after.find("\n    def ", 200)
    method_src = (marker + after[:end_rel]).strip("\n")
    method_src = textwrap.dedent(method_src)

    class CT(enum.Enum):
        TABLE = "table"
        FK_MAP = "fk_map"

    class SemanticChunk:
        @staticmethod
        def from_payload(p):
            return p

    ns = {
        "ChunkType": CT, "SemanticChunk": SemanticChunk, "time": _time,
        "ThreadPoolExecutor": ThreadPoolExecutor, "as_completed": as_completed,
        "Any": object,
        "logger": type("L", (), {"warning": staticmethod(lambda **k: None)})(),
    }
    exec(method_src, ns)
    fn = ns["_fetch_mandatory_chunks_for"]

    class FakeQdrant:
        def search(self, query_text, top_k, chunk_types, filter_payload):
            _time.sleep(0.3)
            return [{"chunk_id": f"{filter_payload['table_name']}:{chunk_types[0].value}",
                     "table_name": filter_payload["table_name"],
                     "chunk_type": chunk_types[0].value}]

    class Fake:
        qdrant = FakeQdrant()

    tables = ["t1", "t2", "t3", "t4", "t5"]
    t0 = _time.time()
    chunks, misses = fn(Fake(), tables, {})
    elapsed = _time.time() - t0
    # Sequential would be 2 calls x 5 tables x 0.3s = 3.0s.
    assert elapsed < 1.5, f"not parallelized: {elapsed:.2f}s"
    assert len(chunks) == 10 and misses == 10

    class FlakyQdrant:
        n = 0
        def search(self, **kw):
            FlakyQdrant.n += 1
            if FlakyQdrant.n == 3:
                raise RuntimeError("boom")
            return [{"chunk_id": "x", "table_name": kw["filter_payload"]["table_name"],
                     "chunk_type": kw["chunk_types"][0].value}]

    class Fake2:
        qdrant = FlakyQdrant()

    c2, m2 = fn(Fake2(), ["a", "b"], {})
    assert len(c2) == 3, "one failed fetch must not lose the other three"

    c3, m3 = fn(Fake(), [], {})
    assert c3 == [] and m3 == 0

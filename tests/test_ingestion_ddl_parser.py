"""
tests/test_ingestion_ddl_parser.py
──────────────────────────────────────
ingestion/ddl_parser.py — parser robustness (comment-only chunks, zero
parse errors against the real DDL), and consistency between the DDL version
in use and the artefacts that declare which version they target.

CONSOLIDATED FROM: test_security_hardening.py (A5: comment-only chunk
detection, real DDL zero parse errors), test_run6_hardening.py (seed
vocabulary extraction feeding array-column CHECK-equivalents, real DDL
still parses to 63 tables), and test_llm_provider_switch.py
(TestSchemaVersionUpgrade — misfiled under "provider switch" originally,
but it has nothing to do with LLM providers: it is entirely about the DDL
version and config/derived_fks.yaml staying in sync).
"""

from __future__ import annotations

from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# Parser robustness (FIX-A5)
# ═════════════════════════════════════════════════════════════════════════════

def test_a5_comment_only_chunk_detection():
    from ingestion.ddl_parser import _is_comment_only

    assert _is_comment_only("-- just a comment\n-- another line\n")
    assert _is_comment_only("/* block */\n-- line\n   \n")
    assert _is_comment_only("")
    assert _is_comment_only("   \n\n  ")
    assert not _is_comment_only("-- comment\nCREATE TABLE x (id int);")
    assert not _is_comment_only("SELECT 1;")


def test_a5_real_ddl_has_zero_parse_errors():
    from ingestion.ddl_parser import DDLParser

    ddl_path = "data/docs/digital_evaluation_schema_v10_10.sql"
    ddl = open(ddl_path, encoding="utf-8").read()
    parser = DDLParser()
    tables = parser.parse(ddl)
    assert parser.parse_errors == [], (
        "SECTION 15 (and any other trailing comment-only block) must not "
        "surface as a parse error"
    )
    assert len(tables) == 63  # 61 tables + 2 views, unchanged from every prior run


def test_real_ddl_still_parses_cleanly(real_schema):
    """The seed-vocabulary pass must not disturb existing parsing."""
    assert len(real_schema) == 63


def test_seed_vocabulary_is_extracted_for_array_columns(real_schema):
    """
    Postgres cannot express "every element of this array is one of N values" as
    a CHECK, so an array column's vocabulary lives only in the seed rows. The
    expected values are read from the DDL, not hardcoded here.
    """
    col = real_schema["workflow_state_transition"].columns["allowed_roles"]
    assert col.allowed_values is None          # no CHECK exists
    assert col.observed_values                 # but seed vocabulary was found
    # Whatever the seed says, an attempt_type value must not be in a role column.
    assert "PRIMARY" not in col.observed_values


# ═════════════════════════════════════════════════════════════════════════════
# Schema version consistency — v10.5 → v10.10 upgrade
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaVersionUpgrade:

    ROOT = Path(__file__).resolve().parent.parent
    DDL = ROOT / "data/docs/digital_evaluation_schema_v10_10.sql"

    def test_default_ddl_path_points_at_v10_10(self):
        from config.settings import Settings
        assert "v10_10" in Settings(_env_file=None).ddl_path

    def test_v10_10_ddl_file_exists(self):
        assert self.DDL.exists(), f"missing {self.DDL}"

    def test_derived_fks_declares_the_new_schema_version(self):
        import yaml
        data = yaml.safe_load((self.ROOT / "config/derived_fks.yaml").read_text(encoding="utf-8"))
        assert str(data["schema_version"]) == "10.10"

    def test_derived_fk_edges_still_exist_in_v10_10(self):
        """
        The five derived (comment-inferred) FK edges are not enforced by the DDL,
        so nothing else would catch it if v10.10 had renamed one of their
        columns. Assert every source/target column is still present.
        """
        import yaml
        from ingestion.ddl_parser import DDLParser

        tables = DDLParser().parse_file(self.DDL)
        data = yaml.safe_load((self.ROOT / "config/derived_fks.yaml").read_text(encoding="utf-8"))

        def col_names(table):
            return {c if isinstance(c, str) else c.name for c in tables[table].columns}

        for edge in data["derived_fks"]:
            src, tgt = edge["source_table"], edge["target_table"]
            assert src in tables, f"derived FK source table {src} missing from v10.10"
            assert tgt in tables, f"derived FK target table {tgt} missing from v10.10"
            for mapping in edge["column_mappings"]:
                assert mapping["source_column"] in col_names(src), (
                    f"{src}.{mapping['source_column']} missing from v10.10"
                )
                assert mapping["target_column"] in col_names(tgt), (
                    f"{tgt}.{mapping['target_column']} missing from v10.10"
                )

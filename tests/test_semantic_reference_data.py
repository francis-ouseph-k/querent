"""
tests/test_semantic_reference_data.py
─────────────────────────────────────────
validation/semantic/reference_data.py::ReferenceDataValidator and its
supporting validation/utils/seed_index.py::build_seed_index — rejects a query
whose literals contradict a fact the DDL's own seed rows already state (e.g.
a relationship-type edge asserted in the reverse of what its seed row says).

CONSOLIDATED FROM: test_run8_correctness.py ("2. reference data"). seed_index
tests are kept in this file rather than a separate one because
build_seed_index exists only to serve ReferenceDataValidator.
"""

from __future__ import annotations

from models.schema import ColumnInfo, ForeignKey, TableInventory
from validation.semantic.reference_data import ReferenceDataValidator
from validation.utils.seed_index import build_seed_index

from conftest import make_ctx


def _col(name, data_type="bigint", *, pk=False, nullable=True):
    return ColumnInfo(name=name, data_type=data_type, nullable=nullable, is_pk=pk)


def _table(name, columns, fks=()):
    return TableInventory(
        table_name=name, columns={c.name: c for c in columns}, foreign_keys=list(fks),
    )


def schema():
    return {
        "unit_type_config": _table("unit_type_config", [
            _col("unit_type", "varchar", pk=True),
        ]),
        "academic_unit": _table(
            "academic_unit",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("unit_type", "varchar", nullable=False),
            ],
            fks=[ForeignKey("academic_unit", "unit_type", "unit_type_config", "unit_type")],
        ),
        "relationship_type_config": _table(
            "relationship_type_config",
            [
                _col("relationship_type", "varchar", pk=True),
                _col("from_unit_type", "varchar", nullable=False),
                _col("to_unit_type", "varchar", nullable=False),
            ],
            fks=[
                ForeignKey("relationship_type_config", "from_unit_type",
                           "unit_type_config", "unit_type"),
                ForeignKey("relationship_type_config", "to_unit_type",
                           "unit_type_config", "unit_type"),
            ],
        ),
        "academic_unit_relationship": _table(
            "academic_unit_relationship",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("from_unit_id", "bigint", nullable=False),
                _col("to_unit_id", "bigint", nullable=False),
                _col("relationship_type", "varchar", nullable=False),
                _col("is_active", "boolean", nullable=False),
            ],
            fks=[
                ForeignKey("academic_unit_relationship", "from_unit_id",
                           "academic_unit", "id"),
                ForeignKey("academic_unit_relationship", "to_unit_id",
                           "academic_unit", "id"),
                ForeignKey("academic_unit_relationship", "relationship_type",
                           "relationship_type_config", "relationship_type"),
            ],
        ),
    }


SEED_DDL = [
    """INSERT INTO unit_type_config (unit_type, display_name) VALUES
        ('CAMPUS','Campus'),('DEPARTMENT','Department'),
        ('PROGRAM','Program'),('COURSE','Course');""",
    """INSERT INTO relationship_type_config
        (relationship_type, display_name, from_unit_type, to_unit_type) VALUES
        ('PROGRAM_COURSE','Program offers Course','PROGRAM','COURSE'),
        ('CROSS_LISTING','Course cross-listed in Dept','COURSE','DEPARTMENT'),
        ('PREREQUISITE','Course prerequisite','COURSE','COURSE');""",
]


def ctx_for(sql, schema_map=None):
    return make_ctx(sql, schema_map if schema_map is not None else schema())


def test_seed_index_captures_whole_rows():
    index = build_seed_index(SEED_DDL)
    rows = index["relationship_type_config"]
    by_key = {r["relationship_type"]: r for r in rows}
    assert by_key["CROSS_LISTING"]["from_unit_type"] == "COURSE"
    assert by_key["CROSS_LISTING"]["to_unit_type"] == "DEPARTMENT"
    assert by_key["PROGRAM_COURSE"]["from_unit_type"] == "PROGRAM"


def _refdata():
    return ReferenceDataValidator(seed_index=build_seed_index(SEED_DDL))


def test_reversed_cross_listing_direction_is_rejected():
    """Q176: CROSS_LISTING runs COURSE -> DEPARTMENT, not the reverse."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit dept ON dept.id = aur.from_unit_id
                               AND dept.unit_type = 'DEPARTMENT'
        JOIN academic_unit course ON course.id = aur.to_unit_id
                                 AND course.unit_type = 'COURSE'
        WHERE aur.relationship_type = 'CROSS_LISTING'
    """
    result = _refdata().run(ctx_for(sql))
    assert not result.passed
    assert "COURSE" in result.message


def test_reversed_program_course_direction_is_rejected():
    """Q177: PROGRAM_COURSE puts PROGRAM on the from-side."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit au_to ON au_to.id = aur.to_unit_id
                                AND au_to.unit_type = 'PROGRAM'
        WHERE aur.relationship_type = 'PROGRAM_COURSE'
    """
    assert not _refdata().run(ctx_for(sql)).passed


def test_correct_direction_passes():
    """Q98's join direction is right; only its nullability is wrong."""
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit au_prog ON au_prog.id = aur.from_unit_id
                                  AND au_prog.unit_type = 'PROGRAM'
        JOIN academic_unit au_course ON au_course.id = aur.to_unit_id
                                    AND au_course.unit_type = 'COURSE'
        WHERE aur.relationship_type = 'PROGRAM_COURSE'
    """
    assert _refdata().run(ctx_for(sql)).passed


def test_unpinned_relationship_type_is_silent():
    sql = """
        SELECT aur.relationship_type, COUNT(*)
        FROM academic_unit_relationship aur
        WHERE aur.is_active = TRUE
        GROUP BY aur.relationship_type
    """
    assert _refdata().run(ctx_for(sql)).passed


def test_validator_without_seeds_is_inert():
    sql = """
        SELECT aur.id
        FROM academic_unit_relationship aur
        JOIN academic_unit dept ON dept.id = aur.from_unit_id
                               AND dept.unit_type = 'DEPARTMENT'
        WHERE aur.relationship_type = 'CROSS_LISTING'
    """
    assert ReferenceDataValidator().run(ctx_for(sql)).passed

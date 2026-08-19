"""
tests/test_ast_joins.py
─────────────────────────
validation/ast/joins.py::JoinValidator — table-level join-domain mismatch
(two FK columns whose target tables differ) and role-aware domain mismatch
(two FK columns targeting the SAME polymorphic table but declaring different
discriminator roles, e.g. a COURSE key equated with a DEPARTMENT key on
academic_unit).

CONSOLIDATED FROM: test_run8_correctness.py ("1. role-aware join domains").
"""

from __future__ import annotations

from models.schema import ColumnInfo, ForeignKey, TableInventory
from validation.ast.joins import JoinValidator

from conftest import make_ctx


def _col(name, data_type="bigint", *, pk=False, nullable=True, comment=""):
    return ColumnInfo(
        name=name, data_type=data_type, nullable=nullable, is_pk=pk, comment=comment,
    )


def _table(name, columns, fks=(), comments=None):
    return TableInventory(
        table_name=name,
        columns={c.name: c for c in columns},
        foreign_keys=list(fks),
        column_comments=dict(comments or {}),
    )


def schema():
    """A miniature of the polymorphic-hierarchy shape under test."""
    return {
        "unit_type_config": _table("unit_type_config", [
            _col("unit_type", "varchar", pk=True),
            _col("display_name", "varchar"),
        ]),
        "academic_unit": _table(
            "academic_unit",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("parent_id", "bigint"),
                _col("unit_type", "varchar", nullable=False),
                _col("code", "varchar"),
                _col("name", "varchar"),
            ],
            fks=[
                ForeignKey("academic_unit", "unit_type", "unit_type_config", "unit_type"),
                ForeignKey("academic_unit", "parent_id", "academic_unit", "id"),
            ],
        ),
        "board": _table(
            "board",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("course_id", "bigint", nullable=False),
                _col("status", "varchar"),
            ],
            fks=[ForeignKey("board", "course_id", "academic_unit", "id")],
            comments={"course_id": "FK to academic_unit where unit_type=COURSE."},
        ),
        "faculty_cache": _table(
            "faculty_cache",
            [
                _col("id", "bigint", pk=True, nullable=False),
                _col("department_id", "bigint"),
                _col("name", "varchar"),
            ],
            fks=[ForeignKey("faculty_cache", "department_id", "academic_unit", "id")],
            comments={"department_id": "FK to academic_unit where unit_type=DEPARTMENT."},
        ),
    }


def ctx_for(sql, schema_map=None):
    return make_ctx(sql, schema_map if schema_map is not None else schema())


def test_course_id_joined_to_department_id_is_rejected():
    """Q43: both sides are academic_unit keys, but of different kinds."""
    sql = """
        WITH department_courses AS (
            SELECT d.id AS dept_id, d.name AS dept_name
            FROM academic_unit d
            WHERE d.unit_type = 'DEPARTMENT'
        ),
        active_boards AS (
            SELECT b.course_id, COUNT(*) AS num_boards
            FROM board b GROUP BY b.course_id
        )
        SELECT dc.dept_name, ab.num_boards
        FROM department_courses dc
        LEFT JOIN active_boards ab ON ab.course_id = dc.dept_id
    """
    result = JoinValidator(fk_graph=None).run(ctx_for(sql))
    assert not result.passed
    assert "role mismatch" in result.message.lower()
    assert "COURSE" in result.message and "DEPARTMENT" in result.message


def test_matching_roles_pass():
    sql = """
        SELECT au.code
        FROM board b
        JOIN academic_unit au ON au.id = b.course_id AND au.unit_type = 'COURSE'
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_parent_child_hierarchy_join_is_not_a_role_mismatch():
    """
    A COURSE's parent legitimately IS a DEPARTMENT. `c.parent_id` declares no
    role, so nothing is compared — reading the alias pin onto the FK column
    would reject this correct join.
    """
    sql = """
        SELECT c.id
        FROM academic_unit d
        JOIN academic_unit c ON c.parent_id = d.id AND c.unit_type = 'COURSE'
        WHERE d.unit_type = 'DEPARTMENT'
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_unpinned_key_reference_is_silent():
    """No discriminator pin on `au` means no role to compare against."""
    sql = """
        SELECT au.code
        FROM board b
        JOIN academic_unit au ON au.id = b.course_id
    """
    assert JoinValidator(fk_graph=None).run(ctx_for(sql)).passed


def test_cross_table_domain_mismatch_still_fires():
    """The pre-existing table-level check is untouched."""
    sql = """
        SELECT 1
        FROM board b
        JOIN faculty_cache fc ON fc.id = b.course_id
    """
    result = JoinValidator(fk_graph=None).run(ctx_for(sql))
    assert not result.passed
    assert "domain mismatch" in result.message.lower()

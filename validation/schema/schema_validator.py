"""
validation/schema/schema_validator.py
─────────────────────────────────────
"""
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from .tables import validate_tables
from .columns import validate_columns
from .types import validate_types

class SchemaValidator(BaseValidationStep):
    name = "SchemaValidator"

    def __init__(self, schema_cache=None):
        self.schema_cache = schema_cache

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 2: Verify tables and columns exist in the schema map.
        Delegates to tables.py, columns.py, and types.py.
        """
        if self.schema_cache:
            ctx.schema_map = self.schema_cache

        sql = ctx.working_sql or ctx.sql

        # 1. Verify table names and extract metadata (modifies ctx state)
        res = validate_tables(ctx)
        if res: return res

        # 2. Verify column existence
        res = validate_columns(ctx)
        if res: return res

        # 3. Verify type-compatibility and enum constraints
        res = validate_types(ctx)
        if res: return res

        return ValidationResult(passed=True, step="schema", sql=sql)

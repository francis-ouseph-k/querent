"""
validation/core/base.py
───────────────────────
BaseValidationStep — the contract every validation step implements.

Each step exposes run(ctx) -> ValidationResult and may override before_run /
after_run hooks. The SQLValidator holds an ordered list of these and executes
them in sequence, stopping at the first failure. Keeping the interface this thin
is what lets the 12-step pipeline be declared as a plain list in
build_default_pipeline() and reordered or extended without touching the runner.

A step returns passed=True to continue, or passed=False with a step label and a
correction message that the retry loop feeds back to the model.
"""

from .context import ValidationContext
from models.schema import ValidationResult

class BaseValidationStep:
    name: str = "BaseValidationStep"

    def before_run(self, ctx: ValidationContext) -> None:
        """Lifecycle hook called before run()."""
        pass

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """Execute the validation or transformation logic."""
        raise NotImplementedError

    def after_run(self, ctx: ValidationContext, result: ValidationResult) -> None:
        """Lifecycle hook called after run()."""
        if result.passed:
            ctx.trace.append(f"{self.name} ✓")
        else:
            ctx.trace.append(f"{self.name} ✗ {result.message}")

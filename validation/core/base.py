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

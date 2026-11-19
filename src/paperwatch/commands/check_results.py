"""PASS / WARN / FAIL / SKIP check results, shared by `doctor` and `access-check`, printed one line per check."""

from __future__ import annotations

from dataclasses import dataclass

from paperwatch.errors import PaperwatchError

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass(frozen=True)
class CheckResult:
    component: str
    check_name: str
    verdict: str
    detail: str
    failure: PaperwatchError | None = None
    hint: str | None = None  # shown under the line; defaults to the failure's operator hint


def print_check_results(check_results: list[CheckResult], command_label: str) -> int:
    """Print every result, then a summary. Returns 0 if nothing FAILED, else the first failure's exit code (or 1)."""
    component_column_width = max(len(result.component) for result in check_results)
    for result in check_results:
        print(f"{result.verdict:4}  {result.component:<{component_column_width}}  {result.check_name}: "
              f"{result.detail}")
        hint = result.hint or (result.failure.operator_hint if result.failure is not None else None)
        if hint:
            print(f"{'':4}  {'':<{component_column_width}}  hint: {hint}")

    failed_results = [result for result in check_results if result.verdict == FAIL]
    if not failed_results:
        print(f"\n{command_label}: all required checks passed")
        return 0
    print(f"\n{command_label}: {len(failed_results)} check(s) FAILED in: "
          f"{', '.join(sorted({result.component for result in failed_results}))}")
    first_failure = next((result.failure for result in failed_results if result.failure is not None), None)
    return first_failure.process_exit_code if first_failure else 1

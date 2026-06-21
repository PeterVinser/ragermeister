from typing import Callable

from solution.models.conflict import ConflictReport, Decision
from solution.services.policies.base import ResolutionPolicy


class HumanPolicy(ResolutionPolicy):
    """Holds reports until an external caller submits a Decision."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[ConflictReport, Callable[[Decision], None]]] = {}

    def resolve(
        self,
        report: ConflictReport,
        apply_decision: Callable[[Decision], None],
    ) -> None:
        self._pending[report.report_id] = (report, apply_decision)

    def submit_human_decision(self, decision: Decision) -> None:
        entry = self._pending.pop(decision.report_id, None)
        if entry is None:
            raise KeyError(f"No pending report: {decision.report_id!r}")
        _, apply = entry
        apply(decision)

    def pending_reports(self) -> list[ConflictReport]:
        return [report for report, _ in self._pending.values()]

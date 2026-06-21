from typing import Callable

from solution.models.conflict import ConflictReport, Decision, DecisionAction
from solution.services.policies.base import ResolutionPolicy


class RecencyPolicy(ResolutionPolicy):
    """Auto-resolves every conflict: newest chunk supersedes all implicated ones."""

    def resolve(
        self,
        report: ConflictReport,
        apply_decision: Callable[[Decision], None],
    ) -> None:
        apply_decision(
            Decision(
                report_id=report.report_id,
                action=DecisionAction.UPDATE,
                new_chunk=report.new_chunk,
                chunk_ids_to_remove=list(report.judge_result.implicated_ids),
            )
        )

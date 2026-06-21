from abc import ABC, abstractmethod
from typing import Callable

from solution.models.conflict import ConflictReport, Decision


class ResolutionPolicy(ABC):
    @abstractmethod
    def resolve(
        self,
        report: ConflictReport,
        apply_decision: Callable[[Decision], None],
    ) -> None:
        """Resolve a conflict report and invoke apply_decision with the result."""

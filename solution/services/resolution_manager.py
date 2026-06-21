import queue
from typing import Callable

from solution.models.conflict import ConflictReport, Decision
from solution.services.policies.base import ResolutionPolicy


class ResolutionManager:
    def __init__(
        self,
        policy: ResolutionPolicy,
        apply_decision: Callable[[Decision], None],
    ) -> None:
        self._policy = policy
        self._apply = apply_decision
        self._queue: queue.SimpleQueue[ConflictReport] = queue.SimpleQueue()

    def submit(self, report: ConflictReport) -> None:
        self._queue.put(report)
        self._drain()

    def _drain(self) -> None:
        while not self._queue.empty():
            try:
                report = self._queue.get_nowait()
            except queue.Empty:
                break
            self._policy.resolve(report, self._apply)

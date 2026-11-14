"""Which tasks of a run to work on: --tasks, --level, --match and --limit."""

import re
from typing import Literal

from mcnulty.clients.arena import Task
from mcnulty.frozen import Frozen


class TaskFilter(Frozen):
    """--tasks, --level, --match and --limit, applied in that order."""

    tasks: tuple[str, ...] = ()
    levels: tuple[Literal[1, 2, 3], ...] = ()
    match: re.Pattern[str] | None = None
    limit: int | None = None

    def select(self, tasks: tuple[Task, ...]):
        selected = [t for t in tasks if not self.tasks or t.task_id in self.tasks]
        selected = [t for t in selected if not self.levels or t.level in self.levels]
        selected = [t for t in selected if not self.match or self.match.search(t.instruction)]
        return selected[: self.limit] if self.limit is not None else selected

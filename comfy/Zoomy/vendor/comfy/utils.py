"""The only ``comfy.utils`` member the engine's vendors need: ProgressBar."""

from __future__ import annotations


class ProgressBar:
    """Silent step counter matching the ``comfy.utils.ProgressBar`` interface."""

    def __init__(self, total: int) -> None:
        """Remember how many steps the bar spans."""
        self.total = total
        self.done = 0

    def update(self, step_count: int) -> None:
        """Advance the bar by ``step_count`` steps."""
        self.done += step_count

    def update_absolute(self, value: int, total: int | None = None) -> None:
        """Set the bar to ``value``, optionally rescaling to ``total``."""
        self.done = value
        if total is not None:
            self.total = total

"""Baseline controllers for honest comparison against MPC.

The point of this module is to make the MPC vs non-MPC gap measurable.
Both baselines expose the same `step()` interface as `MPCController` so
the simulator can swap them without branching.
"""

from __future__ import annotations

from .controller import ControlAction, ControllerInputs


class GreedyController:
    """Charge at maximum amperage whenever solar export exceeds a threshold.

    No look-ahead. Tracks nothing across calls.
    """

    def __init__(self, export_threshold_kw: float, amp_max: float) -> None:
        self.export_threshold_kw = export_threshold_kw
        self.amp_max = amp_max

    def step(self, inputs: ControllerInputs) -> ControlAction:
        raise NotImplementedError("Phase 2: greedy rule.")


class TouRuleController:
    """Time-of-use schedule controller.

    Charges at maximum amperage during off-peak windows defined by a
    schedule, ignores solar entirely. The "what every utility-issued EV
    charger does" baseline.
    """

    def __init__(self, off_peak_hours: tuple[int, int], amp_max: float) -> None:
        self.off_peak_hours = off_peak_hours
        self.amp_max = amp_max

    def step(self, inputs: ControllerInputs) -> ControlAction:
        raise NotImplementedError("Phase 2: ToU rule.")

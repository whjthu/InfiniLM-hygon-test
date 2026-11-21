"""Bound Graph64 node-local rendezvous to the original arm grid."""

from dataclasses import dataclass
from threading import Condition


SETUP_LEAD_NS = 2_000_000_000
ARM_LEAD_NS = 200_000_000


class PhaseClosed(RuntimeError):
    """This owner could not enter the next fixed graph phase."""


@dataclass(frozen=True)
class PhaseRelease:
    arrived_epoch_ns: int
    released_epoch_ns: int
    latest_arrival_epoch_ns: int
    earliest_arrival_epoch_ns: int
    cutoff_epoch_ns: int


class PhaseGate:
    def __init__(self, clock, cutoff_epoch_ns, participants, halted):
        if type(cutoff_epoch_ns) is not int or type(participants) is not int or participants < 1:
            raise ValueError("invalid fixed Graph64 phase")
        self.clock = clock
        self.cutoff_epoch_ns = cutoff_epoch_ns
        self.cutoff_monotonic_ns = cutoff_epoch_ns - clock.anchor
        self.participants = participants
        self.halted = halted
        self.condition = Condition()
        self.arrivals = {}
        self.released = None
        self.failure = None

    def _fail(self, card, stage, error_type, observed):
        if self.failure is None and self.released is None:
            self.failure = dict(card=card, stage=stage, error_type=error_type,
                                epoch_ns=self.clock.wall_from_mono(observed), arrived_cards=sorted(self.arrivals))
            self.condition.notify_all()

    def abort(self, card, stage, error_type):
        with self.condition:
            self._fail(card, stage, error_type, self.clock.monotonic())

    def wait(self, card, stage):
        self.clock.verify()
        with self.condition:
            now = self.clock.monotonic()
            if type(card) is not int or not 0 <= card < self.participants or card in self.arrivals:
                raise ValueError("phase arrival does not identify a unique owner")
            if now >= self.cutoff_monotonic_ns:
                self._fail(card, stage, "PHASE_CUTOFF", now)
            elif self.halted():
                self._fail(card, stage, "COOPERATIVE_STOP", now)
            if self.failure is None:
                self.arrivals[card] = now
                if len(self.arrivals) == self.participants:
                    self.released = now
                    self.condition.notify_all()
            while self.released is None and self.failure is None:
                now = self.clock.monotonic()
                remaining = self.cutoff_monotonic_ns - now
                if remaining <= 0 or self.halted():
                    self._fail(card, stage, "PHASE_CUTOFF" if remaining <= 0 else "COOPERATIVE_STOP", now)
                else:
                    self.condition.wait(min(remaining / 10**9, 0.05))
            if self.failure is not None:
                raise PhaseClosed(f"{self.failure['stage']}: {self.failure['error_type']}")
            convert = self.clock.wall_from_mono
            return PhaseRelease(convert(self.arrivals[card]), convert(self.released),
                                convert(max(self.arrivals.values())), convert(min(self.arrivals.values())),
                                self.cutoff_epoch_ns)

    def summary(self):
        with self.condition:
            return dict(cutoff_epoch_ns=self.cutoff_epoch_ns, expected_cards=self.participants,
                        arrived_epoch_ns={card: self.clock.wall_from_mono(value) for card, value in self.arrivals.items()},
                        released_epoch_ns=None if self.released is None else self.clock.wall_from_mono(self.released),
                        first_failure=None if self.failure is None else dict(self.failure))

"""Run the verified ordinary refill coordinator against the 02:45 scope."""

from hashlib import sha256
from pathlib import Path


COORDINATOR = Path(__file__).with_name("run_training80_2347_ordinary_refill_once.py")
COORDINATOR_SHA256 = "f5cbd175c5b3bca49475c8310cb552f2bf180ca1953565cafca4ce646541ce24"


def scope_source():
    raw = COORDINATOR.read_bytes()
    if sha256(raw).hexdigest() != COORDINATOR_SHA256:
        raise ValueError("qualified ordinary refill coordinator changed")
    source = raw.decode("utf-8")
    for previous, selected in (
        ('CALENDAR = LOCAL / "training80-2347-ordinary-70-exact-predecessor-calendar-preparation-v1.json"',
         'CALENDAR = LOCAL / "training80-0245-ordinary-70-exact-predecessor-calendar-preparation-v1.json"'),
        ('CALENDAR_SHA = "f70f6023e2469d624201e9c1fa28abf573e5d091214ac6c85bc45cafd997d6c8"',
         'CALENDAR_SHA = "23b86ba6ac048440d3dbb8c878900cf6a5388363a389019606e814e4267d538e"'),
        ('WAVE = "training80-2347-ordinary"', 'WAVE = "training80-0245-ordinary"'),
    ):
        if source.count(previous) != 1:
            raise ValueError("qualified coordinator scope declaration changed")
        source = source.replace(previous, selected, 1)
    return compile(source, str(COORDINATOR), "exec")


def main():
    # Keep __file__ on this entry point so the existing private child uses the same scope.
    exec(scope_source(), {"__name__": "__main__", "__file__": __file__})


if __name__ == "__main__":
    main()

"""Run the verified ordinary refill coordinator against the 05:44 scope."""

from hashlib import sha256
from pathlib import Path
import sys


COORDINATOR = Path(__file__).with_name("run_training80_2347_ordinary_refill_once.py")
COORDINATOR_SHA256 = "f5cbd175c5b3bca49475c8310cb552f2bf180ca1953565cafca4ce646541ce24"


def scope_source():
    raw = COORDINATOR.read_bytes()
    if sha256(raw).hexdigest() != COORDINATOR_SHA256:
        raise ValueError("qualified ordinary refill coordinator changed")
    source = raw.decode("utf-8")
    for previous, selected in (
        ('CALENDAR = LOCAL / "training80-2347-ordinary-70-exact-predecessor-calendar-preparation-v1.json"',
         'CALENDAR = LOCAL / "training80-0544-ordinary-70-exact-predecessor-calendar-preparation-v1.json"'),
        ('CALENDAR_SHA = "f70f6023e2469d624201e9c1fa28abf573e5d091214ac6c85bc45cafd997d6c8"',
         'CALENDAR_SHA = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"'),
        ('WAVE = "training80-2347-ordinary"', 'WAVE = "training80-0544-ordinary"'),
    ):
        if source.count(previous) != 1:
            raise ValueError("qualified coordinator scope declaration changed")
        source = source.replace(previous, selected, 1)
    return compile(source, str(COORDINATOR), "exec")


def main():
    if sys.argv[1:] == ["--check"]:
        sys.argv.pop()
    # The original child must return through this entry point to keep the same scope.
    exec(scope_source(), {"__name__": "__main__", "__file__": __file__})


if __name__ == "__main__":
    main()

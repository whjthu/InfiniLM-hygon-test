"""Render a separate observer from the exact first-hour controller and transport contract."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = "run_training80_busy_long_host.py"
BASE_SHA = "ef1262903a59046c2c57cb1b7a93d39147858551305b88d2f1d2a6f31676a099"
LAUNCHER = "run_training80_claimed_hour_once.py"
LAUNCHER_SHA = "55665dc8e4632837c23094ddcb757a6f58f6b8bd3bf6a58a8c74a1d4df559f9d"


def digest(data):
    return sha256(data).hexdigest()


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unexpected controller source boundary")
    return source.replace(old, new)


def driver(base_sha, protocol_sha):
    raw = (REPO / "tools" / LAUNCHER).read_bytes()
    if digest(raw) != LAUNCHER_SHA:
        raise ValueError("Unexpected original hour launcher SHA")
    source = raw.decode()
    source = replace(source, 'INSTALL = PROJECT / "incoming/training80-long-busy-host-d32e877d95e0420c94b7f0ab209732ce"',
                     'PRIOR = PROJECT / "incoming/training80-long-busy-host-d32e877d95e0420c94b7f0ab209732ce"\n'
                     'INSTALL = Path(__file__).resolve().parent')
    source = replace(source, f'BASE_SHA = "{BASE_SHA}"', f'BASE_SHA = "{base_sha}"')
    source = replace(source, 'receipt = json.loads(read(INSTALL / "INSTALL_RESULT.json", 16384, INSTALL_RECEIPT_SHA))',
                     'receipt = json.loads(read(PRIOR / "INSTALL_RESULT.json", 16384, INSTALL_RECEIPT_SHA))')
    source = replace(source, 'and len(members) == receipt["source_count"] == 7 and members[BASE_FILE] == BASE_SHA,',
                     f'and len(members) == receipt["source_count"] == 7 and members[BASE_FILE] == "{BASE_SHA}",')
    source = replace(source, '    for name, sha in members.items():',
                     f'    members = dict(members, **{{BASE_FILE: BASE_SHA, "pulse_schedule.py": "{protocol_sha}"}})\n'
                     '    for name, sha in members.items():')
    return source.encode()


def packet():
    raw = (REPO / "tools" / BASE).read_bytes()
    if digest(raw) != BASE_SHA:
        raise ValueError("Unexpected original controller SHA")
    protocol = (HERE / "pulse_schedule.py").read_bytes()
    source = raw.decode()
    source = replace(source, "import collect_training80_busy_long_window as window\n",
                     "import collect_training80_busy_long_window as window\nimport pulse_schedule\n")
    source = replace(source, "SOURCE_SHA = {\n",
                     f'SOURCE_SHA = {{\n    "pulse_schedule.py": "{digest(protocol)}",\n')
    source = replace(
        source,
        '    interrupted = not observation["hour_interval_coverage_complete"]\n',
        '''    interrupted = not observation["hour_interval_coverage_complete"]
    schedule_ended = (observation.get("persisted_points") == window.POINTS
                      and observation.get("stop_reason") is None and not stop())
    schedule_error = None
    if schedule_ended:
        try:
            transport.send(pulse_schedule.finish_packet(request, window.POINTS),
                           min(hard_deadline, time.monotonic_ns() + 2 * NANO))
        except (OSError, TypeError, ValueError) as error:
            schedule_error = type(error).__name__
''')
    source = replace(source, "hard_deadline, interrupted, shutdown=transport.close)",
                     "hard_deadline, not schedule_ended or schedule_error is not None, shutdown=transport.close)")
    source = replace(source, "startup_failure=failure, stop_reason=stop.reason, hard_deadline_CST=HARD_END,",
                     "startup_failure=failure, stop_reason=stop.reason, hard_deadline_CST=HARD_END,\n"
                     "                   schedule_ended=schedule_ended, finish_delivery_error=schedule_error,")
    source = replace(source, "full_hour_observer_C0=verified and not interrupted and set(released) == set(expected),",
                     "full_hour_observer_C0=verified and not interrupted and schedule_ended\n"
                     "                   and schedule_error is None and set(released) == set(expected),")
    source = replace(source, "short.worker(args.request, args.points, args.pci_pins, args.pins_sha, args.parent)",
                     "pulse_schedule.worker(short, args.request, args.points, args.pci_pins, args.pins_sha, args.parent)")
    files = {BASE: source.encode(), "pulse_schedule.py": protocol,
             "observer.py": driver(digest(source.encode()), digest(protocol))}
    for name, data in files.items():
        compile(data, name, "exec")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    path = parser.parse_args().output
    result = packet()
    if path is not None:
        path.mkdir(mode=0o700)
        for name, data in result.items():
            with (path / name).open("xb") as output:
                output.write(data)
    print(json.dumps({name: {"bytes": len(data), "sha256": digest(data)} for name, data in result.items()},
                     sort_keys=True))


if __name__ == "__main__":
    main()

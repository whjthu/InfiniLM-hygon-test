"""Derive the fixed-copy HIP event join experiment from frozen fleet sources."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = Path("code/bw1000/O01-bf16-dependent-d2h-chunks-fleet-v1")
NEW = HERE.relative_to(ROOT)
ORIGINAL_TRANSFER = Path("code/bw1000/O01-bf16-dependent-transfer-overlap-fleet-v1/transfer.py")
ORIGINAL_TRANSFER_SHA = "26bb931d61c2289fd6f25fc12403eff4d1f2fd9f31a372bfa2e0389af60bc419"
BASELINE = Path("results/bw1000/2026-11-22/O01-dependent-d2h-chunks-original584-terminal.json")
BASELINE_SHA = "2b91874bb8e165e58ce633c6eda8d6c056df34a1c543dacfba0c5789be7bd363"
OLD_LABEL = "O01-BF16-dependent-d2h-chunks-fleet-v1"
OLD_REQUEST = "93c5218eea4e4f1d8a246ba7583dac85"
LABEL = "O01-BF16-dependent-d2h-event-join-fleet-v1"
REQUEST = "53ec0f52789d442fa6a9c48b00c759e1"
ORDER = ("host_two_fences", "event_join", "event_join", "host_two_fences", "common_host_two_fences")
PINS = {
    "SOURCE_FILES.json": "246aa225b881c3b565ca98049bee7d6ddf5e0ae642efdee6d335b3213ffd0bf2",
    "ROOT_CLAIM.json": "04ffe11562cdd3665c9de86b0b3ba356d0fac7517cd832dd8ae3b2efc1a7dffd",
    "package.py": "a2d06506c12fd3a6d0abbdbdac2295ad64d05c3c0bf3234456fe8a35bc46a30b",
    "launch.py": "21e564c0d170e913b30d183bed195725969256387a81853fa7ec82606589a347",
    "overlap.py": "012e289371d5557f834e8d92dc1efd348c9527f7d04d669c02c5ddb4dd914dd9",
}


def sha(raw):
    return sha256(raw).hexdigest()


def check(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("frozen source bytes differ: " + str(path))
    return raw


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen replacement context differs: " + old)
    return text.replace(old, new, 1)


def update_scope(text):
    return once(once(text, OLD_LABEL, LABEL), OLD_REQUEST, REQUEST)


def contract(text):
    text = once(text, "ORDER = ('one_copy', 'two_chunks', 'two_chunks', 'one_copy', 'common_one_copy')",
                f"ORDER = {ORDER!r}")
    text = once(text, "FENCES_PER_UNIT = 2\n", "")
    text = once(text, '''def copy_calls_per_unit(mode):
    need(mode in ORDER, "invalid same-stream copy mode")
    return 2 if mode == "two_chunks" else 1''',
                '''def uses_event_join(mode):
    need(mode in ORDER, "invalid host completion mode")
    return mode == "event_join"''')
    text = once(text, '''    return dict(groups=groups, graph_replays=replays, stream_fences=groups*FENCES_PER_UNIT,
                transfer_calls=groups*copy_calls_per_unit(mode),d2h_payload_bytes=groups*TRANSFER_BYTES,''',
                '''    event_waits = groups if uses_event_join(mode) else 0
    return dict(groups=groups, graph_replays=replays, stream_fences=2*groups-event_waits,
                stream_event_waits=event_waits,transfer_calls=groups,d2h_payload_bytes=groups*TRANSFER_BYTES,''')
    text = once(text, '''    if row["groups"]:
        need(row["terminal_polarity"]''',
                '''    join_ns = row["host_event_join_enqueue_ns"]
    need(type(join_ns) is int and 0 <= join_ns <= row["host_submit_ns"]
         and (row["stream_event_waits"] > 0 or join_ns == 0), "invalid host event-join enqueue time")
    if row["groups"]:
        need(row["terminal_polarity"]''')
    return text


def worker(text, index, output):
    for name, pin in (("contract.py", sha(output["contract.py"])), ("scope.py", sha(output["scope.py"])),
                      ("transfer.py", ORIGINAL_TRANSFER_SHA)):
        text = once(text, index["files"][name]["SHA256"], pin)
    text = once(text, "import argparse\n", "import argparse\nimport ctypes as C\n")
    text = once(text, "cohort_for, copy_calls_per_unit, counters, grid, need", "cohort_for, counters, grid, need, uses_event_join")
    text = once(text, "from duplex_worker import bind, fresh_inventory,", "from duplex_worker import HIP_PATH, bind, fresh_inventory,")
    text = once(text, "from schedule import Clock", "from hip_runtime import DlInfo, I, U, symbol\nfrom schedule import Clock")
    text = once(text, '''def timed_unit(hip, streams, copy, variants, transfer, mode, clock):
    calls = copy_calls_per_unit(mode)''',
                '''def bind_stream_wait_event(hip):
    lookup = C.CDLL("libdl.so.2").dladdr
    lookup.argtypes, lookup.restype = [V, C.POINTER(DlInfo)], I
    expected = Path(HIP_PATH)
    hip.functions["StreamWaitEvent"] = symbol(hip.lib, lookup, "hipStreamWaitEvent", [V, V, U], expected)


def timed_unit(hip, streams, copy, variants, transfer, mode, clock):
    event_join = uses_event_join(mode)''')
    text = once(text, "    transfer.enqueue(dma,calls)", "    transfer.enqueue(dma)")
    text = once(text, '''    prior.interval(hip,copy_last,dma)
    before = clock.monotonic()
    hip.call("StreamSynchronize",stream)
    hip.call("StreamSynchronize",dma)''',
                '''    prior.interval(hip,copy_last,dma)
    join_enqueue_ns = 0
    if event_join:
        join_begin = clock.monotonic()
        hip.call("StreamWaitEvent", dma, last, 0)
        join_enqueue_ns = clock.monotonic()-join_begin
    before = clock.monotonic()
    if not event_join:
        hip.call("StreamSynchronize",stream)
    hip.call("StreamSynchronize",dma)''')
    text = once(text, "host_submit_ns=(graph_queued-graph_start)+(copy_queued-copy_start),",
                '''host_submit_ns=(graph_queued-graph_start)+(copy_queued-copy_start)+join_enqueue_ns,
                host_event_join_enqueue_ns=join_enqueue_ns,''')
    text = once(text, "    quarters = [0]*4", "    join_enqueue_ns = 0\n    quarters = [0]*4")
    text = once(text, "        compute_stream += stats", '        join_enqueue_ns += stats["host_event_join_enqueue_ns"]\n        compute_stream += stats')
    text = once(text, "row.update(state=state,passed=False,**work,**host,",
                "row.update(state=state,passed=False,**work,**host,host_event_join_enqueue_ns=join_enqueue_ns,")
    text = once(text, '''    hip = bind()
    mapping =''', '''    hip = bind()
    bind_stream_wait_event(hip)
    mapping =''')
    return once(text, '''                        transfer.prepare_chunks()
                        pins = []
                        for calls in (1, 2):
                            transfer.poison()
                            transfer.enqueue(copy[0], calls)
                            hip.call("StreamSynchronize", copy[0])
                            pins.append(transfer.audit())
                        need(pins[0] == pins[1], "full and two-chunk D2H source payload differs")
                        row["preparation"]["original_transfer_SHA256"] = pins[0]
                        row["preparation"]["prepared_transfer_calls_per_unit"] = [1, 2]''',
                '''                        poison()
                        transfer.poison()
                        timed_unit(hip, (current, markers[1], markers[2]), copy, submissions["depth9"],
                                   transfer, "event_join", clock)
                        verify((0,1))
                        row["preparation"]["original_transfer_SHA256"] = transfer.audit()
                        row["preparation"]["verified_HIP_event_join"] = True''')


def package(text):
    text = once(text, "TRANSFER_BASELINE=Path(", f"JOIN_BASELINE=Path({BASELINE.as_posix()!r})\nJOIN_BASELINE_SHA={BASELINE_SHA!r}\n\nTRANSFER_BASELINE=Path(")
    text = once(text, "    measured_raw = read(repo / TRANSFER_BASELINE)", '''    fixed_raw = read(repo / JOIN_BASELINE)
    fixed = json.loads(fixed_raw)
    need(sha(fixed_raw) == JOIN_BASELINE_SHA and fixed["request"] == "93c5218eea4e4f1d8a246ba7583dac85"
         and fixed["device_qualified"] is True and fixed["independent_post_clean_nodes"] == 73
         and (fixed["valid_cards"], fixed["completed_arms"], fixed["actual_busy_card_samples"]) == (584, 2920, 35040)
         and fixed["paired_comparison"]["measured_pairs"] == 1168
         and fixed["paired_comparison"]["median_two_chunks_over_one_copy"] == 0.998266112045002,
         "the actual single-copy retention evidence differs")
    measured_raw = read(repo / TRANSFER_BASELINE)''')
    return once(text, "return plan,body,[TRANSFER_BASELINE.as_posix(),", "return plan,body,[JOIN_BASELINE.as_posix(),TRANSFER_BASELINE.as_posix(),")


def render(repo):
    base = {name: check(repo / OLD / name, expected) for name, expected in PINS.items()}
    index = json.loads(base["SOURCE_FILES.json"])
    for name, spec in index["files"].items():
        base[name] = check(repo / spec["path"], spec["SHA256"])
    fixed = check(repo / BASELINE, BASELINE_SHA)
    transfer = check(repo / ORIGINAL_TRANSFER, ORIGINAL_TRANSFER_SHA)
    output = {}

    def change(name, transform):
        output[name] = transform(base[name].decode()).encode()

    change("contract.py", contract)
    change("scope.py", lambda t: once(
        once(update_scope(t), "O01-dependent-d2h-chunks-", "O01-dependent-d2h-join-"),
        "PRIORITY_NODES = frozenset(('f13r1n12',))", "PRIORITY_NODES = frozenset(('f13r4n09',))"))
    change("worker.py", lambda t: worker(t, index, output))
    change("report.py", lambda t: once(once(once(t,
        'prep["prepared_transfer_calls_per_unit"] == [1, 2], "both fixed copy geometries were not verified"',
        'prep["verified_HIP_event_join"] is True, "initial HIP event join, payload or provider proof is missing"'),
        '"stream_fences", "transfer_calls"', '"stream_fences", "stream_event_waits", "transfer_calls", "host_event_join_enqueue_ns"'),
        "per_card_two_chunks_over_one_copy_pairs", "per_card_event_join_over_host_two_fences_pairs"))
    change("controller.py", lambda t: once(once(once(update_scope(t),
        index["files"]["report.py"]["SHA256"], sha(output["report.py"])),
        "per_card_two_chunks_over_one_copy_pairs", "per_card_event_join_over_host_two_fences_pairs"),
        "median_two_chunks_over_one_copy", "median_event_join_over_host_two_fences"))
    change("package.py", package)
    change("launch.py", lambda t: once(once(once(once(once(t, OLD_REQUEST, REQUEST),
        "Dependent same-stream D2H chunk fleet", "Dependent HIP event-join fleet"),
        "O01.bf16-dependent-d2h-chunks-eligible-fleet", "O01.bf16-dependent-d2h-event-join-eligible-fleet"),
        "D2H_REFERENCE, ORIGINAL, QWEN_SHAPE", "D2H_REFERENCE, JOIN_BASELINE, ORIGINAL, QWEN_SHAPE"),
        "TRANSFER_BASELINE, ORIGINAL, D2H_REFERENCE", "JOIN_BASELINE, TRANSFER_BASELINE, ORIGINAL, D2H_REFERENCE"))
    change("overlap.py", lambda t: t.replace("common_one_copy", "common_host_two_fences"))
    claim = json.loads(base["ROOT_CLAIM.json"])
    claim.update(label=LABEL, request=REQUEST)
    output["ROOT_CLAIM.json"] = encode(claim)
    scope = json.loads(base["SCOPE.json"])
    scope.pop("d2h_split_bytes")
    scope.update(label=LABEL, request=REQUEST, arms=ORDER, transfer_calls_per_unit=[1]*5,
                 stream_fences_per_measurement_unit=[2, 1, 1, 2, 2],
                 stream_event_waits_per_measurement_unit=[0, 1, 1, 0, 0],
                 priority_nodes=["f13r4n09"],
                 work_difference="two host fences versus copy waits on compute tail, then one host fence")
    scope["original_584card_copy_size_terminal"] = dict(path=BASELINE.as_posix(), bytes=len(fixed), SHA256=BASELINE_SHA)
    output["SCOPE.json"] = encode(scope)
    index["original_584card_copy_size_terminal"] = scope["original_584card_copy_size_terminal"]
    index["files"]["transfer.py"] = dict(path=ORIGINAL_TRANSFER.as_posix(), bytes=len(transfer), SHA256=ORIGINAL_TRANSFER_SHA)
    for name in index["files"].keys() & output.keys():
        raw = output[name]
        index["files"][name] = dict(path=(NEW / name).as_posix(), bytes=len(raw), SHA256=sha(raw))
    output["SOURCE_FILES.json"] = encode(index)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()
    for name, raw in sorted(render(ROOT).items()):
        path = HERE / name
        if args.write_generated:
            with path.open("xb") as output:
                output.write(raw)
        elif path.read_bytes() != raw:
            raise ValueError("rendered source differs: " + name)
        print(sha(raw), name)


if __name__ == "__main__":
    main()

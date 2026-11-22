"""Derive a distinct same-stream D2H request from the qualified overlap fleet."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = Path("code/bw1000/O01-bf16-dependent-transfer-overlap-fleet-v1")
NEW = HERE.relative_to(REPO)
BASELINE = Path("results/bw1000/2026-11-22/O01-dependent-transfer-overlap-fleet-original592-terminal.json")
BASELINE_SHA = "7ded16778c5faa0f29cca323f3e7e3f19894c9d7ff9dbd8771d5d87b066c922d"
OLD_LABEL = "O01-BF16-dependent-transfer-overlap-fleet-v1"
LABEL = "O01-BF16-dependent-d2h-chunks-fleet-v1"
OLD_REQUEST = "a02a3ada39494c0cec0c233639025f02"
REQUEST = "93c5218eea4e4f1d8a246ba7583dac85"
ORDER = ("one_copy", "two_chunks", "two_chunks", "one_copy", "common_one_copy")
PINS = {
    "SOURCE_FILES.json": "39e655de6114ec11783f3625c5772b4d14046f110627ee9a23b23c0b859f376f",
    "ROOT_CLAIM.json": "89f827c766846bd7a2cf878867ffc00bd6d69def9cc77ab173edd034b473af17",
    "package.py": "a8af1299d60ca020ae5319019bf52d9f70abd18aa642b369331c3b49c5a1e06a",
    "launch.py": "3039006af67abe9fcb4183e5cce0ff40519730b54923071664a5caa851629892",
    "overlap.py": "986b52ed9672f8688b3adf72b18f6c0759bb9153586bdfe1cd9aab82cf82dfbe",
}


def sha(raw):
    return sha256(raw).hexdigest()


def checked(path, pin):
    raw = path.read_bytes()
    if sha(raw) != pin:
        raise ValueError("frozen source bytes differ: " + str(path))
    return raw


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen replacement context differs: " + old)
    return text.replace(old, new, 1)


def contract(text):
    text = once(text, 'ORDER = ("serial", "overlap", "overlap", "serial", "common_overlap")', f"ORDER = {ORDER!r}")
    text = once(text, "def counters(mode, groups):",
                '''def copy_calls_per_unit(mode):
    need(mode in ORDER, "invalid same-stream copy mode")
    return 2 if mode == "two_chunks" else 1


def counters(mode, groups):''')
    return once(text, "transfer_calls=groups,d2h_payload_bytes=groups*TRANSFER_BYTES,",
                "transfer_calls=groups*copy_calls_per_unit(mode),d2h_payload_bytes=groups*TRANSFER_BYTES,")


def transfer(text):
    return once(text, '''    def enqueue(self, stream):
        self.hip.call("MemcpyAsync",self.host[1],self.device[1],TRANSFER_BYTES,self.dma.D2H,stream)''',
                '''    def prepare_chunks(self):
        half = TRANSFER_BYTES // 2
        need(TRANSFER_BYTES == 2 * half and half % 256 == 0, "unaligned fixed D2H midpoint")
        destination, source = self.host[1], self.device[1]
        self.placements = (
            ((destination, source, TRANSFER_BYTES),),
            ((destination, source, half),
             (self.dma.add(destination, half), self.dma.add(source, half), half)),
        )

    def enqueue(self, stream, calls):
        for destination, source, size in self.placements[calls - 1]:
            self.hip.call("MemcpyAsync", destination, source, size, self.dma.D2H, stream)''')


def worker(text, index, output):
    for name in ("contract.py", "scope.py", "transfer.py"):
        text = once(text, index["files"][name]["SHA256"], sha(output[name]))
    text = once(text, "cohort_for, counters, grid, need", "cohort_for, copy_calls_per_unit, counters, grid, need")
    text = once(text, "    stream, first, last = streams\n", "    calls = copy_calls_per_unit(mode)\n    stream, first, last = streams\n")
    text = once(text, '''    wait = 0
    if mode=="serial":
        before = clock.monotonic()
        hip.call("StreamSynchronize",stream)
        wait = clock.monotonic()-before
''', "")
    text = once(text, "    transfer.enqueue(dma)", "    transfer.enqueue(dma,calls)")
    text = once(text, '    if mode!="serial":\n        hip.call("StreamSynchronize",stream)',
                '    hip.call("StreamSynchronize",stream)')
    text = once(text, "host_sync_wait_ns=wait+done-before", "host_sync_wait_ns=done-before")
    return once(text, '''                        transfer.poison()
                        transfer.enqueue(copy[0])
                        hip.call("StreamSynchronize",copy[0])
                        row["preparation"]["original_transfer_SHA256"] = transfer.audit()''',
                '''                        transfer.prepare_chunks()
                        pins = []
                        for calls in (1, 2):
                            transfer.poison()
                            transfer.enqueue(copy[0], calls)
                            hip.call("StreamSynchronize", copy[0])
                            pins.append(transfer.audit())
                        need(pins[0] == pins[1], "full and two-chunk D2H source payload differs")
                        row["preparation"]["original_transfer_SHA256"] = pins[0]
                        row["preparation"]["prepared_transfer_calls_per_unit"] = [1, 2]''')


def package(text):
    text = once(text, "LEDGER=Path(", f"TRANSFER_BASELINE=Path({BASELINE.as_posix()!r})\nTRANSFER_BASELINE_SHA={BASELINE_SHA!r}\n\nLEDGER=Path(")
    text = once(text, "    raw=read(repo/ORIGINAL)", '''    measured_raw = read(repo / TRANSFER_BASELINE)
    measured = json.loads(measured_raw)
    summary = measured["summary"]
    need(sha(measured_raw) == TRANSFER_BASELINE_SHA
         and measured["request"] == "a02a3ada39494c0cec0c233639025f02"
         and summary["device_qualified"] is True and measured["post"]["clean_nodes"] == 74
         and (summary["valid_cards"], summary["completed_arms"], summary["actual_busy_card_samples"]) == (592, 2960, 35520)
         and summary["paired_comparison"]["measured_pairs"] == 1184
         and summary["paired_comparison"]["median_overlap_over_serial"] == 1.0297971503688794
         and measured["workload"] == dict(transfer_bytes_per_unit=761856, graphs=2,
                                          logical_BF16_chains=36, equal_stream_fences=2),
         "the original measured whole-fleet overlap baseline differs")
    raw=read(repo/ORIGINAL)''')
    return once(text, "return plan,body,[ORIGINAL.as_posix(),", "return plan,body,[TRANSFER_BASELINE.as_posix(),ORIGINAL.as_posix(),")


def render(repo):
    base = {name: checked(repo / OLD / name, pin) for name, pin in PINS.items()}
    index = json.loads(base["SOURCE_FILES.json"])
    for name, record in index["files"].items():
        base[name] = checked(repo / record["path"], record["SHA256"])
    baseline_raw = checked(repo / BASELINE, BASELINE_SHA)
    output = {}

    def change(name, transform):
        output[name] = transform(base[name].decode()).encode()

    change("contract.py", contract)
    change("scope.py", lambda t: once(once(once(t, OLD_LABEL, LABEL), OLD_REQUEST, REQUEST),
                                     "O01-dependent-d2h-fleet-", "O01-dependent-d2h-chunks-"))
    change("transfer.py", transfer)
    change("worker.py", lambda t: worker(t, index, output))
    change("report.py", lambda t: once(
        once(t, '            transfer_sha = prep["original_transfer_SHA256"]',
             '''            need(prep["prepared_transfer_calls_per_unit"] == [1, 2], "both fixed copy geometries were not verified")
            transfer_sha = prep["original_transfer_SHA256"]'''),
        "per_card_overlap_over_serial_pairs", "per_card_two_chunks_over_one_copy_pairs"))
    change("controller.py", lambda t: once(once(once(once(once(t, OLD_LABEL, LABEL), OLD_REQUEST, REQUEST),
        index["files"]["report.py"]["SHA256"], sha(output["report.py"])),
        "per_card_overlap_over_serial_pairs", "per_card_two_chunks_over_one_copy_pairs"),
        "median_overlap_over_serial", "median_two_chunks_over_one_copy"))
    change("package.py", package)
    change("launch.py", lambda t: once(once(once(once(t, OLD_REQUEST, REQUEST),
        "Dependent transfer overlap fleet v1", "Dependent same-stream D2H chunk fleet"),
        "O01.bf16-dependent-transfer-overlap-eligible-fleet", "O01.bf16-dependent-d2h-chunks-eligible-fleet"),
        "QWEN_VJP, V5_TERMINAL", "QWEN_VJP, TRANSFER_BASELINE, V5_TERMINAL").replace(
            "ORIGINAL, D2H_REFERENCE, V5_TERMINAL, T01_PROVIDER", "TRANSFER_BASELINE, ORIGINAL, D2H_REFERENCE, V5_TERMINAL, T01_PROVIDER"))
    change("overlap.py", lambda t: t.replace("common_overlap", "common_one_copy"))
    claim = json.loads(base["ROOT_CLAIM.json"])
    claim.update(label=LABEL, request=REQUEST)
    output["ROOT_CLAIM.json"] = encode(claim)
    scope = json.loads(base["SCOPE.json"])
    scope.update(label=LABEL, request=REQUEST, arms=ORDER, transfer_calls_per_unit=[1, 2, 2, 1, 1],
                 d2h_split_bytes=[380928, 380928], shared_submission="overlap on same original compute/copy streams",
                 work_difference="number of consecutive same-stream D2H calls for the same 761856 bytes")
    scope["original_592card_overlap_terminal"] = dict(path=BASELINE.as_posix(), bytes=len(baseline_raw), SHA256=BASELINE_SHA)
    output["SCOPE.json"] = encode(scope)
    index["original_592card_overlap_terminal"] = scope["original_592card_overlap_terminal"]
    for name in index["files"].keys() & output.keys():
        raw = output[name]
        index["files"][name] = dict(path=(NEW / name).as_posix(), bytes=len(raw), SHA256=sha(raw))
    output["SOURCE_FILES.json"] = encode(index)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()
    for name, raw in sorted(render(REPO).items()):
        path = HERE / name
        if args.write_generated:
            with path.open("xb") as output:
                output.write(raw)
        elif path.read_bytes() != raw:
            raise ValueError("rendered source differs: " + name)
        print(sha(raw), name)


if __name__ == "__main__":
    main()

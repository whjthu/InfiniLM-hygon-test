"""Bind the frozen Z-Image route witness to a freshly verified .2075 lease."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO / "local-only"
V2 = HERE.with_name("zimage-transposed-lt-unseen-trace-v2")
V2_MANIFEST_SHA = "f137e7d5e504c1b9c1ea5fcb8b9af0d11db298415e72e551a8270d786e87db3e"
V2_RUNTIME_SHA = "963588ac56a4152ce2f762e9de55144ffe42c0643eae8b31ebe9a0312b161d86"
V2_REQUEST = "0f3f53e006af491da5e4e6d5ace1df21"
V2_LABEL = "O01-zimage-transposed-lt-unseen-solution-v2"
REQUEST = "c1935bca14ff46b89d61abe7808d63ac"
LABEL = "O01-zimage-transposed-lt-unseen-solution-v3"
PARENT, NODE = "1103606", "f12r2n11"
LEASE_REQUEST = "3228d0f2042f4224bb72bdf2313617c5"
LEASE, LEASE_LABEL = "1103606.2075", "training80-dcu-standby-3228d0f2042f-f12r2n11"
LEASE_START = 1795463528.5417025
RENEWAL_EPOCH = 1795463570.382868
RENEWAL_SNAPSHOT_SHA = "5cdee3a7c18d4bfecdf93369375a9b907bf6b0f00079d21677efb7d65cdafdef"
RENEWAL = "training80-standby-f12r2n11-original1989-renewed2075.json"
RENEWAL_SHA = "d784a4c3aa25690a47ebe1bdeaadd24de4eea1233dd278fb42c0d6a85eadc3fa"
OLD_STANDBY, OLD_STANDBY_LABEL = "1103606.1989", "training80-dcu-standby-fa66e1eb8cfc-f12r2n11"
OLD_STANDBY_RECORD = "training80-o01-1989-actual-terminal-only-2b13127581ae40e2a026f9cae87a5868.json"
OLD_STANDBY_SHA = "02e7e893e85ebffa051a6f1f90ee6b850ac9c9d04626bfc747d03571b8ebfed7"
TRANSPORT_PREFIX = "O01-zimage-lt-v2-original-" + V2_REQUEST + "-"
TRANSPORT_SHA = {
    "attempt-1-reply.json": "c67cf2378e9f8d1f16aa48e7a2a3f8b60ea7c6a9ee026cc35b5b3d7ab856a7c1",
    "attempt-2-reply.json": "dd95c64ce9104ef6866ecfbecfe3d457000ea5e5e3bd2f8a83898b99c147dc82",
    "after-preauth-proof-attempt-1-reply.json": "3fdab7bb19d5e13f42bb32af43bb0f5e353a626ed9c858e2f8c4dd2807782afa",
    "single-auth-session-reply.json": "c2e0971d82fd32ad1ee16ccd6abc9c69ced34907d228fbd85d4ff3fd04d24f42",
}


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if not 0 < len(raw) <= 65536 or digest(raw) != expected:
        raise ValueError("source or authority bytes differ: " + path.name)
    return raw


def prior_evidence():
    epochs = []
    for name, digest_value in TRANSPORT_SHA.items():
        receipt = json.loads(pinned(LOCAL / (TRANSPORT_PREFIX + name), digest_value))
        if receipt["request"] != V2_REQUEST or receipt["rc"] != 255 or receipt["records"] != []:
            raise ValueError("archived predecessor receipt differs")
        if name == "single-auth-session-reply.json":
            if receipt["authority_received"] is not False or receipt["bound_payload_send_attempted"] is not False:
                raise ValueError("old authenticated payload delivery remains uncertain")
        elif receipt["pre_authentication_only"] is not True:
            raise ValueError("old preauthentication boundary is unproved")
        epochs.append(receipt["epoch"])
    old = json.loads(pinned(LOCAL / OLD_STANDBY_RECORD, OLD_STANDBY_SHA))
    old_row = [r for item in old["rows"] for r in item.get("rows", []) if r[0] == OLD_STANDBY]
    if (epochs != sorted(set(epochs)) or old["rc"] != 0 or len(old_row) != 1
            or old_row[0][1:4] != [OLD_STANDBY_LABEL, "COMPLETED", "0:0"] or old_row[0][5] != NODE):
        raise ValueError("archived predecessor accounting differs")
    path = REPO / "results/bw1000/2026-11-24" / RENEWAL
    renewal = json.loads(pinned(path, RENEWAL_SHA))
    if (renewal["old_step"] != OLD_STANDBY or renewal["new_step"] != LEASE or renewal["node"] != NODE
            or renewal["new_step_running"] is not True or renewal["same_PCI_with_increasing_compute_counters"] != 8):
        raise ValueError("published standby renewal differs")
    return {"prior_request": V2_REQUEST, "four_attempts_never_sent_payload": True,
            "transport_replies_SHA256": list(TRANSPORT_SHA.values()), "retired_1989_SHA256": OLD_STANDBY_SHA,
            "historical_renewal_2075_SHA256": RENEWAL_SHA}


def frozen_renderer():
    bound = json.loads(pinned(V2 / "SOURCE_FILES.json", V2_MANIFEST_SHA))
    data = {leaf: pinned(V2 / leaf, pin["SHA256"]) for leaf, pin in bound["files"].items()}
    for leaf, raw in data.items():
        if len(raw) != bound["files"][leaf]["bytes"]:
            raise ValueError("frozen v2 source size differs")
    version = ModuleType("zimage_frozen_v2")
    source = V2 / "prepare.py"
    version.__file__ = str(source)
    exec(compile(data["prepare.py"], str(source), "exec"), vars(version))
    previous = version.source_files()
    if digest(previous["worker.py"]) != V2_RUNTIME_SHA:
        raise ValueError("frozen v2 worker bytes differ")
    version.REQUEST, version.LABEL = REQUEST, LABEL
    version.STANDBY, version.STANDBY_NAME = LEASE, LEASE_LABEL
    actual = version.source_files()
    expected = previous["worker.py"].replace(V2_REQUEST.encode(), REQUEST.encode()).replace(V2_LABEL.encode(), LABEL.encode())
    if actual["worker.py"] != expected or previous["contract.py"] != actual["contract.py"]:
        raise ValueError("scope changes exceed request and label")
    return version


def handoff(path, expected):
    if expected == RENEWAL_SNAPSHOT_SHA:
        raise ValueError("historical renewal snapshot cannot authorize current lease")
    value = json.loads(pinned(path, expected))
    before, after = value["before"][LEASE], value["after"][LEASE]
    identity = {"id": LEASE, "nodes": NODE, "label": LEASE_LABEL, "state": "RUNNING", "cards": 8}
    first, second = (value[phase][LEASE_REQUEST]["workers"][NODE] for phase in ("first", "second"))
    summary = value["paired"][LEASE_REQUEST][NODE]
    pcis = set(first["rows"])
    observed = value["epoch"]
    point_times = [row["epoch"] for point in (first, second) for row in point["rows"].values()] + [observed]
    if (before != identity or after != identity or len(pcis) != 8 or "0000:09:00.0" not in pcis
            or set(second["rows"]) != pcis or set(summary["growing"]) != pcis
            or not all(point["step"] == LEASE and point["worker_start"] == LEASE_START for point in (first, second, summary))
            or not all(type(stamp) in (float, int) and math.isfinite(stamp) for stamp in point_times)
            or not all(RENEWAL_EPOCH < first["rows"][pci]["epoch"] < second["rows"][pci]["epoch"] <= observed
                       and second["rows"][pci]["calls"] > first["rows"][pci]["calls"] for pci in pcis)):
        raise ValueError("current .2075 lease has no exact, advancing two-point proof")
    return {"step": LEASE, "name": LEASE_LABEL, "observed_epoch": observed, "SHA256": expected}


def execution_files(version, fresh, evidence):
    files = version.execution_files(fresh)
    source = files["entry.py"].decode()
    source = version.replace(source, "def srun(name, node, cpus, memory, gpu, seconds):", f"""def verify_retired_predecessor():
    prior_run = RESULTS / {V2_LABEL + "-run"!r}
    require(not prior_run.exists() and not prior_run.is_symlink(), "Prior device namespace requires reconciliation")
    wanted = {{ {V2_LABEL!r} + "-" + phase for phase in ("control", "pre", "gpu", "post") }}
    records = query("sacct", "-nP", "-j", PARENT, "--format", ",".join(COLUMNS))
    rows = [line.split("|") for line in records.splitlines() if line]
    require(all(len(row) == 7 for row in rows), "Predecessor accounting schema differs")
    require(not any(row[1] in wanted for row in rows), "Prior device accounting requires reconciliation")
    old = [row for row in rows if row[0] == {OLD_STANDBY!r}]
    require(len(old) == 1 and old[0][1:4] == [{OLD_STANDBY_LABEL!r}, "COMPLETED", "0:0"]
            and old[0][5] == NODE, "The .1989 standby terminal must be reconciled")


def srun(name, node, cpus, memory, gpu, seconds):""")
    source = version.replace(source, "    parent_owned()\n    pinned_standby()\n",
                             "    parent_owned()\n    verify_retired_predecessor()\n    pinned_standby()\n")
    source = version.replace(source, '        claim = json.loads((RUN / "DISPATCH.json").read_bytes())\n        pinned_standby()',
                             '        claim = json.loads((RUN / "DISPATCH.json").read_bytes())\n'
                             '        verify_retired_predecessor()\n        pinned_standby()')
    claim = json.loads(files["DISPATCH.json"])
    files["entry.py"] = source.encode()
    claim["files"]["entry.py"] = version.pins({"entry.py": files["entry.py"]})["entry.py"]
    claim["predecessor"] = evidence
    files["DISPATCH.json"] = version.encode(claim)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    commands.add_parser("source").add_argument("output", type=Path)
    complete = commands.add_parser("envelope")
    complete.add_argument("fresh_evidence", type=Path)
    complete.add_argument("evidence_sha256")
    complete.add_argument("output", type=Path)
    args = parser.parse_args()
    prior = prior_evidence()
    version = frozen_renderer()
    files = version.source_files() if args.kind == "source" else execution_files(
        version, handoff(args.fresh_evidence, args.evidence_sha256), prior)
    print(json.dumps(version.stage(args.output, files), sort_keys=True))

"""Render the isolated full-tree/wave32 comparison from the original .340 device path."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from authority import DSO, FULL_PROOF, INPUT_SHA256, LABEL, NODE, ORDER, PARENT, REQUEST, WAVE_PROOF, pinned


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = HERE.parent / "output-residual-component-GPU-v2"
SOURCE = {
    "controller.py": "ed304309c3a992fcf48295e7cf9e3ee767dc3396ef9ad753fc1fe20691d4cffb",
    "worker.py": "78b494a09f9ab1268849ce8ec03a46305541fc8de551b6581a9c359ef2722fae",
    "run_arm.py": "14f0320fb481fa60b87bb06f535a8b12063875f4c6bd6374bd8a5bcede3f7e50",
    "CLAIM.json": "5bff0fc3ea0fd463b47b824ac20ba43a042c22446096b343d78c5b1f1768a23e",
    "inside.sh": "3445277ccef96010e045c8da4eb6fed6ad02bc0c2441fd01907c2d7e3247a808",
    "outer.sh": "b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d",
    "probe.py": "c70502be58f458d37f224fcead5aa53d2e45d44f01a245629ffe8018e6deac56",
    "compare_pair.py": "aa22d04b06a62e614233948f9303b00c1f79c49701aba5b3bf67e0aaaba29e7d",
    "arm_watchdog.py": "97848eb1672cfdff42ccd0f2a67ab1258f6b5a1e3c672b569b059750bde5d2c8",
    "CHECKPOINT_WEIGHTS.json": "69d600435bcf35906c2d2f72452340467613e652868d1eb80bf8708a371ff23d",
}
EVIDENCE = (
    (*FULL_PROOF, "2026-11-22/Kimi461-original678-full-tree-passive-rocm-verified.json"),
    (*WAVE_PROOF, "2026-11-22/Kimi461-original-wave32-exact-object-link-rocm-CPU751-terminal.json"),
    ("proof-input-340.json", 11654, "c5fd51176309206e5fa4d92851463d4d0e7ab99919217249263d7bbd512b3d15",
     "2026-11-20/Kimi461-original340-residual-component-GPU-v2-terminal.json"),
)


def edit(text, changes):
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError("original source boundary differs: " + before.splitlines()[0])
        text = text.replace(before, after, 1)
    return text


def controller(text):
    return edit(text, [
        ("import time\n", "import time\n\nfrom authority import LABEL, REQUEST, allocation, qualify, terminal\n"),
        ('/' + OLD.name + '"', '/' + HERE.name + '"'),
        ('    library = ROOT / Path(claim["component"]["path"]).relative_to("/refactor")\n'
         '    need(sha(library) == claim["component"]["SHA256"], "CPU331 artifact differs")\n',
         '    need(label == LABEL and claim["request"] == REQUEST, "exact paired experiment lease differs")\n'
         '    rows = query("sacct", "-nP", "-j", parent, "-S", "2026-11-13", "-o", ACCOUNT)\n'
         '    authority = qualify(ROOT, DIR, [line.split("|") for line in rows])\n'),
        ('    out.mkdir(mode=0o700)\n', '    out.mkdir(mode=0o700)\n    save(out/"build-authority.json", authority)\n'),
        ('dict(publication=publication, source_manifest_SHA256=manifest, epoch=epoch,',
         'dict(request=REQUEST, publication=publication, source_manifest_SHA256=manifest, epoch=epoch,'),
        ('        return receipt\n',
         '        until = min(deadline - 3, time.time() + 12)\n'
         '        while time.time() < until:\n'
         '            rows = [row for row in accounting(claim, timeout=min(4, until-time.time()))\n'
         '                    if row[0] == receipt["actual_CPU"]]\n'
         '            need(len(rows) <= 1, "ambiguous known probe accounting")\n'
         '            if rows and rows[0][2].split()[0].rstrip("+") in TERMINAL:\n'
         '                terminal(rows[0], receipt["actual_CPU"], label + "-" + suffix)\n'
         '                receipt["accounting"] = rows[0]\n'
         '                return receipt\n'
         '            time.sleep(0.5)\n'
         '        raise TimeoutError("actual CPU probe terminal not confirmed")\n'),
        ('and "gres/dcu=1" in row[6].split(","), "wrong actual GPU handle")',
         'and allocation(row[6], gpu=True), "wrong actual GPU handle")'),
        ('                owned += [{k: r[k]', '                owned.append(dict(PID=worker["worker_pid"], startticks=worker["worker_startticks"]))\n'
         '                owned += [{k: r[k]'),
        ('worker["complete"] is True\n', 'worker["complete"] is True and worker["bitwise_equal_all_pairs"] is True\n'),
    ])


def worker(text):
    start = text.index('        library = Path(claim["component"]["path"])')
    stop = text.index('        original = ROOT /', start)
    return edit(text, [
        ('import time\n', 'import time\n\nfrom authority import DSO, ORDER, PARENT, REQUEST, runtime_authority\n'),
        ('ORDER = (0, 1, 1, 0, 0, 1)\n', ''),
        ('    pci = address.value.decode("ascii").lower()',
         '    pci = address.value.decode("ascii").lower()\n'
         '    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\\.[0-7]", pci):\n'
         '        pci = "0000:" + pci'),
        ('    result = dict(step=', '    owner_ticks = int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])\n'
         '    result = dict(worker_startticks=owner_ticks, request=REQUEST, step='),
        ('        result["physical_ID"], result["PCI"] = physical_identity(pre, claim)\n',
         '        if (claim["request"] != REQUEST or claim["libraries"] != DSO or os.environ["SLURM_JOB_ID"] != PARENT\n'
         '            or pre["node"] != claim["node"] or pre["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()\n'
         '            or not 0 <= time.monotonic_ns() - pre["monotonic_ns"] <= 45_000_000_000):\n'
         '            raise ValueError("GPU worker differs from the fresh physical admission")\n'),
        (text[start:stop],
         '        authority = json.loads((out / "build-authority.json").read_bytes())\n'
         '        result["runtime_providers"] = runtime_authority(authority, os.environ.get("LD_PRELOAD"), remaining)\n'
         '        result["physical_ID"], result["PCI"] = physical_identity(pre, claim)\n'),
        ('        for ordinal, mode in enumerate(ORDER):',
         '        for ordinal, arm in enumerate(ORDER):\n'
         '            mode = 2\n'
         '            component = DSO[arm]\n'
         '            library = Path(component["path"])'),
        ('            rec = {"ordinal": ordinal, "mode": mode, "complete": False}',
         '            rec = {"ordinal": ordinal, "arm": arm, "mode": mode, "complete": False,\n'
         '                   "component_SHA256": component["SHA256"]}'),
        ('                                          "--mode", str(mode)], stdout=log, stderr=subprocess.STDOUT)',
         '                                          "--mode", str(mode), "--component-sha256", component["SHA256"],\n'
         '                                          "--input-receipt-sha256", claim["original_input_receipt_SHA256"]],\n'
         '                                         stdout=log, stderr=subprocess.STDOUT)'),
        ('            if startup is None or report["startup"] != startup or not (',
         '            if (report["component_SHA256"] != component["SHA256"] or\n'
         '                report["original_input_receipt_SHA256"] != claim["original_input_receipt_SHA256"]):\n'
         '                raise ValueError("worker receipt lost the actual DSO or original input identity")\n'
         '            if startup is None or startup["PID"] != rec["PID"] or report["startup"] != startup or not ('),
        ('old, tiled = sorted(result["records"][i:i+2], key=lambda arm: arm["mode"])',
         'old, tiled = sorted(result["records"][i:i+2], key=lambda record: record["arm"])'),
        ('                row["baseline_ordinal"], row["tiled_ordinal"] = old["ordinal"], tiled["ordinal"]',
         '                row["wave32_SHA256"] = row.pop("tiled_SHA256")\n'
         '                row["full_tree_ordinal"], row["wave32_ordinal"] = old["ordinal"], tiled["ordinal"]'),
        ('all(v["changed_words"] == 0 for v in result["pairs"])',
         'len(result["pairs"]) == 3 and all(v["changed_words"] == 0 for v in result["pairs"])'),
    ])


def arm(text):
    return edit(text, [
        ('import time\n', 'import time\n\nfrom authority import DSO\n'),
        ('choices=(0, 1), required=True)', 'choices=(2,), required=True)'),
        ('    args = parser.parse_args()', '    parser.add_argument("--component-sha256", required=True)\n'
         '    parser.add_argument("--input-receipt-sha256", required=True)\n'
         '    args = parser.parse_args()'),
        ('    binding = json.loads(args.binding.read_bytes())',
         '    raw_binding = args.binding.read_bytes()\n'
         '    if hashlib.sha256(raw_binding).hexdigest() != args.input_receipt_sha256:\n'
         '        raise ValueError("exact original input receipt differs")\n'
         '    binding = json.loads(raw_binding)'),
        ('!= binding["component_SHA256"]:', '!= args.component_sha256:'),
        ('    library_path = args.library.resolve(strict=True)\n',
         '    library_path = args.library.resolve(strict=True)\n'
         '    if not any(library_path.as_posix() == record["path"] and args.component_sha256 == record["SHA256"]\n'
         '               for record in DSO.values()):\n'
         '        raise ValueError("actual qualified DSO path or SHA differs")\n'),
        ('"complete": False, "binding": binding}',
         '"complete": False, "binding": binding, "component_SHA256": args.component_sha256,\n'
         '              "original_input_receipt_SHA256": args.input_receipt_sha256}'),
    ])


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def materialize(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    members = {name: pinned(OLD / name, pin) for name, pin in SOURCE.items()}
    for name, size, pin, source in EVIDENCE:
        members[name] = pinned(REPO / "results" / "bw1000" / source, pin, size)
    for name, change in (("controller.py", controller), ("worker.py", worker), ("run_arm.py", arm)):
        members[name] = change(members[name].decode("utf8")).encode("utf8")
    members["inside.sh"] = edit(members["inside.sh"].decode(), [(OLD.name, HERE.name)]).encode()
    members["probe.py"] = edit(members["probe.py"].decode(), [
        ("import sys\n", "import sys\nimport time\n"),
        ('        "PCI": pci,', '        "monotonic_ns": time.monotonic_ns(), "PCI": pci,'),
        ('{2}.[0-7]', '{2}[.][0-7]'),
    ]).encode()
    claim = json.loads(members["CLAIM.json"])
    for key in ("component", "new_scope", "startup_observation"):
        del claim[key]
    claim.update(label=LABEL, request=REQUEST, node=NODE, parent=PARENT, libraries=DSO,
                 arm_order=ORDER, original_input_receipt_SHA256=INPUT_SHA256,
                 scope="Frozen r461 mode2 C ABI on one allocated DCU with the same input and original weights; only the linked score object differs.",
                 numeric_rule="Keep six planned arms, compare all words across each pair, require all three equal and clean terminal.",
                 timing_rule="Independent startup and first-use synchronized dispatch. Retain raw arm timings; no steady-state or model speed claim.")
    members["CLAIM.json"] = canonical(claim)
    members["authority.py"] = (HERE / "authority.py").read_bytes()
    manifest = canonical(dict(files={name: sha256(data).hexdigest() for name, data in members.items()}))
    with TemporaryDirectory(dir=output.parent, prefix="kimi-two-dso-") as temp:
        staged = Path(temp) / "package"
        staged.mkdir(mode=0o700)
        for name, data in dict(members, **{"SOURCE_FILES.json": manifest}).items():
            if name.endswith(".py"):
                compile(data, name, "exec")
            with (staged / name).open("xb") as stream:
                stream.write(data)
        staged.rename(output)
    return dict(files=len(members) + 1, SOURCE_FILES_SHA256=sha256(manifest).hexdigest(),
                CLAIM_SHA256=sha256(members["CLAIM.json"]).hexdigest(), request=REQUEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True))

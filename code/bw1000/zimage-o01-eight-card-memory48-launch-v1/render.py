"""Derive the hosted 48 GiB witness and single zero-GRES evidence sidecar."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory_link as new


OLD = HERE.with_name("zimage-lt-eight-card-root-launch-v1")
OLD_REQUEST, OLD_LABEL = "e23d24ea0c9c4d00af06154d82dc22e8", "O01-zimage-lt-eight-card-supervisor-v1"
OLD_PRE, OLD_POST = "f0674bbd407942a4a2455634c8cb44c4", "a57632cbc5534cf781abbde00746a7a9"
OLD_STEP = "1103606.2075"
OLD_STANDBY = "3228d0f2042f4224bb72bdf2313617c5"
OLD_RAM = HERE.with_name("zimage-o01-eight-card-host-memory-trace-v1") / "ram_trace.py"
MEMORY_READER = HERE.with_name("zimage-o01-standby2179-host-memory-observer-v1") / "observer.py"
ROOT_PINS = {
    "render.py": "18556e49e193884e6a88ddb87e1ce6d27187a5073ec3fa63d1debddbd5c667d1",
    "host_run.py": "939ab420f4ed76e84d0e3c4b4c1ebc18e00f2de011e83cb95f107be2216b21b0",
    "scheduler.py": "a5525be7ef8893e065065ee1f4fbc6704c1cbf339320d65844a99ab462ad5198",
}
PINS = {
    "runtime": "7ba1d9355838c09d41176f67708d892528832971f83f036be63d6ddacbbdb35d",
    "attested": "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6",
    "device_host": "402fce88ebf96b57ed474059e7bc0c080dc377abfebb52dce6447c85336a4259",
    "host_scope": "a50bfad3b2e2a7616c464b44ddd5adacbabb8979e79bf94e2c64ae15963d811f",
    "old_bridge": "785ed7e948c0484243c7c325eb470d4555b6304e24759667f7b1289f9d67676f",
    "ram_trace": "83cfd41ed9239e19100fbde87d0092d18f67e7f6d9dac2512f57dd60722542a8",
    "reader": "bdff67d465583fef518b304dc50979c9bf03f05238f428d325e8ff152ce10c0e",
}


def pin(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def exact(path, expected):
    return require_bytes(path.read_bytes(), expected, path.name)


def require_bytes(raw, expected, label):
    if pin(raw)["sha256"] != expected:
        raise ValueError("Frozen predecessor differs: " + label)
    return raw


def patch(text, old, value):
    if text.count(old) != 1:
        raise ValueError("Frozen source anchor differs: " + old[:120])
    return text.replace(old, value, 1)


def reidentify(text):
    for prior, next_ in ((OLD_REQUEST, new.REQUEST), (OLD_LABEL, new.LABEL), (OLD_PRE, new.PRE),
                         (OLD_POST, new.POST), (OLD_STEP, new.PREDECESSOR_STEP), (OLD_STANDBY, new.PREDECESSOR_REQUEST)):
        text = text.replace(prior, next_)
    return text.replace("training80-dcu-standby-3228d0f2042f-f12r2n11", new.PREDECESSOR_LABEL)


def encoded(value):
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def baseline():
    path = OLD / "render.py"
    renderer = ModuleType("published_o01_root_renderer")
    renderer.__file__ = str(path)
    exec(compile(exact(path, ROOT_PINS["render.py"]), str(path), "exec"), vars(renderer))
    with TemporaryDirectory() as temp:
        target = Path(temp) / "base"
        document = renderer.build(target)
        if len(document["files"]) != 18:
            raise ValueError("Unexpected original file set")
        members = {name: (target / name).read_bytes() for name in document["files"]}
    for name in ("host_run.py", "scheduler.py"):
        require_bytes(members[name], ROOT_PINS[name], name)
    return members


def render_runtime(members):
    require_bytes(members["runtime/RUNTIME_FILES.json"], PINS["runtime"], "old runtime")
    super_ = reidentify(members["runtime/supervisor.py"].decode())
    super_ = patch(super_, "import linux_owner as owned\n", "import linux_owner as owned\nimport ram_trace\n")
    super_ = patch(super_, '"standby_worker.py", "witness.py", "contract.py"}',
                   '"standby_worker.py", "witness.py", "contract.py", "ram_trace.py"}')
    super_ = patch(super_, "    def wait_until(self, probe, seconds, latest, *, honor_stop=True):",
                   '    def phase(self, name):\n'
                   '        try:\n'
                   '            ram_trace.record_phase(self.args.out / "memory-phases", self.scope, '
                   'self.args.scope_sha256, self.step, name)\n'
                   '        except (OSError, ValueError) as error:\n'
                   '            self.results.setdefault("host_RAM_phase_errors", []).append('
                   '{"phase": name, "type": type(error).__name__})\n\n'
                   '    def wait_until(self, probe, seconds, latest, *, honor_stop=True):')
    anchors = (
        ("    step.inventory()\n", "inventory_enter", False),
        ("    all_cards = step.start_standbys()\n", "cold_standbys_enter", False),
        ('    report["initial_eight_growth"] = step.growth(all_cards, 30, 140)\n', "eight_standbys_ready", True),
        ('    report["physical_pre_witness"] = step.clean_target(siblings)\n', "seven_standbys_ready", True),
        ('    report["witness"] = step.witness(siblings)\n', "witness_enter", False),
        ('    report["witness"] = step.witness(siblings)\n', "witness_return", True),
        ('    all_cards[TARGET] = step.restore()\n', "restore_enter", False),
        ('    report["restored_eight_proved"] = True\n', "eight_restored", True),
    )
    for anchor, marker, after in anchors:
        call = f'    step.phase("{marker}")\n'
        super_ = patch(super_, anchor, anchor + call if after else call + anchor)
    super_ = patch(super_, '        runtime = Step(args, scope, step, report)\n',
                   '        (args.out / "memory-phases").mkdir(mode=0o700)\n'
                   '        report["host_RAM_baseline"] = ram_trace.await_baseline(\n'
                   '            args.out.parent / "ram-observer/ready.json", scope, args.scope_sha256, step)\n'
                   '        runtime = Step(args, scope, step, report)\n')
    super_ = patch(super_, '                runtime.cleanup()\n',
                   '                runtime.phase("cleanup_enter")\n'
                   '                runtime.cleanup()\n'
                   '                runtime.phase("cleanup_return")\n')
    super_ = patch(super_, '    return int("error" in report or not report["local_process_cleanup"]',
                   '    return int("error" in report or bool(report.get("host_RAM_phase_errors")) '
                   'or not report["local_process_cleanup"]')
    members["runtime/supervisor.py"] = super_.encode()
    members["runtime/witness.py"] = reidentify(members["runtime/witness.py"].decode()).encode()
    members["runtime/ram_trace.py"] = exact(OLD_RAM, PINS["ram_trace"])
    claim = json.loads(reidentify(members["runtime/CLAIM.json"].decode()))
    claim["resources"]["host_RAM_MiB"] = new.GPU_MIB
    claim["host_RAM_sidecar_label"] = new.OBSERVER_LABEL
    members["runtime/CLAIM.json"] = encoded(claim)
    listing = {"request": new.REQUEST, "label": new.LABEL,
               "files": {name.split("/", 1)[1]: pin(raw) for name, raw in members.items()
                         if name.startswith("runtime/") and not name.endswith("/RUNTIME_FILES.json")}}
    members["runtime/RUNTIME_FILES.json"] = encoded(listing)
    return pin(members["runtime/RUNTIME_FILES.json"])["sha256"]


def render_entry(members, runtime_sha):
    original = require_bytes(members["node_entry.py"], PINS["attested"], "old attested entry")
    members["identity_original.py"] = original
    node = reidentify(original.decode())
    node = patch(node, PINS["runtime"], runtime_sha)
    node = patch(node, '    require(matched, "actual Slurm task-process membership differs")\n',
                 '    require(matched, "actual Slurm task-process membership differs")\n'
                 '    require(scope["host_memory_MiB"] == 49152 and '
                 '({"mem=48G", "mem=49152M"} & set(row["TRES"].split(","))), '
                 '"actual Slurm host RAM differs")\n')
    members["node_entry.py"] = node.encode()
    entry_sha = pin(members["node_entry.py"])["sha256"]
    device = require_bytes(members["device_host.py"], PINS["device_host"], "old device host").decode()
    members["device_host.py"] = patch(device, PINS["attested"], entry_sha).encode()
    return entry_sha, pin(members["device_host.py"])["sha256"]


def render_host(members, runtime_sha, entry_sha, device_sha):
    old = require_bytes(members["host_scope.py"], PINS["host_scope"], "old host scope")
    source = reidentify(old.decode()).replace("24576M", "49152M").replace('"24G"', '"48G"')
    source = patch(source, "import time\n", "import time\nimport sys\n\n"
                   "sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport memory_link\n")
    for previous, actual in (("runtime", runtime_sha), ("attested", entry_sha), ("device_host", device_sha)):
        source = patch(source, PINS[previous], actual)
    source = patch(source, '"allocated_DCU": 8, "owner_uid": uid,',
                   f'"allocated_DCU": 8, "owner_uid": uid, "host_memory_MiB": {new.GPU_MIB}, '
                   f'"memory_observer_label": {new.OBSERVER_LABEL!r},')
    source = patch(source, '    read_exact(stage / "node_inventory.py", INVENTORY_SHA)\n',
                   '    read_exact(stage / "node_inventory.py", INVENTORY_SHA)\n'
                   f'    read_exact(stage / "identity_original.py", {PINS["attested"]!r})\n'
                   f'    read_exact(stage / "memory_reader.py", {PINS["reader"]!r})\n'
                   f'    read_exact(stage / "memory_link.py", {pin(members["memory_link.py"])["sha256"]!r})\n')
    source = source.replace("    gpu = exact_row(records, LABEL)\n",
                            "    gpu, role = memory_link.terminal_reference(records, scope, RUN, exact_row, check_resources)\n")
    source = source.replace('    check_resources(gpu[6], 12, 8, {"48G", "49152M"})\n', "")
    source = patch(source, '    if attested is not None:\n        require(adopt_bound_owner(attested, scope) == gpu[0],',
                   '    if attested is not None:\n        require(role == "GPU_STEP_TERMINAL" and '
                   'adopt_bound_owner(attested, scope) == gpu[0],')
    source = patch(source, '    return gpu, render_inventory("post")',
                   '    return gpu, role, render_inventory("post")')
    source = patch(source, "    row, argv = post_after_terminal(scheduler, scope, time.time(), os.getuid(), attested=attested)",
                   "    row, role, argv = post_after_terminal(scheduler, scope, time.time(), os.getuid(), attested=attested)")
    source = patch(source, '"gpu_request": REQUEST, "gpu_accounting": row,',
                   '"gpu_request": REQUEST, "gpu_accounting": row, "terminal_reference_role": role,\n'
                   '                                   "GPU_step_allocated": role == "GPU_STEP_TERMINAL",')
    source = patch(source, '"post_observed_epoch": evidence["independent_post"]["observed_epoch"], "GPU_accounting": gpu,',
                   '"post_observed_epoch": evidence["independent_post"]["observed_epoch"], '
                   '"GPU_accounting": gpu, "terminal_reference_role": role,\n'
                   '            "GPU_step_allocated": role == "GPU_STEP_TERMINAL",')
    source = patch(source, "    names = (PRE_LABEL, LABEL, POST_LABEL)\n",
                   "    names = (PRE_LABEL, LABEL, POST_LABEL, memory_link.OBSERVER_LABEL)\n")
    members["host_scope.py"] = source.encode()
    original = members["pending.py"]
    members["pending_original.py"] = original
    pending = reidentify(original.decode())
    pending = patch(pending, PINS["host_scope"], pin(members["host_scope.py"])["sha256"])
    pending = patch(pending, "o01-pending-decision.json", "o01-memory48-pending-decision.json")
    members["pending.py"] = pending.encode()
    bridge = require_bytes(members["guarded-standby-bridge.py"], PINS["old_bridge"], "old guarded bridge").decode()
    hook = '    pending_namespace["install"](namespace, inputs, directory)\n'
    bridge = patch(bridge, hook, hook +
                   '    memory_namespace = {"__name__": "o01_memory48_pending_guard"}\n'
                   f'    exec(compile({members["pending.py"]!r}, "o01-memory48-pending.py", "exec"), memory_namespace)\n'
                   '    memory_namespace["install"](namespace, inputs, directory)\n')
    members["guarded-standby-bridge.py"] = bridge.encode()
    runner = reidentify(members["host_run.py"].decode()).replace("24576M", "49152M").replace('"24G"', '"48G"')
    runner = patch(runner, "import scheduler\n", "import scheduler\nimport memory_link\nimport pending_original\n")
    runner = patch(runner, PINS["old_bridge"], pin(members["guarded-standby-bridge.py"])["sha256"])
    runner = patch(runner, "    return run_gpu(scope, argv)\n",
                   "    return memory_link.launch_pair(scope, argv, spec, scheduler, run_gpu, safe_environment)\n")
    runner = patch(runner, '{spec.OLD_LABEL, spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL}',
                   '{spec.OLD_LABEL, spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL, memory_link.OBSERVER_LABEL}')
    runner = patch(runner, '            "supervisor_terminal": terminal_summary(owner),\n',
                   '            "supervisor_terminal": terminal_summary(owner),\n'
                   '            "host_RAM_observer": memory_link.observer_status(spec.RUN),\n')
    runner = patch(runner, '            "supervisor_error_type": record.get("error", {}).get("type"),\n',
                   '            "supervisor_error_type": record.get("error", {}).get("type"),\n'
                   '            "host_RAM_baseline": record.get("host_RAM_baseline"),\n'
                   '            "host_RAM_phase_errors": record.get("host_RAM_phase_errors", []),\n')
    runner = patch(runner, '            labels = (spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL)\n',
                   '            labels = (spec.PRE_LABEL, spec.LABEL, spec.POST_LABEL, memory_link.OBSERVER_LABEL)\n')
    runner = patch(runner, '            pin = pending.reserve_locked(Path(spec.RESULTS), held)\n',
                   '            spec.require(not pending_original.decision(Path(spec.RESULTS), spec.REQUEST, '
                   '{"parent": spec.PARENT, "pci": sorted(spec.PCI)})["blocked"], '
                   '"PREDECESSOR_GUARD_UNRESOLVED")\n'
                   '            pin = pending.reserve_locked(Path(spec.RESULTS), held)\n')
    runner = patch(runner, '        result = run_inventory("post", argv)\n',
                   '        result = run_inventory("post", argv)\n'
                   '        post = json_file(intent("post"))\n'
                   '        result.update(terminal_reference_role=post["terminal_reference_role"], '
                   'GPU_step_allocated=post["GPU_step_allocated"])\n')
    members["host_run.py"] = runner.encode()
    scheduler = members["scheduler.py"].decode()
    scheduler = patch(scheduler, 'int(node["RealMemory"]) >= 24576\n'
                      '                 and int(node["AllocMem"]) >= 24576 and int(node["CPUAlloc"]) >= 12',
                      'int(node["RealMemory"]) >= 49408\n'
                      '                 and int(node["AllocMem"]) >= 49408 and int(node["CPUAlloc"]) >= 13')
    members["scheduler.py"] = scheduler.encode()


def render():
    members = baseline()
    members["memory_link.py"] = (HERE / "memory_link.py").read_bytes()
    members["memory_reader.py"] = exact(MEMORY_READER, PINS["reader"])
    runtime_sha = render_runtime(members)
    entry_sha, device_sha = render_entry(members, runtime_sha)
    render_host(members, runtime_sha, entry_sha, device_sha)
    for name, raw in members.items():
        if name.endswith(".py"):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    manifest = {"original_request": new.REQUEST, "runtime_manifest_SHA256": runtime_sha,
                "files": {name: pin(raw) for name, raw in members.items()}}
    return {**members, "HOST_STAGE_FILES.json": encoded(manifest)}


def build(output):
    members = render()
    output.mkdir(mode=0o700)
    (output / "runtime").mkdir(mode=0o700)
    for name, raw in members.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return pin(members["HOST_STAGE_FILES.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(build(parser.parse_args().output), sort_keys=True))

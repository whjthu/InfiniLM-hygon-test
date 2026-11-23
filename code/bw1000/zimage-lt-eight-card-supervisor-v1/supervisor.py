"""Run the untimed Z-Image route witness within one independently admitted eight-card step."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import linux_owner as owned

REQUEST, LABEL = "e23d24ea0c9c4d00af06154d82dc22e8", "O01-zimage-lt-eight-card-supervisor-v1"
PARENT, NODE, FORBIDDEN = "1103606", "f12r2n11", "1103606.2075"
TARGET = "0000:09:00.0"
GPU_SET = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
STANDBY_SET = GPU_SET - {TARGET}
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
HEARTBEAT_AGE = 12
STOP_REQUESTED = False
RUNTIME_MEMBERS = {"CLAIM.json", "supervisor.py", "linux_owner.py", "mask_preflight.py",
                   "standby_worker.py", "witness.py", "contract.py"}


def require(test, boundary):
    if not test:
        raise ValueError(boundary)


def read(path, maximum):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, "RECEIPT_SIZE")
    return raw


def load_pinned(path, pin, maximum):
    raw = read(path, maximum)
    require(sha256(raw).hexdigest() == pin, "INPUT_SHA256")
    return raw


def save(path, value):
    raw = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(raw).hexdigest()


def physical_admission(scope):
    first, end = scope["epoch"], scope["deadline_epoch"]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    boot = read(Path("/proc/sys/kernel/random/boot_id"), 128)
    require(scope["request"] == REQUEST and scope["label"] == LABEL and scope["fresh_root_admitted"] is True
            and re.fullmatch(PARENT + r"\.[0-9]+", step) and step != FORBIDDEN and scope["parent"] == PARENT
            and socket.gethostname().split(".")[0] == scope["node"] == NODE
            and os.environ["SLURM_CPUS_PER_TASK"] == "12"
            and scope["allocated_DCU"] == len(scope["PCI"]) == 8 and set(scope["PCI"]) == GPU_SET
            and sha256(boot).hexdigest() == scope["boot_id_SHA256"]
            and re.fullmatch(PARENT + r"\.[0-9]+", scope["independent_pre_step"])
            and scope["independent_pre_step"] not in (step, FORBIDDEN)
            and re.fullmatch("[0-9a-f]{64}", scope["independent_pre_SHA256"]), "EIGHT_CARD_STEP")
    require(all(type(x) in (float, int) and math.isfinite(x) for x in (first, end))
            and first <= time.time() < end - 8 and end == first + 240, "ORIGINAL_EPOCH")
    return step


def verify_runtime(pin):
    manifest = json.loads(load_pinned(HERE / "RUNTIME_FILES.json", pin, 16 << 10))
    require(manifest["request"] == REQUEST and set(manifest["files"]) == RUNTIME_MEMBERS, "RUNTIME_MEMBERS")
    for name, row in manifest["files"].items():
        require(len(load_pinned(HERE / name, row["sha256"], 64 << 10)) == row["bytes"], "RUNTIME_MEMBER_SIZE")


def read_pulse(child, reference_epoch, now, *, active):
    try:
        row = json.loads(read(child.directory / "card-0.json", 2048))
    except FileNotFoundError:
        return None
    require(row["pci"] == child.pci and row["ready"] is True and row["dimension"] == 8192
            and row["dtype"] == "BF16" and row["synthetic"] is True
            and type(row["calls"]) is int and row["calls"] > 0 and row["calls"] % 8 == 0
            and reference_epoch <= row["epoch"] <= now + 1, "STANDBY_PULSE_IDENTITY")
    if row["active"] is not active or now - row["epoch"] > HEARTBEAT_AGE:
        return None
    return {key: row[key] for key in ("pci", "epoch", "calls", "active")}


def wrapper_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def subordinate(args, scope, step):
    # Imports and HIP initialization occur only in this short inventory or in a masked role.
    import mask_preflight as mask

    require(os.getppid() == args.owner and os.getpid() == os.getpgrp() == os.getsid(0), "CHILD_OWNER")
    before = os.sched_getaffinity(0)
    require(len(before) == 12 and args.cpu and set(args.cpu) <= before
            and len(args.cpu) == len(set(args.cpu)) == (4 if args.role == "witness" else 1), "CPU_ROLE_BOUNDARY")
    os.sched_setaffinity(0, args.cpu)
    require(os.sched_getaffinity(0) == set(args.cpu), "CPU_ROLE_AFFINITY")
    if args.role == "inventory":
        save(args.out / "inventory.json", {"step": step, "PCI": mask.visible(8),
                                          "selectors": mask.selectors(os.environ)})
        return 0
    require(args.pci in GPU_SET and os.environ.get("HIP_VISIBLE_DEVICES") == args.token
            and os.environ.get("CUDA_VISIBLE_DEVICES") == args.token and mask.visible(1) == [args.pci],
            "MASKED_ROLE_PCI")
    if args.role == "witness":
        require(args.pci == TARGET, "FIXED_WITNESS_PCI")
        scope_file = args.out / "witness-scope.json"
        path = Path("/proc/sys/kernel/random/boot_id")
        witness_scope = {key: scope[key] for key in ("request", "label", "node", "parent", "epoch", "deadline_epoch")}
        witness_scope.update(root_admitted=True, pre_step=scope["independent_pre_step"], PCI=args.pci,
                             boot_id=read(path, 128).decode("ascii").strip(), torch_version="2.10.0",
                             torch_hip="6.3.26113")
        pin = save(scope_file, witness_scope)
        os.execv(sys.executable, [sys.executable, "-B", str(HERE / "witness.py"), "--scope", str(scope_file),
                                 "--scope-sha256", pin, "--out", str(args.out / "worker-output")])
    import standby_worker

    signal.signal(signal.SIGTERM, standby_worker.stop)
    signal.signal(signal.SIGINT, standby_worker.stop)
    import torch

    torch.set_num_threads(1)
    require(torch.__version__ == "2.10.0" and torch.version.hip == "6.3.26113"
            and torch.cuda.device_count() == 1 and mask.visible(1) == [args.pci]
            and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx936",
            "STANDBY_HIP_TORCH_PCI")
    hip_maps = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
                if "/libgalaxyhip.so" in line or "/libamdhip64.so" in line}
    require(hip_maps == {mask.HIP}, "STANDBY_ACTIVE_HIP_PROVIDER")
    save(args.out / "binding.json", {"PID": os.getpid(), "step": step, "PCI": args.pci, "cpu": args.cpu,
                                    "HIP_SHA256": mask.HIP_SHA, "torch": torch.__version__,
                                    "torch_hip": torch.version.hip, "epoch": time.time()})
    require(time.time() < scope["deadline_epoch"] - 8, "STANDBY_TIME_BOUND")
    standby_worker.run_device(torch, 0, args.pci, args.out,
                              time.monotonic() + (scope["deadline_epoch"] - 8 - time.time()))
    return 0


class Step:
    def __init__(self, args, scope, step, results):
        self.args, self.scope, self.step, self.results = args, scope, step, results
        self.children = {}
        self.epoch = scope["epoch"]
        self.cpus = sorted(os.sched_getaffinity(0))
        require(len(self.cpus) == 12, "STEP_CPU_COUNT")
        self.end = time.monotonic() + scope["deadline_epoch"] - time.time()
        owned.become_subreaper()
        os.sched_setaffinity(0, self.cpus[11:])

    def wait_until(self, probe, seconds, latest, *, honor_stop=True):
        end = min(time.monotonic() + seconds, self.end - latest)
        while time.monotonic() < end:
            require(not honor_stop or not STOP_REQUESTED, "COOPERATIVE_STEP_STOP")
            value = probe()
            if value:
                return value
            time.sleep(0.2)
        raise ValueError("BOUNDED_PHASE_EXPIRED")

    def launch(self, key, role, cpus, pci=None):
        require(key not in self.children, "DUPLICATE_ROLE")
        directory = self.args.out / key
        directory.mkdir(mode=0o700)
        command = [sys.executable, "-B", str(HERE / "supervisor.py"), "--scope", str(self.args.scope),
                   "--scope-sha256", self.args.scope_sha256, "--runtime-sha256", self.args.runtime_sha256,
                   "--out", str(directory), "--role", role, "--owner", str(os.getpid()), "--cpu", *map(str, cpus)]
        environment = dict(os.environ, OMP_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)))
        event = {"role": role, "PCI": pci, "CPU": cpus, "start_epoch": time.time()}
        if pci:
            token = self.tokens[pci]
            command.extend(["--pci", pci, "--token", token])
            environment.update(HIP_VISIBLE_DEVICES=token, CUDA_VISIBLE_DEVICES=token)
            event["selector"] = token
        self.results["processes"][key] = event
        # Each subprocess starts from the verified Slurm mask, then narrows its own role.
        os.sched_setaffinity(0, self.cpus)
        try:
            child = owned.spawn(command, directory, role, pci, environment, event)
            self.children[key] = child
        finally:
            os.sched_setaffinity(0, self.cpus[11:])
        return child

    def inventory(self):
        child = self.launch("inventory", "inventory", self.cpus[11:])
        self.wait_until(child.exited, 15, 185)
        require(child.retire(min(time.monotonic() + 2, self.end - 185)) == 0, "FULL_PCI_INVENTORY")
        self.children.pop("inventory")
        inventory = json.loads(read(child.directory / "inventory.json", 4096))
        devices, selectors = inventory["PCI"], inventory["selectors"]
        require(inventory["step"] == self.step and len(devices) == len(selectors) == len(set(selectors)) == 8
                and set(devices) == GPU_SET, "PARENT_PCI_SELECTOR_BIJECTION")
        owned.audit_no_other_children({})
        self.tokens = dict(zip(devices, selectors))
        self.results["parent_PCI_selectors"] = self.tokens

    def start_standbys(self):
        return {pci: self.launch("standby-" + pci[5:7], "standby", [self.cpus[index]], pci)
                for index, pci in enumerate([*sorted(STANDBY_SET), TARGET])}

    def observe(self, cards):
        now = time.time()
        out = {}
        for pci, child in cards.items():
            require(not child.exited(), "STANDBY_EXITED")
            row = read_pulse(child, child.event["start_epoch"], now, active=True)
            if row is not None:
                if "binding_SHA256" not in child.event:
                    raw = read(child.directory / "binding.json", 2048)
                    binding = json.loads(raw)
                    require(binding["PID"] == child.identity.pid and binding["step"] == self.step
                            and binding["PCI"] == pci and binding["cpu"] == child.event["CPU"]
                            and binding["HIP_SHA256"] == HIP_SHA and binding["torch"] == "2.10.0"
                            and binding["torch_hip"] == "6.3.26113"
                            and child.event["start_epoch"] <= binding["epoch"] <= row["epoch"],
                            "STANDBY_BINDING_RECEIPT")
                    child.event["binding_SHA256"] = sha256(raw).hexdigest()
                out[pci] = row
        return out

    def growth(self, cards, seconds, latest):
        first = self.wait_until(lambda: self.ready(cards), seconds, latest)

        def increased():
            second = self.ready(cards)
            if second and all(second[pci]["calls"] > first[pci]["calls"]
                              and second[pci]["epoch"] > first[pci]["epoch"] for pci in cards):
                return {"first": first, "second": second}
            return None

        return self.wait_until(increased, seconds, latest)

    def ready(self, cards):
        actual = self.observe(cards)
        return actual if set(actual) == set(cards) else None

    def stop_target(self, target):
        target.signal(signal.SIGTERM)
        require(target.retire(min(time.monotonic() + 6, self.end - 115)) == 0, "TARGET_STANDBY_EXIT")
        self.children.pop("standby-09")
        row = read_pulse(target, target.event["start_epoch"], time.time(), active=False)
        require(row, "TARGET_STOP_RECEIPT")
        return row

    def clean_target(self, siblings):
        return owned.audit_device_holders(siblings, GPU_SET)

    def witness(self, siblings):
        child = self.launch("witness", "witness", self.cpus[7:11], TARGET)

        def complete():
            require(self.ready(siblings), "SIBLING_PULSE_DURING_WITNESS")
            return child.exited()

        try:
            self.wait_until(complete, 90, 40)
        except ValueError as error:
            self.results["witness_wait_error"] = str(error)
            child.signal(signal.SIGTERM)
        try:
            rc = child.retire(min(time.monotonic() + 4, self.end - 32))
        except ValueError:
            child.signal(signal.SIGKILL)
            rc = child.retire(min(time.monotonic() + 2, self.end - 30))
        self.children.pop("witness")
        terminal = child.directory / "worker-output" / "terminal.json"
        row = {"returncode": rc, "terminal_SHA256": None, "qualified_operator_route": False}
        if terminal.is_file():
            try:
                raw = read(terminal, 32 << 10)
                row["terminal_SHA256"] = sha256(raw).hexdigest()
                value = json.loads(raw)
                require(value["request"] == REQUEST and value["step"] == self.step and value["PCI"] == TARGET
                        and value["timed_calls"] == 0 and value["planned_maximum_solutions"] == 1,
                        "UNTIMED_WITNESS_SCOPE")
                row.update(decision=value.get("decision"),
                           qualified_operator_route=(rc == 0 and value["passed"] is True and value["completed"] is True))
            except (OSError, ValueError, KeyError, TypeError):
                row["terminal_invalid"] = True
        return row

    def restore(self):
        return self.launch("restored-09", "standby", [self.cpus[7]], TARGET)

    def hold(self, all_cards):
        def done():
            require(self.ready(all_cards), "RESTORED_EIGHT_CARD_PULSES")
            return time.monotonic() >= self.end - 17

        self.wait_until(done, 240, 16)

    def hold_degraded(self):
        if self.children and not STOP_REQUESTED:
            # Leave known, running standby sessions in place while an unresolved PCI is reserved.
            while time.monotonic() < self.end - 17 and not STOP_REQUESTED:
                active = [child for child in self.children.values() if child.role == "standby" and not child.exited()]
                if not active:
                    break
                time.sleep(0.2)
            self.results["degraded_hold_did_not_qualify_coverage"] = True

    def cleanup(self):
        for child in self.children.values():
            if not child.exited():
                child.signal(signal.SIGTERM)
        graceful = min(time.monotonic() + 4, self.end - 4)
        for key, child in list(self.children.items()):
            try:
                child.retire(graceful)
                self.children.pop(key)
            except ValueError:
                child.signal(signal.SIGKILL)
        forced = min(time.monotonic() + 2, self.end - 1)
        for key, child in list(self.children.items()):
            child.retire(forced)
            self.children.pop(key)
        owned.audit_no_other_children({})


def cycle(step, report):
    step.inventory()
    all_cards = step.start_standbys()
    report["initial_eight_growth"] = step.growth(all_cards, 30, 140)
    siblings = {pci: child for pci, child in all_cards.items() if pci != TARGET}
    report["target_standby_stop"] = step.stop_target(all_cards[TARGET])
    report["seven_growth_before_witness"] = step.growth(siblings, 10, 100)
    report["physical_pre_witness"] = step.clean_target(siblings)
    report["witness"] = step.witness(siblings)
    report["physical_pre_restore"] = step.clean_target(siblings)
    all_cards[TARGET] = step.restore()
    report["restored_eight_growth"] = step.growth(all_cards, 12, 18)
    report["restored_eight_proved"] = True
    step.hold(all_cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--role", choices=("inventory", "standby", "witness"))
    parser.add_argument("--owner", type=int)
    parser.add_argument("--cpu", nargs="+", type=int)
    parser.add_argument("--token")
    parser.add_argument("--pci")
    args = parser.parse_args()
    verify_runtime(args.runtime_sha256)
    scope = json.loads(load_pinned(args.scope, args.scope_sha256, 16 << 10))
    step = physical_admission(scope)
    if args.role:
        return subordinate(args, scope, step)
    args.out.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM, wrapper_signal)
    signal.signal(signal.SIGINT, wrapper_signal)
    report = {"request": REQUEST, "step": step, "scope_SHA256": args.scope_sha256,
              "runtime_SHA256": args.runtime_sha256, "planned_PCI": sorted(GPU_SET), "planned_standbys": 8,
              "planned_witness_candidates": 1, "formal_performance": False, "restored_eight_proved": False,
              "local_process_cleanup": False, "processes": {}}
    runtime = None
    try:
        runtime = Step(args, scope, step, report)
        cycle(runtime, report)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["error"] = {"type": type(error).__name__, "detail": str(error)[:180]}
        try:
            if runtime is not None:
                runtime.hold_degraded()
        except (OSError, ValueError, subprocess.SubprocessError):
            report["degraded_hold_unproved"] = True
    finally:
        try:
            if runtime is not None:
                runtime.cleanup()
                report["local_process_cleanup"] = True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["cleanup_error"] = {"type": type(error).__name__, "detail": str(error)[:180]}
        save(args.out / "terminal.json", report)
    return int("error" in report or not report["local_process_cleanup"] or not report["restored_eight_proved"]
               or not report["witness"]["qualified_operator_route"])


if __name__ == "__main__":
    raise SystemExit(main())

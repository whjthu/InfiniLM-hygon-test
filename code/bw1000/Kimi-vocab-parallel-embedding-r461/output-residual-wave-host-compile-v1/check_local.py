"""Exercise the controller with real rendered source and recorded scheduler/compiler I/O."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

H = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().with_name("host_once.py"))))
CLAIM = sha256((H.HERE / "CLAIM.json").read_bytes()).hexdigest()
MEMBER, TERMINAL, GROUPS, RENDER = H.frozen(CLAIM)
EPOCH = 1795300000.5
GOOD = ["1103606.712", H.LABEL, "COMPLETED", "0:0", "83", H.NODE, "cpu=4,mem=8G,node=1"]
RUNNING = [*GOOD[:2], "RUNNING", *GOOD[3:]]


def recorded_output(out, source, case):
    pins = H.generated_payload(source)
    plan = json.loads((source / "BUILD_PLAN.json").read_bytes())
    receipt = dict(original_step=GOOD[0], node=H.NODE, manifest_SHA256=H.MANIFEST, epoch=EPOCH,
                   deadline=EPOCH + 300, planned_device_TUs=2, actual_device_TUs=2,
                   actual_GPU_steps=0, runtime_calls=0, all_fresh_M_before_compile=True, passed=True,
                   completed=True, device_qualified=False, arms={},
                   commands=[dict(label=label, argv=argv, rc=0) for label, argv in H.compile_commands(plan)])
    out.mkdir()
    for index, arm in enumerate(H.ARMS):
        path = out / arm
        path.mkdir()
        obj = H.INSIDE_OUTPUT / arm / "fused_scores.o"
        elf = bytearray(64)
        elf[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<HH", elf, 16, 2 if case == "wrong_elf_type" and index else 1, 62)
        elf[-1] = index
        (path / obj.name).write_bytes(elf)
        deps = {str(H.INSIDE_SOURCE / name): pin for name, pin in pins.items()
                if name.startswith(arm + "/") and Path(name).name in
                ("fused_scores.cu", "reduction.hpp", "score_math.hpp")}
        deps[plan["compiler"]["cuda_runtime_header"]] = plan["compiler"]["cuda_runtime_header_SHA256"]
        before = f"{obj}: " + (" " + chr(92) + "\n ").join(deps) + "\n"
        after = before + ("/opt/unexpected.h\n" if case == "md_dependency_drift" and index else "")
        (path / "fused_scores.before.d").write_bytes(before.encode())
        (path / "fused_scores.d").write_bytes(after.encode())
        name = arm + "-symbols"
        symbol = "kimi_output_residual_other" if case == "wrong_symbol" and index else H.SYMBOL
        stdout = f"0000000000000000 T {symbol}\n".encode()
        (out / (name + ".stdout")).write_bytes(stdout)
        [record] = [item for item in receipt["commands"] if item["label"] == name]
        record["stdout_SHA256"] = sha256(stdout).hexdigest()
        receipt["arms"][arm] = dict(object=str(obj), object_SHA256=sha256(elf).hexdigest(),
                                    fresh_M_SHA256=sha256(before.encode()).hexdigest(),
                                    actual_MD_SHA256=sha256(after.encode()).hexdigest(), dependencies_SHA256=deps,
                                    exports=[H.SYMBOL])
    if case == "command_drift":
        receipt["commands"][2]["argv"][3] = "-arch=gfx90a"
    (out / "CPU-result.json").write_bytes(json.dumps(receipt).encode())


def fixture(case):
    with TemporaryDirectory() as work:
        root = Path(work).resolve()
        scope = root / "results" / "controller"
        out = root / "candidates" / H.LABEL
        scope.parent.mkdir()
        out.parent.mkdir()
        events = []
        slot = {"state": "NUMBERED_STEP" if case == "slot_busy" else "NO_NUMBERED_STEP",
                "numbered_step_assigned_DCU_bounds": [0, 0]}
        snapshot = dict(membership_qualified=True, observed_epoch=EPOCH - (15 if case == "stale" else 1),
                        parents={parent: {"owned_by_observer": case != "foreign"} for parent in H.PARENTS})
        snapshot["parents"][H.PARENT]["coverage_by_group_and_node"] = {H.ZONE: {H.NODE: slot}}
        clock = lambda: EPOCH
        mono = lambda: 0.
        rows = iter([GOOD] if case == "duplicate" else
                    [None, None, RUNNING, None] if case == "unknown" else
                    [None, RUNNING, [*GOOD[:6], "cpu=4,mem=8G,node=1,gres/dcu=1"]] if case == "unexpected_gres" else
                    [None, RUNNING, [*GOOD[:2], "FAILED", "1:0", *GOOD[4:]]] if case == "slurm_failed" else
                    [None, RUNNING, GOOD])
        wait = dict(TERMINAL)
        wait["wait_terminal"] = lambda read, end, max_observations: TERMINAL["wait_terminal"](
            read, end, max_observations=3, monotonic=mono, sleep=lambda _: None)

        def runner(argv, **kwargs):
            events.append("exec")
            assert (scope / "exec-attempt.json").is_file()
            assert argv[0] == "srun" and "--cpu-bind=threads" in argv and "--gres=none" in argv
            assert "$(/usr/bin/date +%s)" in argv[-1]
            assert all(f"{var}=-1" in argv[-1] for var in
                       ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            assert f"{scope / 'generated'}:{H.INSIDE_SOURCE}:ro" in argv
            if case == "unknown":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            recorded_output(out, scope / "generated", case)
            return SimpleNamespace(returncode=0)

        def render(path):
            result = RENDER(path)
            if case == "rendered_source_drift":
                (path / "full-tree" / "fused_scores.cu").write_bytes(b"unexpected source")
            return result

        args = dict(member=MEMBER, terminal=wait, render=render, observe=lambda: snapshot,
                    expiry=lambda: EPOCH + (299 if case == "parent_expiry" else 600),
                    query=lambda _: next(rows), run=runner, wall=clock, monotonic=mono)
        if case in {"duplicate", "slot_busy", "stale", "foreign", "parent_expiry"}:
            try:
                H.launch(root, scope, out, CLAIM, **args)
            except ValueError:
                assert not events and not scope.exists()
                return "PRELAUNCH_REFUSED"
            raise AssertionError("unsafe placement accepted")
        result = H.launch(root, scope, out, CLAIM, **args)
        if case == "rendered_source_drift":
            assert not events and result["state"] == "PRELAUNCH_REFUSED" and result["exec_request_attempted"] is False
            assert not (scope / "exec-attempt.json").exists()
            return result["state"]
        expected = ("COMPILE_VERIFIED" if case == "success" else "UNKNOWN" if case == "unknown"
                    else "TERMINAL_UNVERIFIED")
        assert result["state"] == expected, result
        assert events == ["exec"]
        try:
            H.launch(root, scope, out, CLAIM, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("original namespace replay accepted")
        if case == "unknown":
            assert result["client_timeout"] is True and result["observations"] == [None, RUNNING, None]
            still_unknown = H.reconcile(scope, out, CLAIM, MEMBER, TERMINAL, query=lambda _: RUNNING)
            assert still_unknown["state"] == "UNKNOWN"
            recorded_output(out, scope / "generated", "later_actual")
        if case in {"success", "unknown"}:
            immutable = (scope / "accounting-result.json").read_bytes()
            later = H.reconcile(scope, out, CLAIM, MEMBER, TERMINAL, query=lambda _: GOOD)
            assert later["state"] == "COMPILE_VERIFIED" and later["independent_observation"] is True
            assert later["columns"] == H.COLUMNS and later["row"] == GOOD
            assert (scope / "accounting-result.json").read_bytes() == immutable
        return expected


def main():
    cases = ("success", "unknown", "duplicate", "slot_busy", "stale", "foreign", "parent_expiry", "rendered_source_drift",
             "md_dependency_drift", "wrong_elf_type", "wrong_symbol", "command_drift", "unexpected_gres", "slurm_failed")
    results = {case: fixture(case) for case in cases}
    raw = ("|".join(GOOD) + "\n").encode()
    assert H.parse_accounting(raw) == GOOD
    for bad in (raw + raw, raw.replace(b"1103606.712", b"1087656.712"), raw.replace(b"f12r2n11", b"f12r2n12")):
        try:
            H.parse_accounting(bad)
        except ValueError:
            continue
        raise AssertionError("ambiguous or foreign accounting accepted")
    if len(sys.argv) == 2:
        argv = H.command(Path("/fixture/root"), Path("/fixture/generated"), 1.)
        subprocess.run([sys.argv[1], "-n"], input=argv[-1].encode(), timeout=15, check=True)
        # Replace the absent DTK setup only. The real expired shell branch must exit before the worker.
        shell = argv[-1].replace("source /opt/dtk/env.sh", ":")
        stale = subprocess.run([sys.argv[1], "-s"], input=shell.encode(), timeout=15, check=False)
        assert stale.returncode == 124
    print(json.dumps(dict(fixtures=results, accounting_parse_rejections=3, actual_target_compiles=0,
                          actual_slurm_submissions=0, actual_GPU_steps=0, actual_SSH_attempts=0,
                          exact_rendered_members=13, bash_validation=len(sys.argv) == 2), indent=2))


if __name__ == "__main__":
    main()

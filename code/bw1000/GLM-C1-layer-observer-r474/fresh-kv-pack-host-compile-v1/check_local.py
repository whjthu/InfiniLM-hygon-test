"""Exercise the real HOST entry with isolated CPU object and accounting fixtures."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_once as H


def digest(raw):
    return sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")


def test_account():
    row = ["1103606.650", H.LABEL, "COMPLETED", "0:0", "8", H.NODE, "cpu=4,mem=8G,node=1"]
    return row


def account_line(row):
    return ("|".join(row) + "\n").encode("utf8")


def prepare_fixture(root):
    root.mkdir()
    for d in ("incoming", "results", "candidates"):
        (root / d).mkdir()
    stage = root / "incoming" / (H.LABEL + "-host-source")
    runpy.run_path(str(H.SOURCE / "prepare_target.py"))["stage"](stage)
    return root / "results" / (H.LABEL + "-controller"), root / "candidates" / H.LABEL


def local_elf(path):
    cxx, coff, elf = (path / name for name in ("fixture.cpp", "fixture.obj", "fixture.elf"))
    cxx.write_text("\n".join(f'extern "C" int {name}() {{ return {i}; }}' for i, name in enumerate(H.EXPORTS)),
                   encoding="utf8")
    subprocess.run(["g++", "-c", str(cxx), "-o", str(coff)], check=True, timeout=20)
    subprocess.run(["objcopy", "-O", "elf64-x86-64", str(coff), str(elf)], check=True, timeout=20)
    return elf.read_bytes()


def build_fixture(intent_path, output, row, elf, drift):
    intent = json.loads((intent_path / "intent.json").read_bytes())
    epoch = intent["epoch"]
    assert intent["deadline"] == epoch + 300
    plan = json.loads((H.SOURCE / "PLAN.json").read_bytes())
    output.mkdir()
    header = plan["cuda_runtime_header"]
    ticket = f"{H.OBJECT}: {H.SOURCE_LEAF} {header}\n".encode()
    md = ticket if not drift else ticket.replace(header.encode(), b"/refactor/changed-header.h")
    for name, raw in (("fresh_kv_pack.o", elf), ("fresh_kv_pack.before.d", ticket),
                      ("fresh_kv_pack.d", md)):
        (output / name).write_bytes(raw)
    flags = plan["compiler_prefix"]
    commands = {
        "fresh-M": flags + ["-M", "-MF", H.OBJECT.removesuffix(".o") + ".before.d", "-MT", H.OBJECT, H.SOURCE_LEAF],
        "compile": flags + ["-MD", "-MF", H.OBJECT.removesuffix(".o") + ".d", "-MT", H.OBJECT, "-c",
                            H.SOURCE_LEAF, "-o", H.OBJECT],
        "symbols": ["/usr/bin/nm", "-g", "--defined-only", H.OBJECT],
    }
    written = []
    for name, argv in commands.items():
        log = "".join(f"0 T {s}\n" for s in H.EXPORTS).encode() if name == "symbols" else b""
        (output / (name + ".stdout")).write_bytes(log)
        (output / (name + ".stderr")).write_bytes(b"")
        written.append(dict(label=name, argv=argv, returncode=0, stdout_SHA256=digest(log), stderr_SHA256=digest(b"")))
    write_json(output / "CPU-result.json", dict(
        label=H.LABEL, manifest_SHA256=H.MANIFEST, original_step=row[0], node=H.NODE, epoch=epoch,
        deadline=epoch+300, GRES="none", planned_device_TUs=1, actual_device_TUs=1,
        actual_GPU_steps=0, explicit_Runtime_calls=0, exports=H.EXPORTS, object=H.OBJECT, passed=True,
        completed=True, device_qualified=False, object_SHA256=digest(elf), fresh_M_SHA256=digest(ticket),
        actual_MD_SHA256=digest(md), commands=written, dependencies_SHA256={
            H.SOURCE_LEAF: plan["source_SHA256"], header: plan["cuda_runtime_header_SHA256"]}))
    return epoch


def reject(fn, wanted):
    try:
        fn()
    except ValueError as error:
        assert wanted in str(error), str(error)
    else:
        raise AssertionError(wanted)


def main():
    claim = digest((H.HERE / "CLAIM.json").read_bytes())
    target = test_account()
    accounting = runpy.run_path(str(H.REPO / H.ACCOUNTING))["wait_terminal"]

    def shorter_wait(read, deadline, **kwargs):
        return accounting(read, deadline, interval=.001, **kwargs)

    with TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        elf = local_elf(base)
        for name in ("success", "unknown", "ambiguous", "preexisting", "wrong_node", "md_drift", "gres", "object"):
            root = base / name
            controller, output = prepare_fixture(root)
            calls = []
            checks = []
            row = target[:]
            if name == "gres":
                row[-1] = "cpu=4,mem=8G,node=1,gres/dcu=1"
            if name == "wrong_node":
                row[5] = "f12r2n12"
            running = row[:]
            running[2] = "RUNNING"
            replies = {
                "unknown": [b""] + [account_line(running)] * 6,
                "ambiguous": [account_line(row) * 2],
                "preexisting": [account_line(row)],
                "wrong_node": [account_line(row)],
            }.get(name, [b"", account_line(running), account_line(row)])
            responses = iter(replies)

            def query(args, timeout):
                assert args[0] == "sacct" and args[args.index("-j") + 1] == "1087656,1103606"
                assert 0 < timeout <= 8
                return next(responses)

            def run(command, **kwargs):
                calls.append(command)
                assert command[0] == "srun" and "--gres=none" in command
                assert "--nodelist=" + H.NODE in command and "--jobid=" + H.PARENT in command
                assert "--cpus-per-task=4" in command and "--time=00:05:00" in command
                assert "--rocm" not in command and 0 < kwargs["timeout"] < 300
                if name == "unknown":
                    raise subprocess.TimeoutExpired(command, 1)
                original_epoch = build_fixture(controller, output, row, elf if name != "object" else b"plain text", name == "md_drift")
                assert f"TUNING_CPU_EPOCH={original_epoch}" in command[-1]
                assert f"TUNING_CPU_DEADLINE={original_epoch+300}" in command[-1]
                return subprocess.CompletedProcess(command, 0)

            def launch():
                return H.execute(claim, observe_slot=lambda: checks.append(True), runner=run,
                                 waiter_override=shorter_wait)

            with patch.object(H, "ROOT", root), patch.object(subprocess, "check_output", side_effect=query):
                prelaunch_refusal = {"ambiguous": "multiple accounting", "preexisting": "already submitted",
                                     "wrong_node": "different step or node"}
                if name in prelaunch_refusal:
                    reject(launch, prelaunch_refusal[name])
                    assert not calls and not controller.exists()
                    continue
                result = launch()
                assert len(calls) == len(checks) == 1
                if name == "success":
                    assert result["state"] == "COMPILE_VERIFIED" and result["actual_device_TUs"] == 1
                    assert result["fresh_M_SHA256"] == result["actual_MD_SHA256"]
                    before = (controller / "accounting-result.json").read_bytes()
                    reconciled = H.execute(claim, reconcile=True, query=lambda timeout: row,
                                           runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError("relaunch")))
                    assert reconciled["state"] == "COMPILE_VERIFIED" and reconciled["independent_observation"]
                    assert (controller / "accounting-result.json").read_bytes() == before
                    reject(lambda: H.execute(claim, reconcile=True, query=lambda timeout: running), "disagree")
                elif name == "unknown":
                    assert result["state"] == "UNKNOWN" and result["client_timeout"] and len(result["observations"]) == 6
                    assert H.execute(claim, reconcile=True, query=lambda timeout: running)["state"] == "UNKNOWN"
                    reject(launch, "namespace already exists")
                    assert len(calls) == 1 and not output.exists()
                else:
                    assert result["state"] == "TERMINAL_UNVERIFIED"
                    want = {"md_drift": "fresh M/actual MD differ", "gres": "zero-GRES",
                            "object": "lacks host ELF64 REL"}[name]
                    assert want in result["error"]
    print(json.dumps(dict(actual_local_host_elf_compiles=1, fixture_normal_launch_terminal=1,
                          fixture_readonly_reconcile=2, fixture_unknown_preserved=1,
                          fixture_refusals=8, actual_new_slurm_steps=0,
                          actual_target_compiles=0, actual_GPU_steps=0), indent=2))


if __name__ == "__main__":
    main()

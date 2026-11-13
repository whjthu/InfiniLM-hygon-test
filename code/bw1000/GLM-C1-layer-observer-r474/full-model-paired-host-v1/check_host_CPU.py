"""CPU rehearsal of the HOST workflow using recorded-shaped Slurm fixtures."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import host_launch as host
from host_probe import verify_before


def rejects(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("expected refusal")


def rehearsal(root, case):
    run = "0" * 32
    events = []

    def step(label, phase, out, env, end):
        events.append(phase)
        handle = "1103606." + str(len(events))
        if case == "unresolved" and phase == "gate":
            return {"terminal": False, "passed": False, "row": None}
        for node in host.NODES:
            pci = [f"0000:{i:02x}:00.0" for i in range(8)]
            row = dict(node=node, boot_id="boot", step=handle, pci=pci,
                       pci_to_kfd={p: i+1 for i, p in enumerate(pci)} if phase == "gate" else None,
                       monotonic_ns=len(events), kfd_holder_count=0,
                       ports_bindable={"43843": True, "43844": True})
            if phase == "post" and case == "changed-hardware":
                row["boot_id"] = "reboot"
            (out / phase / (node + ".json")).write_text(json.dumps(row))
        if phase == "gate":
            assert env["GLM470_GPU_MF"] == host.H(out / "RUN_MANIFEST.json")
            assert env["GLM474_PAIR_ARM"] in host.ARM_ORDER
        return dict(terminal=True, passed=case != "failed" or phase != "gate",
                    row=[handle, label, "COMPLETED", "0:0", "3", "fixture"])

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(host, "HOST", root))
        stack.enter_context(patch.object(host, "check_idle"))
        stack.enter_context(patch.object(host, "account", return_value=None))
        stack.enter_context(patch.object(host, "run_step", side_effect=step))
        owners = stack.enter_context(patch.object(host, "verify_owners"))
        (root / "results").mkdir()
        if case == "pass":
            step_id, hardware = host.run_arm(run, "baseline", "c"*64, "a"*64, {})
            host.run_arm(run, "single-drain", "c"*64, "a"*64, {"baseline": {"hardware": hardware}})
            assert step_id == "1103606.2" and owners.call_count == 2
            assert events == ["pre", "gate", "post"] * 2
            try:
                host.run_arm(run, "baseline", "c"*64, "a"*64, {})
            except FileExistsError:
                pass
            else:
                raise AssertionError("replayed existing run")
            assert len(events) == 6
        else:
            rejects(lambda: host.run_arm(run, "baseline", "c"*64, "a"*64, {}))
            assert events == (["pre", "gate"] if case == "unresolved" else ["pre", "gate", "post"])
            owners.assert_not_called()


def main():
    for case in ("pass", "unresolved", "failed", "changed-hardware"):
        with tempfile.TemporaryDirectory() as tmp:
            rehearsal(Path(tmp), case)
    out = Path("scope")
    assert "--gres=dcu:8" in host.command("label", "gate", out)
    assert "--gres=none" in host.command("label", "post", out)
    row = dict(node="f13r1n05", boot_id="boot", pci=["pci"], monotonic_ns=1)
    verify_before(row, dict(row, monotonic_ns=10))
    rejects(lambda: verify_before(row, dict(row, monotonic_ns=90_000_000_002)))
    rejects(lambda: verify_before(row, dict(row, pci=["different"])))
    for line, ok in (("1103606.1|exact|COMPLETED|0:0|3|nodes", True),
                     ("1103606.extern|exact|COMPLETED|0:0|3|nodes", False)):
        with patch.object(host, "query", return_value=[line]):
            if ok:
                assert host.account("exact", time.monotonic()+5)[0] == "1103606.1"
            else:
                rejects(lambda: host.account("exact", time.monotonic()+5))
    batch = "StepId=1103606.batch State=RUNNING NodeList=other"
    extern = "StepId=1103606.extern State=RUNNING NodeList=f13r1n[05,07]"
    active = "StepId=1103606.71 State=RUNNING NodeList=f13r1n[05,07]"
    host.check_live_steps([batch, extern])
    rejects(lambda: host.check_live_steps([]))
    rejects(lambda: host.check_live_steps([batch, "StepId=1103606.71 State=RUNNING"]))
    rejects(lambda: host.check_live_steps(["unexpected controller output"]))
    with patch.object(host, "expand", side_effect=lambda nodes: set(host.NODES) if "[05,07]" in nodes else {"other"}):
        rejects(lambda: host.check_live_steps([batch, extern, active]))
        host.check_live_steps([batch, extern, active.replace("f13r1n[05,07]", "other")])
    print("PASS: serial arms, duplicate refusal, unresolved/failed accounting, hardware changes, GPU gate arguments")


if __name__ == "__main__":
    main()

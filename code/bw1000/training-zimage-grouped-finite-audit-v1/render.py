"""Derive one continuous Z-Image run with grouped finite audits and paired timing."""

from hashlib import sha256
import json
import re


PARENT_MANIFEST = "eb0c0129b0ab6274e90206d75190aad98def974186652daddcca8375ac1feb2d"
PREVIOUS_LABEL = "training-zimage-COCO20-recovery-boundaries-hip-v2"


def edit(raw, before, after):
    key = before.encode()
    if raw.count(key) != 1:
        raise ValueError(f"Frozen source anchor differs: {before[:85]}")
    return raw.replace(key, after.encode(), 1)


def derive(manifest, old, audit, reader, label, request):
    original = json.loads(manifest)["files"]
    if sha256(manifest).hexdigest() != PARENT_MANIFEST or set(old) != set(original):
        raise ValueError("Frozen Z-Image source inventory differs")
    if any(len(raw) != original[name]["bytes"] or sha256(raw).hexdigest() != original[name]["SHA256"]
           for name, raw in old.items()):
        raise ValueError("Frozen Z-Image source member differs")
    if re.fullmatch(r"[a-z0-9-]{12,80}", label) is None or re.fullmatch(r"[0-9a-f]{32}", request) is None:
        raise ValueError("Invalid privately bound run identity")
    files = dict(old)
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=label, request=request, actual_optimizer_updates_planned=25, whole_seconds=720,
                 worker_seconds=650, maximum_device_seconds={"continuous": 540})
    files["CLAIM.json"] = json.dumps(claim, indent=2, sort_keys=True).encode()

    witness = files["observe_update.py"]
    witness = edit(witness, "if not all(p.grad is not None and bool(self.torch.isfinite(p.grad).all()) for p in self.named):",
                   "if not self.finite_audit.check((p.grad for p in self.named), stage):")
    files["observe_update.py"] = witness
    loop = files["training_loop.py"]
    for a, b in (
        ("from boundaries import OBSERVED_STEPS, TensorWitness, after_update",
         "from boundaries import OBSERVED_STEPS, TensorWitness, after_update\nfrom finite_audit import FiniteAudit"),
        ('(output/"steps.jsonl").open("x",encoding="utf-8") as metrics:',
         '(output/"steps.jsonl").open("x",encoding="utf-8") as metrics, \\\n'
         '         (output/"timing.jsonl").open("x",encoding="utf-8") as timing:'),
        ("        for step in schedule(mode):\n            deadline()",
         "        for step in schedule(mode):\n            step_start = time.monotonic()\n            deadline()"),
        ("            with captures.observe_native_loss(native,witness):",
         "            audit = FiniteAudit(torch, step)\n            witness.finite_audit = audit\n"
         "            native_start = time.monotonic()\n            with captures.observe_native_loss(native,witness):"),
        ('            append(journal,dict(global_step=step,event="optimizer_returned"))',
         '            native_seconds = time.monotonic() - native_start\n'
         '            append(journal,dict(global_step=step,event="optimizer_returned"))'),
        ('need(all(bool(torch.isfinite(p).all()) for p in named.values()),"Updated LoRA parameter is not finite")',
         'need(audit.check(named.values(), "updated"),"Updated LoRA parameter is not finite")'),
        ('            if step in claim["checkpoint_steps"]:',
         '            checkpoint_started = time.monotonic()\n            if step in claim["checkpoint_steps"]:'),
        ("                del saved\n    need(completed",
         '                del saved\n'
         '            append(timing, {"global_step":step,"native_update_seconds_including_witness":native_seconds,\n'
         '                  "checkpoint_seconds":time.monotonic()-checkpoint_started if step in claim["checkpoint_steps"] else None,\n'
         '                  "worker_step_seconds":time.monotonic()-step_start,"paired_finite_checks":audit.pairs})\n'
         '    need(completed'),
    ):
        loop = edit(loop, a, b)
    files["training_loop.py"] = loop

    control = files["controller.py"]
    control = edit(control, "from boundaries import wait_for_worker_files",
                   "from read_timing import verify_continuous")
    control = edit(control, 'claim["actual_optimizer_updates_planned"] == 30',
                   'claim["actual_optimizer_updates_planned"] == 25')
    control = edit(control, '    def remaining(reserve=0):\n        available = deadline - time.time() - reserve',
                   '    monotonic_end = time.monotonic() + deadline - time.time()\n\n'
                   '    def remaining(reserve=0):\n'
                   '        available = min(deadline - time.time(), monotonic_end - time.monotonic()) - reserve')
    control = edit(control, '        args = ["srun", "--jobid="+parent,',
                   '        seconds = int(budget)\n'
                   '        need(seconds > 0, "No time remains for Slurm phase")\n'
                   '        limit = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"\n'
                   '        args = ["srun", "--jobid="+parent,')
    control = edit(control, '"--time="+str(int(budget / 60) + 1), *command]', '"--time="+limit, *command]')
    control = edit(control, '            if gpu:\n                result["gpu_srun_attempted"] = True',
                   '            if gpu:\n                result["gpu_srun_attempted"] = True\n'
                   '            def owned():\n'
                   '                stat = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()\n'
                   '                return (int(stat[2]), int(stat[19]))\n'
                   '            owner = owned()\n'
                   '            need(owner[0] == child.pid, "Slurm child is outside its own process group")\n'
                   '            save(phase+"-owner.json", {"pid":child.pid, "pgid":owner[0], "start_ticks":owner[1]})')
    control = edit(control, '                os.killpg(child.pid, signal.SIGTERM)',
                   '                need(owned() == owner, "Owned Slurm process changed before TERM")\n'
                   '                os.killpg(child.pid, signal.SIGTERM)')
    control = edit(control, '                    os.killpg(child.pid, signal.SIGKILL)',
                   '                    need(owned() == owner, "Owned Slurm process changed before KILL")\n'
                   '                    os.killpg(child.pid, signal.SIGKILL)')
    control = edit(control, '        compare = load_compare(json.loads(manifest_raw)["files"])\n', "")
    start = control.index(b"        if continuous:\n")
    end = control.index(b'        result["domain_workers"]', start)
    block = control[start:end]
    if b'device("resume20")' not in block or b"comparison = compare(ROOT)" not in block:
        raise ValueError("Frozen continuous/resume controller separation differs")
    control = control[:start] + (
        '        if continuous:\n'
        '            audit = verify_continuous(ROOT, claim["request"], phases["continuous"]["accounting"][0])\n'
        '            result["timing_SHA256"] = save("timing-review.json", audit)\n'
        '            result["paired_passed"] = audit["passed"]\n'
    ).encode() + control[end:]
    control = edit(control, 'for mode in ("continuous", "resume20"):', 'for mode in ("continuous",):')
    files["controller.py"] = control

    deploy = files["deploy.py"]
    deploy = edit(deploy, f'LABEL = "{PREVIOUS_LABEL}"', f'LABEL = "{label}"')
    deploy = edit(deploy, 'phases = ("admission", "pre", "continuous", "between", "resume20", "post")',
                  'phases = ("admission", "pre", "continuous", "post")')
    files["deploy.py"] = deploy
    files["finite_audit.py"] = audit
    files["read_timing.py"] = reader
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    files["SOURCE_FILES.json"] = json.dumps({"files":{
        name: {"bytes":len(raw), "SHA256":sha256(raw).hexdigest()} for name, raw in files.items()}},
        indent=2, sort_keys=True).encode()
    return files

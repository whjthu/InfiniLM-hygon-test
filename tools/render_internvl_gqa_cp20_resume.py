"""Render a bounded native CP20 continuation from the exact continuous GQA25 recipe."""

import argparse
import json
from pathlib import Path

import render_internvl_gqa64_continuous25 as original


REQUEST = "5146842eeddb4698abdc2d08d8ac7e16"
LABEL = "training-internvl-gqa64-cp20-resume-hip-v1"
ORIGINAL_MANIFEST = "52444c213cac9529bb150e8b5d10aa107c8cc9c29a162a6b490060223ad92ed7"
WHOLE, NATIVE, SLURM = 1740, 1380, 1500
FAMILY = original.ARTIFACT_READER.parent
EXTRA = ("internvl_gqa_resume_summary.py", "internvl_gqa_val96.py",
         "internvl_gqa_val96_native.py", "internvl_gqa_val96_run.py",
         "internvl_gqa_miopen_home.py", "internvl_gqa_resume_handoff.py")


def digest(raw):
    return original.first.digest(raw)


def edit(files, name, changes):
    return original.edit(files, name, changes)


def render(claim_bytes):
    files = original.render()
    if digest(files.pop("SOURCE_FILES.json")) != ORIGINAL_MANIFEST:
        raise ValueError("Original continuous training source manifest differs")
    original_claim = json.loads(files["CLAIM.json"])
    claim = json.loads(claim_bytes)
    if claim["request"] != REQUEST or claim["label"] != LABEL or claim["native_seconds"] != NATIVE:
        raise ValueError("Unexpected CP20 recovery claim")
    files["CLAIM.json"] = original.first.encode({
        **claim, "root_GO": True, "GPU_GO": True, "training_effect_GO": False, "formal17_qualified": False,
        "exports_SHA256": original_claim["exports_SHA256"],
    })
    for name in ("scope.py", "control_once.py"):
        edit(files, name, [(original.REQUEST, REQUEST), (original.LABEL, LABEL)])
    edit(files, "scope.py", [("WHOLE, NATIVE, SLURM = 3600, 3180, 3300",
                             f"WHOLE, NATIVE, SLURM = {WHOLE}, {NATIVE}, {SLURM}")])
    edit(files, "control_once.py", [("WHOLE, GPU_LIMIT = 3600, 3300", f"WHOLE, GPU_LIMIT = {WHOLE}, {SLURM}")])
    old_run = f"RESULTS / {original.LABEL!r}"
    edit(files, "node.py", [
        ('    binds=[*plan["binds"],', f'''    prior = {old_run}
    binds=[*plan["binds"],f"{{prior}}:{{prior}}:ro",
           f"{{prior / 'train/checkpoint-20'}}:/gqa-recovery-cp20:ro",'''),
        ("native_schedule_steps=25,planned_updates=25,planned_first_two_micro_forwards_by_rank=8",
         "native_schedule_steps=25,planned_updates=5,planned_recovery_micro_forwards_by_rank=20"),
    ])
    files["callback.py"] = (FAMILY / "internvl_gqa_resume_callback.py").read_bytes()
    edit(files, "native.py", [
        ("from internvl_gqa25_artifacts import gather", "from internvl_gqa_resume_summary import gather"),
        ('    argv=[sys.executable,"-B",*launch["command"]]',
         '''    argv=[sys.executable,"-B",*launch["command"],"--resume_from_checkpoint","/gqa-recovery-cp20",
          "--resume_only_model","false","--ignore_data_skip","false"]'''),
        ("planned_update_slots=25,original_schedule_steps=25,", "planned_update_slots=5,original_schedule_steps=25,"),
        ("planned_input_slots=800,first_two_witness_slots=64,binding_SHA256=",
         "planned_input_slots=160,starting_global_step=20,binding_SHA256="),
    ])
    text = files["native.py"].decode()
    start = text.index("def report_rank(")
    stop = text.index("def run(epoch, plan_sha):", start)
    text = text[:start] + text[stop:]
    start = text.index("    reports=[]")
    stop = text.index('    write_once(RUN/"native-outcome.json",report)', start)
    files["native.py"] = (text[:start] + '''    recovery = None
    clean = cleanup["owned_processes_remaining"] == 0 and not cleanup["termination_needed"]
    if child is not None and child.returncode == 0 and error is None and clean:
        try:
            recovery = gather(RUN)
        except (ValueError, KeyError, OSError, TypeError) as failure:
            error = type(failure).__name__ + ": " + str(failure)
    complete = recovery is not None
    report = dict(request=REQUEST,actual_step=intent["step"],passed=complete,cleanup=cleanup,
                  native_returncode=child.returncode if child else None,error=error,recovery=recovery,
                  original_A100_qualification=False,formal_performance=False)
''' + text[stop:]).encode()
    del files["internvl_gqa25_artifacts.py"]
    for name in EXTRA:
        files[name] = (FAMILY / name).read_bytes()
    files["SOURCE_FILES.json"] = original.first.encode({
        "request": REQUEST,
        "files": {name: {"bytes": len(raw), "SHA256": digest(raw)} for name, raw in files.items()},
    })
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", type=Path)
    args = parser.parse_args()
    bundle = render(args.claim.read_bytes())
    print(json.dumps({"manifest_SHA256": digest(bundle["SOURCE_FILES.json"]), "files": len(bundle),
                      "request": REQUEST, "local_check": True}, sort_keys=True))

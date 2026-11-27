"""Render the frozen .4703 production scope with a separately gated CP2-to-CP3 delta."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cp3_contract as cp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V3 = HERE.with_name("training-internvl-gqa56-first2-native-observed-v3")/"render.py"
V3_MANIFEST = "61598d967fd5b06f368262de97082f0607fa6bb5a5ca8fef19a771b0e3fcdd3b"
ARCHIVES = REPO/"results/bw1000/2026-11-28"
RECEIPTS = {
    "specialized-internvl-4703-real-two-updates-and-original-cp2-readback.json": cp.CP2_ARCHIVE,
    "specialized-internvl-4723-4724-original-cp2-cpu-state-and-rng.json": cp.CPU_RESTORE_ARCHIVE,
    "specialized-internvl-4726-epoch1-native-order-cp3-32-slots.json": cp.EPOCH_ARCHIVE,
}


def edit(files, name, pairs):
    source = files[name].decode()
    for old, new in pairs:
        cp.need(source.count(old) == 1, f"EXACT_BASELINE_{name}: {old[:75]}")
        source = source.replace(old, new)
    files[name] = source.encode()


def cut(files, name, start, stop, replace):
    source = files[name].decode()
    left = source.index(start)
    right = source.index(stop, left)
    files[name] = (source[:left] + replace + source[right:]).encode()


def baseline():
    spec = importlib.util.spec_from_file_location("internvl_cp3_original_v3", V3)
    original = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(V3.parent))
    try:
        spec.loader.exec_module(original)
        files = original.render()
    finally:
        sys.path.remove(str(V3.parent))
    cp.need(cp.digest(files.pop("SOURCE_FILES.json")) == V3_MANIFEST, "EXACT_FROZEN_4703_SOURCE_MANIFEST")
    return original, files


def render():
    original, files = baseline()
    for name, expected in RECEIPTS.items():
        raw = (ARCHIVES/name).read_bytes()
        cp.need(cp.digest(raw) == expected, "EXACT_ORIGINAL_EVIDENCE")
        if expected == cp.CP2_ARCHIVE:
            files["cp2-archive.json"] = raw
    for name in ("callback.py", "cp3_contract.py", "owner_gate.py"):
        files[name] = (HERE/name).read_bytes()
    for name in ("scope.py", "control_once.py"):
        edit(files, name, [(original.gate.REQUEST, cp.REQUEST), (original.gate.LABEL, cp.LABEL),
                           (str(original.gate.PLACEMENT["port"]), "29991")])
    edit(files, "scope.py", [
        (repr("/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/internvl-gqa56-native-arrow-v4-0947b7dac1da45ac842fae8bb0adc25e/output"),
         repr(cp.ARROW_ROOT.as_posix())),
        ("internvl_gqa56_first2_observed_engineering", "internvl_gqa56_cp3_observed_engineering"),
    ])
    edit(files, "node.py", [
        ("    from scope import ARROW_AUTHORITY, GQA_FINGERPRINT_RECEIPT, GQA_SPLIT",
         "    from cp3_contract import ORIGINAL_CP2, CHECKPOINT\n"
         "    from scope import ARROW_AUTHORITY, GQA_FINGERPRINT_RECEIPT, GQA_SPLIT"),
        ('f"{ASSETS}:{ASSETS}:ro",f"{REFERENCE}:{REFERENCE}:ro",',
         'f"{ASSETS}:{ASSETS}:ro",f"{ORIGINAL_CP2}:{CHECKPOINT}:ro",'),
        ('("result.json","physical-rank-matrix.private.json"))',
         '("result.json","physical-rank-matrix.private.json","cp3-first4.private.json"))'),
        ("planned_updates=2,planned_micro_forwards_by_rank=8", "planned_updates=1,planned_micro_forwards_by_rank=4"),
    ])
    edit(files, "control_once.py", [
        ("from owner_gate import CPU_ROW, load, verify_cpu_accounting",
         "from owner_gate import CPU_ROW, exact_accounting, load, verify_cpu_accounting"),
        ('    load(HERE,launch["owner_SHA256"],launch["publication"],launch["claim_SHA256"],launch["manifest_SHA256"],pin,time.time())',
         '''    owner=load(HERE,launch["owner_SHA256"],launch["publication"],launch["claim_SHA256"],launch["manifest_SHA256"],pin,time.time())
    exact=exact_accounting(owner["latest_holder_step"])
    need(exact==owner["holder_terminal_row"],"Current exact holder no longer matches authorized terminal")'''),
        ("Exact prior .4701 accounting is absent", "Exact prior .4726 accounting is absent"),
    ])
    edit(files, "domain_checks.py", [
        ("import live_guard", "import live_guard\nimport cp3_contract"),
        ("    by_rank,arrow=live_guard.authority(ARROW_AUTHORITY,exposures,original,proposed,subset,pin)",
         '''    matrix=cp3_contract.load_arrow(ARROW_AUTHORITY,pin)
    source_rows={row["ordinal"]:row["row_SHA256"] for row in subset.values()}
    by_rank,arrow=live_guard.bind_matrix(exposures,original,proposed,source_rows,matrix)
    need(live_guard.digest(live_guard.encode(arrow))==cp3_contract.ARROW_ORDER,"EXACT_CP3_REBUILT_ARROW")'''),
    ])
    cut(files, "live_guard.py", "def authority(", "def live_arrow(", "")
    edit(files, "live_guard.py", [
        ('need(set(counts)==set(source_rows) and Counter(counts.values())=={1:48,2:8} and\n'
         '         [len({slot["source_ordinal"] for slot in evidence if slot["update"]==step}) for step in (1,2)]==[31,31],\n'
         '         "EXACT_4699_56_PLUS_8_AND_31_31_WINDOWS")',
         'need(set(counts)==set(source_rows) and Counter(counts.values())=={1:48,2:8},'
         '"EXACT_CP3_56_PLUS_8_EPOCH1")'),
        ('MATRIX_PIN="a931dc4bc7152c5aa866bc257b2c0be19706ca7974d0f8e606a19e8666ff7e1d"',
         f'MATRIX_PIN="{cp.MATRIX}"'),
        ('            batches=original(instance)\n            return self.witness(batches) if instance is sampler else batches',
         '''            if instance is sampler:
                need(instance.base_seed==42 and instance.curr_seed==43,"ACTUAL_NATIVE_EPOCH1_NOT_SET")
                return self.witness(original(instance))
            return original(instance)'''),
    ])
    edit(files, "slot_guard.py", [
        ('def seal_checkpoint(root, check_deadline=lambda: None):',
         'def seal_checkpoint(root, check_deadline=lambda: None):'),
        ('optional = {"training_args.bin","additional_config.json","README.md"}',
         'optional = {"training_args.bin","additional_config.json","README.md"}\n    required.add("args.json")'),
        ('for name in ("trainer_state.json","adapter_config.json"):',
         'for name in ("trainer_state.json","adapter_config.json","args.json"):'),
        ('==(2,25),"EXACT_CP2_STEP_AND_SCHEDULE")', '==(3,25),"EXACT_CP3_STEP_AND_SCHEDULE")'),
        ('        else:\n            need((state["r"],state["lora_alpha"],state["lora_dropout"])',
         '        elif name=="adapter_config.json":\n            need((state["r"],state["lora_alpha"],state["lora_dropout"])'),
        ('    need(observed=={path.name for path in root.iterdir()}',
         '''        else:
            need(type(state) is dict and state["max_steps"]==25 and state["gradient_accumulation_steps"]==4,
                 "EXACT_NATIVE_CP3_SWIFT_ARGS_SIDECAR")
    need(observed=={path.name for path in root.iterdir()}'''),
        ("return dict(step=2,native_schedule_steps=25", "return dict(step=3,native_schedule_steps=25"),
    ])
    edit(files, "native.py", [
        ('    need(INITIAL.resolve(strict=True)==INITIAL and INITIAL.is_file() and INITIAL.stat().st_size==INITIAL_BYTES,\n'
         '         "Original .362 rank00 initial file extent differs")',
         '''    import cp3_contract
    cp2_members=cp3_contract.verify_cp2(HERE/"cp2-archive.json",cp3_contract.CHECKPOINT,pin,
                                       lambda: seconds_left(epoch,NATIVE))'''),
        ('    argv=[sys.executable,"-B",*launch["command"]]',
         '    argv=[sys.executable,"-B",*cp3_contract.resume_argv(launch["command"])]'),
        ('planned_update_slots=2,original_schedule_steps=25,\n               planned_input_slots=64',
         'planned_update_slots=1,original_schedule_steps=25,verified_original_cp2_members=cp2_members,\n'
         '               planned_input_slots=32'),
    ])
    cut(files, "native.py", "def report_rank(", "def run(", "")
    cut(files, "native.py", "    reports=[]\n    attempted=[]", '    write_once(RUN/"native-outcome.json",report)', '''    reports=[]
    partial=[]
    for rank in range(8):
        folder=RUN/"capture"/f"rank-{rank:02d}"
        observed=[]
        try:
            for offset in range(1,5):
                witness=folder/f"forward-{offset}.json"
                if witness.is_file():
                    entry=json.loads(pin(witness,cap=1024))
                    need(entry==dict(request=REQUEST,rank=rank,microbatch=offset,global_step_before=2),
                         "EXACT_CP3_PARTIAL_FORWARD")
                    observed.append(offset)
            update=folder/"update-3-observed.json"
            if update.is_file():
                entry=json.loads(pin(update,cap=1024))
                need(entry["request"]==REQUEST and entry["rank"]==rank and entry["native_update_returned"] is True
                     and entry["global_step_before"]==2 and entry["AdamW_step"]==3,
                     "EXACT_CP3_PARTIAL_UPDATE")
            partial.append(dict(rank=rank,completed_forward_slots=observed,actual_update_observed=update.is_file()))
        except (ValueError,KeyError,OSError,TypeError):
            error=error or "INVALID_CP3_PARTIAL_PROGRESS"
        path=folder/"resume3.json"
        if not path.is_file():
            continue
        try:
            row=json.loads(pin(path,cap=8192))
            need((row["rank"],row["request"],row["final_global_step"],row["updates"],row["cp3_native_save_events"],
                  row["rng_restore_calls"])==(rank,REQUEST,3,1,1,1)
                 and row["actual_device_rng_restored"] is True and row["rows"]==domain_slots[rank][:4]
                 and len(row["losses"])==4 and row["initial_parameter_SHA256"]==cp3_contract.CP2_FP
                 and row["final_parameter_SHA256"]!=cp3_contract.CP2_FP,"EXACT_NATIVE_CP3_RANK")
            reports.append(row)
        except (ValueError,KeyError,OSError,TypeError):
            error=error or "INVALID_CP3_RANK_RECEIPT"
    complete=(child is not None and child.returncode==0 and error is None and len(reports)==8
              and cleanup["owned_processes_remaining"]==0 and not cleanup["termination_needed"]
              and len({row["final_parameter_SHA256"] for row in reports})==1
              and len({row["optimizer_index_to_name_SHA256"] for row in reports})==1)
    sealed=None
    if complete:
        try:
            sealed=slot_guard.seal_checkpoint(RUN/"train"/"checkpoint-3",lambda: seconds_left(epoch,NATIVE))
            write_once(RUN/"checkpoint-3-files.json",sealed)
        except (ValueError,KeyError,OSError,TypeError):
            error="CP3_INVENTORY_OR_DURABILITY_FAILED"
            complete=False
    report=dict(request=REQUEST,actual_step=intent["step"],native_returncode=child.returncode if child else None,
                error=error,cp3_one_update_complete=complete,checkpoint3_sealed=sealed is not None and complete,
                planned_ranks=8,qualified_rank_receipts=len(reports),planned_input_slots=32,
                qualified_domain_input_slots=32 if complete else None,rank_receipts=reports,cleanup=cleanup,
                partial_rank_progress=partial,
                formal17_qualified=False,training_effect_GO=False,formal_performance=False)
''')
    claim = json.loads(files["CLAIM.json"])
    for stale in ("expected_unique_source_by_update", "independent_matrix_proof_SHA256"):
        claim.pop(stale)
    claim.update(request=cp.REQUEST, label=cp.LABEL, schema="INTERNVL_GQA56_CP2_TO_CP3_NATIVE_V1",
                 state="NO_ADOPTION_WITHOUT_NEW_ROOT_AUTHORITY_AND_FRESH_HOLDER_RELEASE",
                 root_GO=False, GPU_GO=False, planned_updates=1, planned_input_slots=32,planned_microbatches=32,
                 native_occurrence_matrix_SHA256=cp.MATRIX,actual_cpu_arrow_source_step="1103606.4726",
                 actual_cpu_arrow_result_SHA256=cp.ARROW_RESULT,
                 prior_zero_gres_step="1103606.4726", prior_zero_gres_result_SHA256=cp.ARROW_RESULT,
                 prior_zero_gres_archive_SHA256=cp.EPOCH_ARCHIVE,
                 original_cp2_archive_SHA256=cp.CP2_ARCHIVE, epoch1_CP3_first4_SHA256=cp.FIRST4)
    claim["placement"]["port"] = 29991
    files["CLAIM.json"] = cp.encode(claim)
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    files["SOURCE_FILES.json"] = cp.encode(dict(request=cp.REQUEST, source_parent_SHA256=V3_MANIFEST,
        files={name:dict(bytes=len(raw),SHA256=cp.digest(raw)) for name,raw in files.items()}))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = render()
    if args.output is not None:
        args.output.mkdir()
        for name, data in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(data)
    print(cp.encode(dict(request=cp.REQUEST,label=cp.LABEL,files=len(files),
                         bytes=sum(map(len,files.values())),manifest_SHA256=cp.digest(files["SOURCE_FILES.json"]),
                         target_launched=False)).decode(),end="")


if __name__ == "__main__":
    main()

"""Render one native CP3–25 continuation from actual GPU CP3 and CPU ordering evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE.with_name("training-internvl-gqa56-cp2-to-cp3-native-v1")
sys.path.append(str(BASE))
import series_contract as series
from cp3_contract import digest, encode


CP3_SOURCE="c6c615a6188544568a647fe63837ec2c858cbd3d447086b947fb9e316d91c506"
CP3_RUN="4c23112f1a064af6bfd27d23e55ef89e"
CP3_LABEL="training-internvl-gqa56-cp2-to-cp3-native-v1"
CPU_RECEIPT="9190231e38ce2e36e132ee2c907753c6c7caae3a8b9e9490021179398b8f1b9d"
RECURRENCE_RECEIPT="8c0a235eb9eb4e330d3a326caccc645009c4bf49222066ad7278420d99e566fd"
RECURRENCE_PATH=REPO/"local-only/internvl-arrow-4738-independent-and-skip-route-89f331c266d340eaa1d48e2b22ba4c1e.json"
SKIP_RECEIPT="246fc258c8368634cf43246546a2a07ec405df45fd309595430b5c4ad326e809"
SKIP_PATH=REPO/"local-only/internvl-native-resume-skip-providers-7f83f51c82454f588f620519733c2c5d.json"
CALL_RECEIPT="bb3aabaf2885593b85dc6f5434ea134be24e009e6497f7a40c9fe0af9170bebc"
CALL_PATH=REPO/"local-only/internvl-native-resume-skip-exact-call-e0afbcbf632f461abf25544294b7b972.json"


def skip_binding(path):
    raw=path.read_bytes()
    series.need(digest(raw)==SKIP_RECEIPT,"EXACT_ORIGINAL_4757_SKIP_PROVIDER_RECEIPT")
    wire=json.loads(raw)
    response=wire["response"]
    series.need(wire["authority_verified"] is True and wire["remote_received"] is True and
                wire["rc"]==response["rc"]==response["srun_rc"]==0 and
                response["accounting"]==[["1103606.4757","internvl-resume-provider-7f83f51c8245",
                  "COMPLETED","0:0","0","f13r1n14","cpu=1,mem=1G,node=1",
                  "2026-11-27T19:36:52","2026-11-27T19:36:52"]],"EXACT_ORIGINAL_4757_ZERO_GRES_TERMINAL")
    rows=response["providers"]
    sources={}
    for key,module in (("trainer","transformers"),("accelerate","accelerate")):
        item=rows[module]
        sources[key]=dict(bytes=item["bytes"],sha256=item["SHA256"],path=item["path"])
    series.need(sources=={
        "trainer":dict(path="/usr/local/lib/python3.10/dist-packages/transformers/trainer.py",
                       bytes=215477,sha256="17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a"),
        "accelerate":dict(path="/usr/local/lib/python3.10/dist-packages/accelerate/data_loader.py",
                          bytes=66830,sha256="e08c5f5cd3f7a8b8aaece03ad51f1c37994e390faf47d368281ad49c31bc84d1"),
    },"EXACT_DEPLOYED_SKIP_PROVIDER_PATH_AND_CONTENT")
    skip={row["name"]:row for row in rows["accelerate"]["functions"]
          if row["name"] in ("SkipBatchSampler","skip_first_batches")}
    series.need(set(skip)=={"SkipBatchSampler","skip_first_batches"} and
                skip["skip_first_batches"]["range"]==[1393,1469] and
                skip["SkipBatchSampler"]["range"]==[1330,1350] and
                {part["text"] for part in skip["SkipBatchSampler"]["selected"]}>={"self.batch_sampler = batch_sampler",
                  "self.skip_batches = skip_batches","if index >= self.skip_batches:"},
                "EXACT_DEPLOYED_NATIVE_SKIP_STRUCTURE")
    call=CALL_PATH.read_bytes()
    series.need(digest(call)==CALL_RECEIPT,"EXACT_ORIGINAL_4775_TRAINER_CALL_RECEIPT")
    wire=json.loads(call)
    response=wire["response"]
    series.need(wire["authority_verified"] is True and wire["remote_received"] is True and
                wire["rc"]==response["rc"]==response["srun_rc"]==0 and
                response["accounting"]==[["1103606.4775","internvl-resume-provider-e0afbcbf632f",
                   "COMPLETED","0:0","1","f13r1n14","cpu=1,mem=1G,node=1",
                   "2026-11-27T19:40:58","2026-11-27T19:40:59"]],
                "EXACT_ORIGINAL_4775_ZERO_GRES_TERMINAL")
    providers=response["providers"]
    for key,module in (("trainer","transformers"),("accelerate","accelerate")):
        item=providers[module]
        series.need(sources[key]==dict(path=item["path"],bytes=item["bytes"],sha256=item["SHA256"]),
                    "SAME_PINNED_RUNTIME_PROVIDERS_ACROSS_BOTH_TARGET_PROBES")
    methods={entry["name"]:entry for entry in providers["transformers"]["functions"]}
    expected={
        "_run_epoch":("c704c082dae4b742beb3787afb7636c247294aefbe5803b79f02994ab241221c",[1653,1816],
                      {1678:"train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)",
                       1709:"self._load_rng_state(resume_from_checkpoint)"}),
        "_init_training_state":("4d8fa053043f4c4b9743e862c53376e803a3fbd7250922f6d15167b276e22683",[1524,1553],
                      {1546:"epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)",
                       1548:"steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch",
                       1549:"steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps"}),
    }
    for name,(sha,lines,anchors) in expected.items():
        actual=methods[name]
        selected={row["line"]:row["text"] for row in actual["selected"]}
        series.need((actual["source_SHA256"],actual["range"])==(sha,lines) and
                    all(selected.get(no)==text for no,text in anchors.items()),"EXACT_4775_TRAINER_NATIVE_CURSOR")
    return dict(qualified=True,source_files=sources,original_receipt_sha256=SKIP_RECEIPT,
                original_cpu_step="1103606.4757",trainer_call_cpu_step="1103606.4775",
                trainer_call_receipt_sha256=CALL_RECEIPT)


def bind_originals(schedule_receipt):
    archive=REPO/"results/bw1000/2026-11-28/specialized-internvl-4728-actual-cp2-to-cp3-eight-dcu-independent.json"
    raw=archive.read_bytes()
    series.need(digest(raw)==series.CP3_ARCHIVE,"EXACT_4728_PUBLIC_ARCHIVE")
    cp=json.loads(raw)
    series.need(cp["dispatch_step"]=="1103606.4728" and cp["exact_post_resources_released"] is True and
                cp["actual_native_end"]["checkpoint_named_tensor_fingerprint"]==series.CP3_FP,
                "EXACT_4728_TERMINAL_CP3")
    raw=schedule_receipt.read_bytes()
    series.need(digest(raw)==CPU_RECEIPT,"EXACT_ORIGINAL_4738_CPU_TRANSPORT_RECEIPT")
    wire=json.loads(raw)
    response=wire["response"]
    series.need(wire["authority_verified"] is True and wire["remote_received"] is True and
                wire["rc"]==response["rc"]==response["srun_rc"]==0 and response["accounting"]==[series.CPU_ROW]
                and response["manifest_sha256"]==series.SOURCE_MANIFEST
                and response["output/result.json"]["sha256"]==series.CPU_RESULT
                and response["output/steps4-25-plan.private.json"]==dict(bytes=141110,sha256=series.CPU_PLAN),
                "EXACT_4738_ACTUAL_TERMINAL_AND_OUTPUT_PINS")
    fields=response["output/result.json"]["fields"]
    series.validate_cpu_result(dict(fields,request=response["request"]),bounded_projection=True)
    independent=RECURRENCE_PATH.read_bytes()
    series.need(digest(independent)==RECURRENCE_RECEIPT,"EXACT_4738_INDEPENDENT_RECURRENCE")
    bound=json.loads(independent)
    row=bound["response"]
    series.need(bound["authority_verified"] is True and bound["rc"]==row["rc"]==0 and
                row["output_plan_sha"]==series.CPU_PLAN and row["accounting"]==[series.CPU_ROW]
                and (row["qualified_next_update_slots"],row["physical_slots"],row["epoch_count"],row["updates"],
                     row["internal_rows"],row["data_seed"])==(704,768,12,22,0,42)
                and row["actual_training_forward"] is False,"EXACT_4738_INDEPENDENT_RECURRENCE_LINEAGE")


def original_render():
    spec=importlib.util.spec_from_file_location("internvl_original_cp3",BASE/"render.py")
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    files=renderer.render()
    series.need(digest(files.pop("SOURCE_FILES.json"))==CP3_SOURCE,"EXACT_ACTUAL_4728_SOURCE_CLOSURE")
    return renderer,files


def render(schedule_receipt):
    bind_originals(schedule_receipt)
    old,files=original_render()
    edit,cut=old.edit,old.cut
    for name in ("callback.py","series_contract.py","series_live.py","series_result.py"):
        if name=="callback.py":
            previous=files[name].decode()
            files["resume_common.py"]=previous[:previous.index("    def on_train_begin(")].encode()
        files[name]=(HERE/name).read_bytes()
    files.pop("cp2-archive.json")
    files["SKIP_SOURCE.json"]=encode(skip_binding(SKIP_PATH))
    edit(files,"resume_common.py",[
        ('import cp3_contract as contract','import series_contract as contract\nfrom cp3_contract import optimizer_mapping'),
        ('contract.CP2_FP','contract.CP3_FP'),
        ('mapped = contract.optimizer_mapping(optimizer, named, saved)','mapped = optimizer_mapping(optimizer, named, saved)'),
        ('original["step"].item() == 2','original["step"].item() == 3'),
        ('plan["last_epoch"] == 2 and plan["_step_count"] == 3','plan["last_epoch"] == 3 and plan["_step_count"] == 4'),
        ('plan["_last_lr"] == [contract.LR for _ in optimizer.param_groups]',
         'plan["_last_lr"] == [contract.lr(3) for _ in optimizer.param_groups]'),
        ('contract.digest(contract.encode(', 'digest(contract.encode('),
        ('from cp3_contract import optimizer_mapping','from cp3_contract import digest, optimizer_mapping'),
    ])
    edit(files,"cp3_contract.py",[
        ('CHECKPOINT = Path("/gqa-recovery-cp2")','CHECKPOINT = Path("/gqa-recovery-cp3")'),
        ('LR = 9.957224306869053e-05','LR = 1e-4 * (0.5 * (1 + math.cos(math.pi * 2 / 24)))'),
        ('"internvl_gqa56_cp3_observed_engineering"', '"internvl_gqa56_cp25_series_engineering"'),
    ])
    for name in ("scope.py","control_once.py"):
        edit(files,name,[(CP3_RUN,series.REQUEST),(CP3_LABEL,series.LABEL),("29991",str(series.PORT))])
    edit(files,"scope.py",[
        ("internvl_gqa56_cp3_observed_engineering","internvl_gqa56_cp25_series_engineering"),
    ])
    edit(files,"node.py",[
        ("from cp3_contract import ORIGINAL_CP2, CHECKPOINT",
         "from series_contract import CP3_ROOT, CPU_ROOT, CHECKPOINT"),
        ('f"{ASSETS}:{ASSETS}:ro",f"{ORIGINAL_CP2}:{CHECKPOINT}:ro",',
         '''f"{ASSETS}:{ASSETS}:ro",f"{CP3_ROOT/'train'/'checkpoint-3'}:{CHECKPOINT}:ro",
           *(f"{CP3_ROOT/name}:{CP3_ROOT/name}:ro" for name in ("native-outcome.json","checkpoint-3-files.json")),
           *(f"{CPU_ROOT/name}:{CPU_ROOT/name}:ro" for name in ("result.json","steps4-25-plan.private.json")),'''),
        ("planned_updates=1,planned_micro_forwards_by_rank=4","planned_updates=22,planned_micro_forwards_by_rank=88")
    ])
    edit(files,"owner_gate.py",[
        ('from cp3_contract import ARROW_RESULT, ARROW_ROOT, LABEL, REQUEST, need',
         'from series_contract import CPU_RESULT, CPU_ROOT, CPU_ROW, LABEL, REQUEST, need'),
        ("port=29991", f"port={series.PORT}"),
        ('CPU_ROW = ["1103606.4726", "internvl-gqa56-arrow-e1-c6d2ed29e945", "COMPLETED", "0:0",\n'
         '           "29", "f13r1n14", "cpu=8,mem=24G,node=1", "2026-11-27T18:28:33", "2026-11-27T18:29:02"]\n',""),
        ("EXACT_4726_ZERO_GRES_TERMINAL","EXACT_4738_ZERO_GRES_TERMINAL"),
        ("CPU_ROW[0], ARROW_RESULT","CPU_ROW[0], CPU_RESULT"),
        ('>= 4708','>= 4730'),
        ('pin(ARROW_ROOT/"result.json", expected=ARROW_RESULT, extent=6740)',
         'pin(CPU_ROOT/"result.json", expected=CPU_RESULT, extent=9579)'),
        ('    owner = json.loads(pin(here/"owner-authority.json", expected=authority_sha, cap=8192))',
         '''    skip = json.loads(pin(here/"SKIP_SOURCE.json",cap=16384))
    need(skip["qualified"] is True and set(skip["source_files"])=={"trainer","accelerate"},
         "QUALIFIED_DEPLOYED_SKIP_SOURCE_REQUIRED_BEFORE_OWNER")
    owner = json.loads(pin(here/"owner-authority.json", expected=authority_sha, cap=8192))'''),
    ])
    edit(files,"control_once.py",[("Exact prior .4726 accounting is absent","Exact prior .4738 accounting is absent")])
    edit(files,"slot_guard.py",[
        ("def seal_checkpoint(root, check_deadline=lambda: None):",
         "def seal_checkpoint(root, step, check_deadline=lambda: None):"),
        ('    need(root.resolve(strict=True)==root and stat.S_ISDIR(root.lstat().st_mode),"EXACT_NATIVE_CP2_PATH")',
         '    need(step in (10,20,25) and root.name==f"checkpoint-{step}" and root.resolve(strict=True)==root '
         'and stat.S_ISDIR(root.lstat().st_mode),"EXACT_NATIVE_CHECKPOINT_PATH")'),
        ('==(3,25),"EXACT_CP3_STEP_AND_SCHEDULE")','==(step,25),"EXACT_SEALED_STEP_AND_SCHEDULE")'),
        ("return dict(step=3,native_schedule_steps=25","return dict(step=step,native_schedule_steps=25"),
    ])
    edit(files,"native.py",[
        ('''    import cp3_contract
    cp2_members=cp3_contract.verify_cp2(HERE/"cp2-archive.json",cp3_contract.CHECKPOINT,pin,
                                       lambda: seconds_left(epoch,NATIVE))''',
         '''    import cp3_contract
    import series_contract as series
    import series_result
    cp3_members=series.verify_checkpoint(series.CP3_ROOT,series.CHECKPOINT,pin,lambda:seconds_left(epoch,NATIVE))
    plan=series.load_plan(series.CPU_ROOT,pin)'''),
        ('planned_update_slots=1,original_schedule_steps=25,verified_original_cp2_members=cp2_members,\n'
         '               planned_input_slots=32',
         'planned_update_slots=22,original_schedule_steps=25,verified_original_cp3_members=cp3_members,\n'
         '               planned_input_slots=704'),
    ])
    cut(files,"native.py","    reports=[]\n    partial=[]",'    write_once(RUN/"native-outcome.json",report)','''    reports,partial,errors=series_result.collect(RUN,plan,lambda:seconds_left(epoch,NATIVE))
    complete=(child is not None and child.returncode==0 and error is None and len(reports)==8 and not errors
              and cleanup["owned_processes_remaining"]==0 and not cleanup["termination_needed"])
    sealed={}
    if complete:
        try:
            sealed=series_result.seal(RUN,reports,slot_guard.seal_checkpoint,lambda:seconds_left(epoch,NATIVE))
        except (ValueError,KeyError,OSError,TypeError):
            error="CHECKPOINT_10_20_25_INVENTORY_OR_DURABILITY_FAILED"
            complete=False
    report=dict(request=REQUEST,actual_step=intent["step"],native_returncode=child.returncode if child else None,
                error=error,receipt_validation_errors=errors,continuation_complete=complete,
                starting_checkpoint=3,nominal_steps=25,planned_updates=22,planned_ranks=8,planned_input_slots=704,
                qualified_rank_receipts=len(reports),qualified_domain_input_slots=704 if complete else None,
                actual_rank_forward_witnesses=sum(row["forward_slots"] for row in partial),
                actual_rank_optimizer_witnesses=sum(len(row["optimizer_steps"]) for row in partial),
                checkpoint_inventory_SHA256s=sealed,ordered_schedule_SHA256=series.CPU_PLAN,
                partial_rank_progress=partial,rank_receipts=reports,cleanup=cleanup,formal17_qualified=False,
                training_effect_GO=False,formal_performance=False)
''')
    claim=json.loads(files["CLAIM.json"])
    claim.update(request=series.REQUEST,label=series.LABEL,schema="INTERNVL_GQA56_CP3_TO_CP25_NATIVE_V1",
                 state="NO_ADOPTION_WITHOUT_FRESH_EXPLICIT_OWNER_AND_ACTUAL_4738",
                 root_GO=False,GPU_GO=False,planned_updates=22,planned_nonzero_updates=22,
                 planned_input_slots=704,planned_microbatches=704,actual_cpu_arrow_source_step=series.CPU_ROW[0],
                 actual_cpu_arrow_result_SHA256=series.CPU_RESULT,prior_zero_gres_step=series.CPU_ROW[0],
                 prior_zero_gres_result_SHA256=series.CPU_RESULT,prior_zero_gres_archive_SHA256=CPU_RECEIPT,
                 original_cp3_archive_SHA256=series.CP3_ARCHIVE,
                 original_cp3_native_SHA256=series.CP3_NATIVE,continuation_plan_SHA256=series.CPU_PLAN,
                 continuation_independent_recurrence_SHA256=RECURRENCE_RECEIPT,
                 continuation_checkpoint_steps=list(series.SAVE_STEPS),native_schedule_steps=25,
                 skip_source_qualified=True,original_skip_source_receipt_SHA256=SKIP_RECEIPT,
                 original_trainer_call_receipt_SHA256=CALL_RECEIPT)
    claim["placement"]["port"]=series.PORT
    del claim["original_cp2_archive_SHA256"],claim["epoch1_CP3_first4_SHA256"]
    files["CLAIM.json"]=encode(claim)
    for name,data in files.items():
        if name.endswith(".py"):
            compile(data,name,"exec")
    files["SOURCE_FILES.json"]=encode(dict(request=series.REQUEST,source_parent_SHA256=CP3_SOURCE,
        files={name:dict(bytes=len(data),SHA256=digest(data)) for name,data in files.items()}))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=render(args.schedule_receipt)
    if args.output is not None:
        args.output.mkdir()
        for name,data in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(data)
    print(encode(dict(request=series.REQUEST,label=series.LABEL,files=len(files),
                     bytes=sum(map(len,files.values())),manifest_SHA256=digest(files["SOURCE_FILES.json"]),
                     target_launched=False)).decode(),end="")


if __name__=="__main__":
    main()

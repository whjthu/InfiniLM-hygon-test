"""Render the native GQA56 capture and guards for review without permitting dispatch."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUEST = "d955fae7a2644ff8aececf23b543b386"
LABEL = "training-internvl-gqa56-first2-native-candidate-v1"
PORT = 29983
SPLIT = "internvl-gqa64-train56-internal8-aa810b24efcc4a5ab04a3cdd1c7825a4/output"
BASE_SHA = "49153458e62d6ff8ee3a14ac5e8fda248ff6557d570a360f0dea6a6abf2ac1fc"


def digest(raw):
    return sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()


def edit(files,name,substitutions):
    body = files[name].decode()
    for old, new in substitutions:
        if body.count(old)!=1:
            raise ValueError(f"Exact {name} source anchor changed: {old[:60]}")
        body = body.replace(old,new)
    files[name] = body.encode()


def render():
    sys.path.insert(0,str(REPO/"tools"))
    import render_internvl_gqa64_first2 as base

    files = base.render()
    if digest(files.pop("SOURCE_FILES.json")) != BASE_SHA:
        raise ValueError("Original first-two manifest changed")
    files["slot_guard.py"] = (HERE/"slot_guard.py").read_bytes()
    edit(files,"scope.py",[
        (base.REQUEST,REQUEST),(base.LABEL,LABEL),('29967',str(PORT)),
        ('DATASET = DOMAIN_DATA / "train.jsonl"',
         f'GQA_SPLIT = RESULTS / {SPLIT!r}\nORIGINAL64 = DOMAIN_DATA / "train.jsonl"\n'
         'DATASET = GQA_SPLIT / "proposed-exposures64.jsonl"'),
        ('internvl_usgs72_first2_engineering','internvl_gqa56_first2_engineering'),
        ('    argv[argv.index("--model")+1]=str(MODEL)',
         '''    argv[argv.index("--model")+1]=str(MODEL)
    baseline=str(ASSETS/"plugins/swift_callback_nvidia_baseline_v2.py")
    need(argv.count("nvidia_baseline")==argv.count(baseline)==1,
         "Frozen timing-only finalizer tokens differ")
    argv.remove("nvidia_baseline")
    argv.remove(baseline)''')])
    edit(files,"domain_checks.py",[
        ('from scope import DATASET, GQA_FINGERPRINT_RECEIPT, need',
         'from scope import ORIGINAL64, GQA_SPLIT, GQA_FINGERPRINT_RECEIPT, need, pin\nimport slot_guard'),
        ('def verified_domain():\n    return load_roster',
         'def verified_split():\n    roster = load_roster'),
        (f', DATASET, {base.TRAIN_EXTENT},',f', ORIGINAL64, {base.TRAIN_EXTENT},'),
        ('\n\ndef before_sample(','''
    original = pin(ORIGINAL64,expected="78de19307549affa877be85756e7077bbb70d8d9feea09e1a8214157ee2fa000",extent=19607)
    return slot_guard.load(GQA_SPLIT,original,roster,pin)


def verified_domain():
    return verified_split()[0]


def before_sample(''')])
    edit(files,"node.py",[
        ('    from scope import GQA_FINGERPRINT_RECEIPT','    from scope import GQA_FINGERPRINT_RECEIPT, GQA_SPLIT'),
        ('f"{GQA_FINGERPRINT_RECEIPT}:{GQA_FINGERPRINT_RECEIPT}:ro",',
         '''f"{GQA_FINGERPRINT_RECEIPT}:{GQA_FINGERPRINT_RECEIPT}:ro",
           *(f"{GQA_SPLIT/name}:{GQA_SPLIT/name}:ro" for name in
             ("SPLIT.json","proposed-exposures64.jsonl","proposed-slots.json","train56.jsonl")),''')])
    edit(files,"callback.py",[
        ('from domain_checks import PREPARE_SHA, after_sample, before_sample, verified_domain',
         'from domain_checks import PREPARE_SHA, after_sample, before_sample, verified_split\nimport slot_guard'),
        ('        roster = verified_domain()', '        roster,self.domain_slots = verified_split()'),
        ('"Runtime GQA64 roster differs"', '"Runtime GQA56 roster differs"'),
        ('                return before_sample(inputs, roster, lambda value: fingerprint(cpu(value)))',
         '''                observed=before_sample(inputs, roster, lambda value: fingerprint(cpu(value)))
                slot_guard.exact_source(self.domain_slots,self.rank,self.inputs,observed)
                return observed'''),
        ('            source = self.consensus_check(lambda: after_sample(value["kwargs"], self.pending_source, fingerprint))',
         '''            def verify_forward():
                need(value["micro_ordinal"]==self.inputs+1 and
                     value["global_step"]==self.inputs//4,"EXACT_NATIVE_ACCUMULATION_FORWARD")
                observed=after_sample(value["kwargs"],self.pending_source,fingerprint)
                return slot_guard.exact_source(self.domain_slots,self.rank,self.inputs,observed)
            source=self.consensus_check(verify_forward)'''),
        ('        self.capture.guarded(self.capture.before_optimizer)()',
         '''        def verify_boundary():
            cap=self.capture
            rates=[float(group["lr"]) for group in self.trainer.optimizer.param_groups]
            need(all(math.isfinite(rate) for rate in rates),"Finite native LR required")
            slot_guard.optimizer_boundary(state.global_step,cap.updates,cap.inputs,cap.outputs,cap.pending_source,rates)
        self.capture.guarded(self.capture.consensus_check)(verify_boundary)
        self.capture.guarded(self.capture.before_optimizer)()'''),
        ('        self.capture=None','        self.capture=None\n        self.saved_cp2=False'),
        ('        self.capture.guarded(self.capture.attach)(model)',
         '''        observed=self.capture.consensus_check(lambda: slot_guard.sampler_identity(kwargs.get("train_dataloader")))
        self.capture.write_json("sampler-identity.json",dict(rank=rank,request=REQUEST,**observed))
        self.capture.guarded(self.capture.attach)(model)'''),
        ('            control.should_training_stop=True', '            control.should_training_stop=True\n            control.should_save=True'),
        ('    def on_train_end(self,args,state,control,**kwargs):',
         '''    def on_save(self,args,state,control,**kwargs):
        need(state.global_step==2 and not self.saved_cp2,"EXACT_CP2_NATIVE_SAVE_CALLBACK")
        self.saved_cp2=True
        return control

    def on_train_end(self,args,state,control,**kwargs):
        need(self.saved_cp2,"MISSING_NATIVE_CP2_SAVE_CALLBACK")''')])
    edit(files,"native.py",[
        ('import owned_processes','import owned_processes\nimport slot_guard'),
        ('def report_rank(root,rank,source_rows):','def report_rank(root,rank,source_rows,domain_slots):'),
        ('            samples.append(src["ordinal"])','            samples.append(src)'),
        ('    need(len(samples) == len(set(samples)) == 8, "Repeated or missing native samples in a rank")',
         '    slot_guard.exact_terminal(domain_slots,rank,samples)'),
        ('sample_ordinals=samples,','sample_ordinals=[sample["ordinal"] for sample in samples],'),
        ('    report=dict(request=REQUEST,actual_step=intent["step"]',
         '''    cp2=None
    if complete:
        try:
            cp2=slot_guard.seal_checkpoint(RUN/"train"/"checkpoint-2",lambda: seconds_left(epoch,NATIVE))
            write_once(RUN/"checkpoint-2-files.json",cp2)
        except (ValueError,KeyError,OSError,TypeError):
            error="CP2_INVENTORY_OR_DURABILITY_FAILED"
            complete=False
    report=dict(checkpoint2_sealed=cp2 is not None and complete,request=REQUEST,actual_step=intent["step"]'''),
        ('    verified=binding()','    verified=binding()\n    domain_slots=verified_split()[1]'),
        ('from domain_checks import verified_domain','from domain_checks import verified_domain, verified_split'),
        ('reports.append(report_rank(output,rank,verified["domain_source_rows"]))',
         'reports.append(report_rank(output,rank,verified["domain_source_rows"],domain_slots))'),
        ('and len({slot for row in reports for slot in row["sample_ordinals"]})==64)',
         'and sum(len(row["sample_ordinals"]) for row in reports)==64)'),
        (f'expected="{base.STAGED_TRAIN_SHA}",extent={base.TRAIN_EXTENT}',
         'expected="28643ee29fc0a2d59ccf26883a66bd870dbb89be1f7f710c23dc9aca0727d174",extent=19599')])
    old_claim = json.loads(files["CLAIM.json"])
    old_claim.update(request=REQUEST,label=LABEL,state="DISABLED_PENDING_ACTUAL_NATIVE_SAMPLER_AND_FRESH_PLACEMENT",
                     planned_distinct_train_rows=56,planned_repeat_slots=8,planned_internal_rows=8,GPU_GO=False,
                     native_order_verified=False,previous_train_input_SHA256=base.STAGED_TRAIN_SHA)
    old_claim["placement"]["port"] = PORT
    files["CLAIM.json"] = encode(old_claim)
    # The original controller scopes are retained for inspection. A fresh owner must consume this package.
    edit(files,"control_once.py",[(base.REQUEST,REQUEST),(base.LABEL,LABEL)])
    files["control_once.py"] = b'raise SystemExit("DISABLED_REVIEW_SOURCE_NO_NATIVE_SAMPLER_AUTHORITY")\n' + files["control_once.py"]
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    files["SOURCE_FILES.json"] = encode(dict(request=REQUEST,files={
        name:dict(bytes=len(raw),SHA256=digest(raw)) for name,raw in files.items()}))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=render()
    if args.output is not None:
        args.output.mkdir()
        for name,raw in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(raw)
    print(encode(dict(request=REQUEST,files=len(files),manifest_SHA256=digest(files["SOURCE_FILES.json"]),
                      actual_launch=False,native_order_verified=False)).decode(),end="")


if __name__=="__main__":
    main()

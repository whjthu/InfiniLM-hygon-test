"""Render the new native GQA56 two-update compute package from the pinned v1 source."""
import argparse
import importlib.util
import json
from pathlib import Path

import live_guard


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE.with_name("training-internvl-gqa56-first2-native-candidate-v1")/"render.py"
BASE_SOURCE_SHA="454cdc644c913316d417c82d356b2317979d165a447483eef87a51c0e80db7f0"
BASE_MANIFEST_SHA="5efc12f7d23e61f9d6c0c28b974f014a5cb5cba6a7d9405c9622dde15d0a21b6"
REQUEST="883dca77519d4ef2af89a941248fca12"
LABEL="training-internvl-gqa56-first2-native-observed-v2"
PORT=29985
REREAD_SHA="4b7c17f9c65018fe6ce0a80171eeac7a274278bdb7a88e53a0539034464ca37f"


def edit(files,name,changes):
    text=files[name].decode()
    for old,new in changes:
        live_guard.need(text.count(old)==1,"Exact "+name+" source anchor differs: "+old[:60])
        text=text.replace(old,new)
    files[name]=text.encode()


def render():
    raw=BASE.read_bytes()
    live_guard.need(live_guard.digest(raw)==BASE_SOURCE_SHA,"EXACT_FIRST2_BASE_SOURCE")
    spec=importlib.util.spec_from_file_location("internvl_gqa56_original_candidate",BASE)
    baseline=importlib.util.module_from_spec(spec)
    exec(compile(raw,str(BASE),"exec"),baseline.__dict__)
    files=baseline.render()
    live_guard.need(live_guard.digest(files.pop("SOURCE_FILES.json"))==BASE_MANIFEST_SHA,"EXACT_FIRST2_BASE_MANIFEST")
    files["live_guard.py"]=(HERE/"live_guard.py").read_bytes()
    edit(files,"scope.py",[
        (baseline.REQUEST,REQUEST),(baseline.LABEL,LABEL),(str(baseline.PORT),str(PORT)),
        ('DATASET = GQA_SPLIT / "proposed-exposures64.jsonl"',
         'DATASET = GQA_SPLIT / "proposed-exposures64.jsonl"\n'
         f'ARROW_AUTHORITY = Path({live_guard.ARROW_ROOT!r})'),
        ('internvl_gqa56_first2_engineering','internvl_gqa56_first2_observed_engineering'),
    ])
    edit(files,"control_once.py",[
        (baseline.REQUEST,REQUEST),(baseline.LABEL,LABEL),
        ('DISABLED_REVIEW_SOURCE_NO_NATIVE_SAMPLER_AUTHORITY','DISABLED_UNTIL_FRESH_OWNER_AND_ZERO_GRES_QUALIFICATION'),
    ])
    edit(files,"domain_checks.py",[
        ('from scope import ORIGINAL64, GQA_SPLIT, GQA_FINGERPRINT_RECEIPT, need, pin',
         'from scope import ARROW_AUTHORITY, DATASET, ORIGINAL64, GQA_SPLIT, GQA_FINGERPRINT_RECEIPT, need, pin\n'
         'import json\nimport live_guard'),
        ('    return slot_guard.load(GQA_SPLIT,original,roster,pin)',
         '''    subset,_ = slot_guard.load(GQA_SPLIT,original,roster,pin)
    size,sha=slot_guard.FILES[DATASET.name]
    exposures=pin(DATASET,expected=sha,extent=size)
    name="proposed-slots.json"
    size,sha=slot_guard.FILES[name]
    proposed=json.loads(pin(GQA_SPLIT/name,expected=sha,extent=size))
    by_rank,arrow=live_guard.authority(ARROW_AUTHORITY,exposures,original,proposed,subset,pin)
    return subset,by_rank,arrow'''),
    ])
    edit(files,"node.py",[
        ('from scope import GQA_FINGERPRINT_RECEIPT, GQA_SPLIT','from scope import ARROW_AUTHORITY, GQA_FINGERPRINT_RECEIPT, GQA_SPLIT'),
        ('("SPLIT.json","proposed-exposures64.jsonl","proposed-slots.json","train56.jsonl")),',
         '''("SPLIT.json","proposed-exposures64.jsonl","proposed-slots.json","train56.jsonl")),
           *(f"{ARROW_AUTHORITY/name}:{ARROW_AUTHORITY/name}:ro" for name in
             ("result.json","physical-rank-matrix.private.json")),'''),
    ])
    edit(files,"callback.py",[
        ('import slot_guard','import slot_guard\nimport live_guard'),
        ('def bind_domain(self, agrees, source_rows):','def bind_domain(self, agrees, source_rows, loader):'),
        ('        roster,self.domain_slots = verified_split()','        roster,self.domain_slots,arrow = verified_split()'),
        ('        self.pending_source = None\n        prepare =', '''        self.pending_source = None
        self.live_cursor=live_guard.LiveCursor(self.rank,self.domain_slots[self.rank],arrow)
        observed=self.consensus_check(lambda:self.live_cursor.attach(loader,pin,torch.distributed))
        self.write_json("sampler-identity.json",dict(rank=self.rank,request=REQUEST,**observed))
        prepare ='''),
        ('                slot_guard.exact_source(self.domain_slots,self.rank,self.inputs,observed)\n                return observed',
         '                return self.live_cursor.prepare(self.inputs,observed)'),
        ('        super().detach()\n        if getattr(self, "original_prepare", None) is not None:',
         '''        super().detach()
        cursor=getattr(self,"live_cursor",None)
        if cursor is not None:
            cursor.detach()
        if getattr(self, "original_prepare", None) is not None:'''),
        ('                return slot_guard.exact_source(self.domain_slots,self.rank,self.inputs,observed)',
         '''                observed.update(logical=self.pending_source["logical"],physical=self.pending_source["physical"])
                return self.live_cursor.forward(self.inputs,observed)'''),
        ('''        observed=self.capture.consensus_check(lambda: slot_guard.sampler_identity(kwargs.get("train_dataloader")))
        self.capture.write_json("sampler-identity.json",dict(rank=rank,request=REQUEST,**observed))
        self.capture.guarded(self.capture.attach)(model)
        self.capture.guarded(self.capture.bind_domain)(agrees, binding["domain_source_rows"])''',
         '''        self.capture.guarded(self.capture.attach)(model)
        self.capture.guarded(self.capture.bind_domain)(agrees,binding["domain_source_rows"],kwargs.get("train_dataloader"))'''),
        ('            slot_guard.optimizer_boundary(state.global_step,cap.updates,cap.inputs,cap.outputs,cap.pending_source,rates)',
         '''            slot_guard.optimizer_boundary(state.global_step,cap.updates,cap.inputs,cap.outputs,cap.pending_source,rates)
            cap.live_cursor.confirmed(cap.inputs)'''),
        ('        need(self.saved_cp2,"MISSING_NATIVE_CP2_SAVE_CALLBACK")',
         '''        need(self.saved_cp2,"MISSING_NATIVE_CP2_SAVE_CALLBACK")
        self.capture.live_cursor.confirmed(8)'''),
    ])
    claim=json.loads(files["CLAIM.json"])
    claim.update(request=REQUEST,label=LABEL,state="DISABLED_PENDING_FRESH_OWNER_AND_ZERO_GRES_QUALIFICATION",
                 case="T03.domain-gqa56-native-sft",schema="INTERNVL_GQA56_FIRST2_OBSERVED_HIP_V2",root_GO=False,
                 actual_cpu_arrow_source_step="1103606.4699",actual_cpu_arrow_result_SHA256=live_guard.RESULT_PIN,
                 native_occurrence_matrix_SHA256=live_guard.MATRIX_PIN,independent_matrix_proof_SHA256=REREAD_SHA,
                 dataset_sha256="28643ee29fc0a2d59ccf26883a66bd870dbb89be1f7f710c23dc9aca0727d174",
                 expected_unique_source_by_update=[31,31],production_workers=4,production_order_verified=False)
    claim["placement"]["port"]=PORT
    files["CLAIM.json"]=live_guard.encode(claim)
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    files["SOURCE_FILES.json"]=live_guard.encode(dict(request=REQUEST,files={
        name:dict(bytes=len(raw),SHA256=live_guard.digest(raw)) for name,raw in files.items()}))
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
    result=dict(request=REQUEST,files=len(files),bytes=sum(map(len,files.values())),
                manifest_SHA256=live_guard.digest(files["SOURCE_FILES.json"]),actual_launch=False)
    print(live_guard.encode(result).decode(),end="")


if __name__=="__main__":
    main()

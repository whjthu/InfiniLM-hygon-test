"""Render the native Swift rebuilt-loader successor for offline review."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import skip_cpu


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE.with_name("training-internvl-gqa56-cp3-to-cp25-native-v1")
BASE_MANIFEST="339aaaeb96c671e4afa49d20214508980773312e78d0cdc9416ae985d6b02e14"
SCHEDULE_RECEIPT=REPO/"local-only/internvl-gqa56-arrow-continuation-epochs1to12-once-1ccc01ddf76049dca2e32d9f013be644.json"
FAILED=REPO/"results/bw1000/2026-11-28/specialized-internvl-4783-native-resume-skip-provider-mismatch.json"
FAILED_SHA="39e82fcd1722a11529a9c2077d8a7f2de521dbc9fb399216209b531542d42616"
NATIVE_PRIMITIVES=REPO/"local-only/internvl-4783-swift-exact-rebuilt-loader-1a8c21fb4f734693b857095c79fe5c8c.json"
NATIVE_PRIMITIVES_SHA="fdfbbe8ba92653013e33f29eb54dd763e542a9ffaa4cdfacd773b501c655670d"
SKIP_ARCHIVE=REPO/"results/bw1000/2026-11-28/specialized-internvl-4790-native-swift-skip-independent-704-slots.json"
SKIP_INDEPENDENT=REPO/"local-only/internvl-swift-4790-independent-49caf43e46004a14b1d0dee843791bb1.json"
SKIP_TRANSPORT=REPO/"local-only/internvl-gqa56-native-swift-resume-skip-once-69052f94c9554014821c2cf81e29ebfa.json"
REQUEST="83d7a712b3174ffe9893bb5a1cf8784b"
LABEL="training-internvl-gqa56-cp3-to-cp25-swift-rebuilt-v2"
PORT=29997


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def need(ok,message):
    if not ok:
        raise ValueError(message)


def edit(files,name,changes):
    source=files[name].decode()
    for before,after in changes:
        need(source.count(before)==1,"EXACT_FROZEN_4783_SOURCE_ANCHOR:"+name)
        source=source.replace(before,after)
    files[name]=source.encode()


def cut(source,left,right,replacement):
    start=source.index(left)
    stop=source.index(right,start+len(left))
    return source[:start]+replacement+source[stop:]


def baseline():
    failed=FAILED.read_bytes()
    primitives=NATIVE_PRIMITIVES.read_bytes()
    need(digest(failed)==FAILED_SHA and digest(primitives)==NATIVE_PRIMITIVES_SHA,
         "EXACT_ORIGINAL_FAILURE_AND_AUTHENTICATED_PRIMITIVES")
    verdict=json.loads(failed)
    record=json.loads(primitives)
    need(verdict["source_manifest_sha256"]==BASE_MANIFEST and verdict["source_CP3_unchanged"] is True and
         verdict["recorded_new_optimizer_updates"]==0 and
         verdict["accounting"]["1103606.4783"][2:4]==["FAILED","1:0"] and
         verdict["accounting"]["1103606.4784"][2:4]==["COMPLETED","0:0"] and
         record["authority_verified"] is True and record["remote_received"] is True and record["rc"]==0,
         "EXACT_TERMINAL_NO_UPDATES_AND_CLEAN_POST")
    spec=importlib.util.spec_from_file_location("actual_4783_series_renderer",BASE/"render.py")
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(BASE))
    try:
        spec.loader.exec_module(module)
        files=module.render(SCHEDULE_RECEIPT)
    finally:
        sys.path.remove(str(BASE))
    need(digest(files.pop("SOURCE_FILES.json"))==BASE_MANIFEST,"EXACT_4783_EXECUTABLE_SOURCE")
    return module,files


def checked_skip_evidence():
    raw,independent,original=(file.read_bytes() for file in (SKIP_ARCHIVE,SKIP_INDEPENDENT,SKIP_TRANSPORT))
    need([digest(value) for value in (raw,independent,original)]==
         [skip_cpu.PUBLIC_ARCHIVE,skip_cpu.INDEPENDENT,skip_cpu.ORIGINAL_TRANSPORT] and len(independent)==1171,
         "EXACT_ORIGINAL_4790_AND_INDEPENDENT_EVIDENCE")
    archive=json.loads(raw)
    wrapper=json.loads(independent)
    transport=json.loads(original)
    proof=wrapper["response"]
    actual=transport["response"]
    need(all(wire["authority_verified"] is True and wire["remote_received"] is True and wire["rc"]==0
             for wire in (wrapper,transport)) and proof["rc"]==actual["rc"]==actual["srun_rc"]==0
         and proof["accounting"]==actual["accounting"]==[skip_cpu.ROW]
         and archive["independent_verification"]==proof
         and archive["source_manifest_SHA256"]==actual["manifest_sha256"]==skip_cpu.MANIFEST
         and archive["publication"]==skip_cpu.PUBLICATION and archive["request"]==skip_cpu.REQUEST
         and (proof["rank_count"],proof["epoch_count"],proof["actual_epoch_slots"],
              proof["qualified_next_update_slots"],proof["planned_updates"],proof["internal_rows"])==(8,12,768,704,22,0)
         and (proof["skipped_on_epoch1"],proof["no_dataset_fetch_on_skipped"],proof["actual_metadata_fetches_epoch1"])==(32,0,32)
         and proof["actual_sampler_seed_range"]==[43,54] and proof["production_worker_validation"] is False
         and proof["output_plan_sha"]==skip_cpu.PLAN and archive["prior_CP3_unchanged"] is True,
         "EXACT_4790_INDEPENDENT_REBUILT_CPU_ROUTE")
    result=actual["output/result.json"]
    need((result["sha256"],result["bytes"])==(skip_cpu.RESULT,10063),"EXACT_4790_ACTUAL_RESULT")
    skip_cpu.validate_result(dict(result["fields"],request=actual["request"]))
    return skip_cpu.binding()


def render():
    old,files=baseline()
    for name in ("swift_watch.py","live_hooks.py","skip_cpu.py"):
        files[name]=(HERE/name).read_bytes()
    for name in ("scope.py","control_once.py","series_contract.py"):
        edit(files,name,[(old.series.REQUEST,REQUEST),(old.series.LABEL,LABEL),
                         (str(old.series.PORT),str(PORT))])
    source=files["series_live.py"].decode()
    source=cut(source,"    def attach(","    def prepare(",
               """    def attach(self,loader,pin,dist,owner):
        return live_hooks.attach(self,loader,pin,dist,owner)

""")
    source=source[:source.index("    def detach(")]+"""    def detach(self,strict=True):
        return live_hooks.detach(self,strict)
"""
    files["series_live.py"]=source.encode()
    edit(files,"series_live.py",[
        ("from series_contract import EPOCHS, need",
         "from series_contract import EPOCHS, need\nimport live_hooks"),
        ("def __init__(self,rank,plan,order):","def __init__(self,rank,plan,order,epoch1):"),
        ("        self.skip_consumed=False",
         "        self.skip_consumed=False\n        self.hooks=None\n        self.rebuilt_sampler=None\n"
         "        need(len(epoch1)==8,\"EXACT_ORIGINAL_4726_FULL_RANK_EPOCH\")\n"
         "        self.skipped_epoch1=[(row[\"logical\"],row[\"physical\"]) for row in epoch1[:4]]"),
        ('''                need((epoch==1 and offset<4) or (epoch==12 and offset>=4),"EXACT_BOUNDARY_NONFORWARD_SLOT")''',
         '''                need((epoch==1 and offset<4) or (epoch==12 and offset>=4),"EXACT_BOUNDARY_NONFORWARD_SLOT")
                if epoch==1:
                    need((logical,physical)==self.skipped_epoch1[offset],"EXACT_4726_SKIPPED_EPOCH1_RAW_POSITION")'''),
    ])
    edit(files,"callback.py",[
        ("from swift.callbacks import callbacks_map","from swift.callbacks import callbacks_map\n"
         "from swift.trainers.mixin import SwiftMixin\nimport swift_watch"),
        ("self.cursor=SeriesCursor(self.rank,plans[self.rank],order)",
         "self.cursor=SeriesCursor(self.rank,plans[self.rank],order,original[self.rank])"),
        ('self.cursor.attach(kwargs["train_dataloader"],pin,torch.distributed)',
         'self.cursor.attach(kwargs["train_dataloader"],pin,torch.distributed,self.trainer)'),
        ("callbacks_map[CALLBACK]=ContinueSeries",
         "swift_watch.wrap_native_context(SwiftMixin,pin)\ncallbacks_map[CALLBACK]=ContinueSeries"),
    ])
    files["SWIFT_SKIP_CPU.json"]=encode(checked_skip_evidence())
    edit(files,"node.py",[
        ("from series_contract import CP3_ROOT, CPU_ROOT, CHECKPOINT",
         "from series_contract import CP3_ROOT, CPU_ROOT, CHECKPOINT\n"
         "    from skip_cpu import ROOT as SKIP_CPU_ROOT, PLAN_NAME as SKIP_PLAN_NAME"),
        ('''           *(f"{CPU_ROOT/name}:{CPU_ROOT/name}:ro" for name in ("result.json","steps4-25-plan.private.json")),''',
         '''           *(f"{CPU_ROOT/name}:{CPU_ROOT/name}:ro" for name in ("result.json","steps4-25-plan.private.json")),
           *(f"{SKIP_CPU_ROOT/name}:{SKIP_CPU_ROOT/name}:ro" for name in ("result.json",SKIP_PLAN_NAME)),'''),
    ])
    edit(files,"native.py",[
        ("    plan=series.load_plan(series.CPU_ROOT,pin)",
         '''    from skip_cpu import verify as verify_swift_skip_cpu
    observed_skip_plan=verify_swift_skip_cpu(HERE,pin)
    need(observed_skip_plan==pin(series.CPU_ROOT/"steps4-25-plan.private.json",expected=series.CPU_PLAN,
                                extent=141110,cap=150000),"EXACT_4790_AND_4738_BYTE_EQUAL_704_PLAN")
    plan=series.load_plan(series.CPU_ROOT,pin)'''),
    ])
    edit(files,"owner_gate.py",[
        ("from series_contract import CPU_RESULT, CPU_ROOT, CPU_ROW, LABEL, REQUEST, need",
         "from series_contract import CPU_RESULT, CPU_ROOT, CPU_ROW, LABEL, REQUEST, need\nimport skip_cpu"),
        ("port=29995",f"port={PORT}"),
        (">= 4730",">= 4785"),
        ('    need(row == CPU_ROW, "EXACT_4738_ZERO_GRES_TERMINAL")',
         '''    need(row == CPU_ROW, "EXACT_4738_ZERO_GRES_TERMINAL")
    need(exact_accounting(skip_cpu.ROW[0],run)==skip_cpu.ROW,"EXACT_4790_ZERO_GRES_TERMINAL")'''),
        ("    owner = json.loads(pin(here/\"owner-authority.json\", expected=authority_sha, cap=8192))",
         '''    skip_cpu.verify(here,pin)
    owner = json.loads(pin(here/"owner-authority.json", expected=authority_sha, cap=8192))'''),
    ])
    claim=json.loads(files["CLAIM.json"])
    claim.update(request=REQUEST,label=LABEL,schema="INTERNVL_GQA56_CP3_TO_CP25_SWIFT_REBUILT_V2",
         root_GO=False,GPU_GO=False,state="EXACT_4790_BOUND_FRESH_ROOT_OWNER_STILL_REQUIRED",
         original_4783_failed_manifest_SHA256=BASE_MANIFEST,original_4783_failure_SHA256=FAILED_SHA,
         original_swift_rebuild_primitives_SHA256=NATIVE_PRIMITIVES_SHA,actual_gpu_launched=False,
         verified_4790_zero_gres_metadata=skip_cpu.binding())
    claim["placement"]["port"]=PORT
    files["CLAIM.json"]=encode(claim)
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    files["SOURCE_FILES.json"]=encode(dict(request=REQUEST,source_parent_SHA256=BASE_MANIFEST,
        files={name:dict(bytes=len(raw),SHA256=digest(raw)) for name,raw in files.items()}))
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
    print(encode(dict(request=REQUEST,label=LABEL,files=len(files),bytes=sum(map(len,files.values())),
                      manifest_SHA256=digest(files["SOURCE_FILES.json"]),target_launched=False)).decode(),end="")


if __name__=="__main__":
    main()

"""Render an immutable source-only CPU witness for the actual Swift restart branch."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE.with_name("training-internvl-gqa56-native-arrow-CPU-epochs1to12-v1")
BASE_SHA="a8e6744dc55ebbf02977a8acd335fe7d60a7353af5b3af9d2d5cf2eca07dbfce"
EVIDENCE={
    "results/bw1000/2026-11-28/specialized-internvl-4783-native-resume-skip-provider-mismatch.json":
        "39e82fcd1722a11529a9c2077d8a7f2de521dbc9fb399216209b531542d42616",
    "results/bw1000/2026-11-28/specialized-internvl-4738-cp3-to-cp25-704-native-slots-independent.json":
        "8d8a795dbd1c2d9bbb93619bae461ea494bc222f48558a0c2740193e81156974",
    "local-only/internvl-4783-swift-train-patch-source-6ba580fd4c4c4b9d927a06b20ce1bdab.json":
        "71ca9d43a5b65d642bebad5e8d0577bf1ed66f553e2d879442e89b03cc3405fd",
    "local-only/internvl-4783-swift-exact-rebuilt-loader-1a8c21fb4f734693b857095c79fe5c8c.json":
        "fdfbbe8ba92653013e33f29eb54dd763e542a9ffaa4cdfacd773b501c655670d",
    "local-only/internvl-native-resume-skip-exact-call-e0afbcbf632f461abf25544294b7b972.json":
        "bb3aabaf2885593b85dc6f5434ea134be24e009e6497f7a40c9fe0af9170bebc",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def edit(raw,changes):
    source=raw.decode()
    for old,new in changes:
        need(source.count(old)==1,"EXACT_BASE_CPU_SOURCE_ANCHOR")
        source=source.replace(old,new)
    return source.encode()


def source_evidence():
    documents=[]
    for filename,expected in EVIDENCE.items():
        raw=(REPO/filename).read_bytes()
        need(digest(raw)==expected,"EXACT_LOCAL_EVIDENCE_MEMBER")
        documents.append(json.loads(raw))
    failure,prior,train,primitives,hf=documents
    need(failure["accounting"]["1103606.4783"][2:4]==["FAILED","1:0"] and
         failure["accounting"]["1103606.4784"][2:4]==["COMPLETED","0:0"] and
         failure["recorded_new_optimizer_updates"]==0 and
         failure["native_guard"]["error_tags"]==["EXACT_ACTIVE_TRAINER_ACCELERATE_SKIP"],
         "EXACT_ORIGINAL_FAILED_GPU_WITH_POST")
    need(all(wire["authority_verified"] is True and wire["remote_received"] is True and wire["rc"]==0
             for wire in (train,primitives,hf)),"AUTHENTICATED_PASSIVE_SOURCE_READS")
    facts=primitives["response"]["files"]
    source=facts["trainers/mixin.py"]
    need(source["sha256"]==train["response"]["files"]["trainers/mixin.py"]["sha256"]==
         "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9" and
         facts["dataloader/shard.py"]["sha256"]=="ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c",
         "SAME_ACTUAL_PINNED_SWIFT_PROVIDERS")
    lines={no:text.strip() for group in source["functions"] for no,text in group["items"]}
    need(lines[1221]=="return self.get_train_dataloader(skip_batches=num_batches)" and
         "SkipBatchSampler(batch_sampler, skip_batches=skip_batches)" in lines[1316],
         "EXACT_SWIFT_REBUILT_LOADER_PRIMITIVES")
    shard={no:text.strip() for group in facts["dataloader/shard.py"]["functions"] for no,text in group["items"]}
    need(shard[91]=="self.batch_sampler.batch_sampler.set_epoch(epoch)","EXACT_NESTED_EPOCH_PROPAGATION")


def packet():
    source_evidence()
    spec=importlib.util.spec_from_file_location("original_arrow_4738",BASE/"prepare.py")
    original=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(BASE))
    try:
        spec.loader.exec_module(original)
        files=original.packet()
    finally:
        sys.path.remove(str(BASE))
    manifest=files.pop("SOURCE_FILES.json")
    need(digest(manifest)==BASE_SHA,"EXACT_ORIGINAL_4738_SOURCE_PACKET")
    files["skip_route.py"]=(HERE/"skip_route.py").read_bytes()
    files["probe.py"]=edit(files["probe.py"],[
        ("import native_route\n","import native_route\nimport skip_route\n"),
        ("continuation.EPOCH1_REQUEST,continuation.CP3_REQUEST)",
         "continuation.EPOCH1_REQUEST,continuation.CP3_REQUEST,skip_route.CPU_REQUEST,skip_route.CP3_FAILED_REQUEST)"),
        ('''            epochs,order=native_route.observe_rank(torch,mixin.DataLoaderMixin,impl,Dataset,parsed,load,
                                                    chosen_rank,frozen,reference,str(binding.EXPOSURES),clock)
            runtime.write(output/f"rank-{chosen_rank}-arrow-private.json",
                          dict(rank=chosen_rank,order=order,epochs=epochs,physical_values_verified=64))''',
         '''            epochs,order,skip=skip_route.observe(torch,mixin,impl,Dataset,parsed,load,
                                        chosen_rank,frozen,reference,str(binding.EXPOSURES),clock,binding.pin)
            runtime.write(output/f"rank-{chosen_rank}-arrow-private.json",
                          dict(rank=chosen_rank,order=order,epochs=epochs,physical_values_verified=64,native_skip=skip))'''),
        ('''        reports.append(lineage.strict_json(raw))
    plan,verified=continuation.assemble(reports,expected,reference,native["rank_logical_indices"])
    out=output/"steps4-25-plan.private.json"''',
         '''        row=lineage.strict_json(raw)
        resume=row["native_skip"]
        order=[item["logical"] for item in row["epochs"][0]["rows"]]
        lineage.need(resume==dict(constructions=[0,4],original_separate=True,new_nested_sampler=True,
                                 skipped=order[:4],retained=order[4:],skipped_dataset_fetches=0,
                                 successful_native_patch_calls=1),"EXACT_ALL_EIGHT_RESUMED_ROUTING_RECEIPTS")
        reports.append(row)
    plan,verified=continuation.assemble(reports,expected,reference,native["rank_logical_indices"])
    lineage.need(lineage.encode(plan)==skip_route.prior_plan(binding.pin),"BYTE_EXACT_ALL_704_ORIGINAL_4738_SLOTS")
    out=output/"native-swift-resume-skip-plan.private.json"'''),
        ("actual_loader_constructions_per_rank=1,actual_set_epoch_calls_per_rank=12,",
         '''actual_loader_constructions_per_rank=2,actual_set_epoch_calls_per_rank=12,
                  actual_swift_skip_patch_calls_per_rank=1,actual_native_epoch1_skipped_sampler_positions=32,
                  actual_native_epoch1_retained_fetches=32,actual_native_epoch1_skipped_dataset_fetches=0,
                  prior_4738_byte_identical=True,prior_4738_result_SHA256=skip_route.CPU_RESULT,'''),
        ("    continuation.epoch1_reference(binding.pin)\n",
         "    continuation.epoch1_reference(binding.pin)\n    skip_route.prior_plan(binding.pin)\n"),
        ("    summary.setdefault(\"verified_continuation_slots\",0)",
         '''    summary.setdefault("verified_continuation_slots",0)
    summary.update(planned_rank_count=8,planned_resume_epoch1_retained=32,planned_resume_epoch1_skipped=32,
                   original_4783_replayed=False,production_model_or_training_invoked=False)'''),
    ])
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    doc=json.loads(manifest)
    doc.update(scope="InternVL actual Swift native patched resume route, metadata-only zero GRES",
               parent_4738_source_manifest_SHA256=BASE_SHA,original_source_evidence_SHA256=EVIDENCE,
               original_4783_replayed=False,actual_gpu=False,training_authorized=False,
               files={name:dict(bytes=len(raw),SHA256=digest(raw)) for name,raw in files.items()})
    files["SOURCE_FILES.json"]=(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=packet()
    if args.output is not None:
        args.output.mkdir(mode=0o700)
        for name,raw in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(raw)
    print(json.dumps(dict(files=len(files),bytes=sum(map(len,files.values())),
                          manifest_SHA256=digest(files["SOURCE_FILES.json"]),target_launched=False),sort_keys=True))


if __name__=="__main__":
    main()

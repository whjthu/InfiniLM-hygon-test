"""Run the original Swift data path and eight native DataLoaderShards at zero GRES."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

import binding
import continuation
import lineage
import native_route


HERE=Path(__file__).resolve().parent
SCRATCH_ENV="GQA56_ARROW_PRIVATE_SCRATCH"
MASKS=("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")
SOURCE_FACTS_SHA="0b3939ed451bfd4bcc382cdc2039180b577df9705c724e82149d114fd5fec841"
PREVIOUS_REQUESTS=("b260bad52c3f473394ea2731bccc1530","8624c80413434f02a50670bb43a629a9",
                   "cc8b0d3f1cea4a88ae3fdf4d7322a3ad","0947b7dac1da45ac842fae8bb0adc25e",
                   continuation.EPOCH1_REQUEST,continuation.CP3_REQUEST)
PREVIOUS_FAILED_STEP="1103606.4685"
PREVIOUS_ARCHIVE_SHA="437cf3d709295263fd06cf531fe04b7db56c65c88436256c26a93e637db1f4cd"
PREVIOUS_RESULT_SHA="53a99af5566497680bc6257231eeed6de27c70c785370b1dba835fe583810b60"
PROCESS_INDEX_WITNESS_SHA="b22499736fa84ca61476fa5be9f2a02fd062759ffe982cb08d88446c35558743"
ROUTE_WITNESS_SHA="3101d9405c13335d4f97bc4f58da5081a78fd89cd4b4c33a37425ec11e5b09df"
CPU_ARG_WITNESS_SHA="d7ba02cb6aa028d6f39f9fad342aa709f2e0585bf3f46e4b794d95d0f00a4d6f"
EPOCH0_RESULT_SHA="0c58bb4680e1e551d29abeef07ce86fb9e8cf8ca8bb5fd9192fb7cc5f65182af"
EPOCH0_ARCHIVE_SHA="9bdf9668c966133ab0a19a4b27ed3c39923286d8a85672957dbf418bf7df5faf"
CP2_CPU_ARCHIVE_SHA="37a59fec9f72c31bae49159c7df5ac59fbf77d3f6ad76d75aeead06f06fb5691"


def deny(*args,**kwargs):
    raise PermissionError("NO_MODEL_BODY_OR_GPU_INITIALIZATION")


def no_weight_reads(event,values):
    if event=="open" and values and isinstance(values[0],(str,bytes)):
        path=os.fsdecode(values[0])
        if path.startswith(str(binding.ROOT)+"/") and re.search(r"\.(safetensors|pt|pth|bin)$",path,re.I):
            deny()


def validate_processor():
    private=binding.ORIGINAL/"BINDING.json"
    with private.open("rb") as stream:
        raw=stream.read(32769)
    lineage.need(len(raw)<=32768 and private.resolve(strict=True)==private,"BOUNDED_ORIGINAL_PROCESSOR_BINDING")
    metadata=lineage.strict_json(raw)
    actual=metadata["metadata"]
    lineage.need(metadata["source_train_SHA256"]=="0ccbe304ad1d7e686bcfcf8b81bf17ef43bab4c4fcaeef5410fc7010bf170752"
                 and metadata["staged_train_SHA256"]=="78de19307549affa877be85756e7077bbb70d8d9feea09e1a8214157ee2fa000"
                 and len(actual)==13 and set(actual)=={item.name for item in binding.PROCESSOR.iterdir()},
                 "EXACT_ORIGINAL_13_PROCESSOR_ONLY_FILES")
    for name,proof in actual.items():
        lineage.need(re.fullmatch(r"[A-Za-z0-9_.-]+",name) and not re.search(r"\.(pt|pth|bin|safetensors)$",name,re.I),
                     "NO_MODEL_BODY_IN_PROCESSOR")
        binding.pin(binding.PROCESSOR/name,32*1024*1024,proof["SHA256"],exact=False)


def scratch_root(path):
    lineage.need(path.is_absolute() and path.resolve(strict=True)==path and path.is_relative_to(Path("/tmp")) and
                 len(os.fsencode(path))<=42 and path.stat().st_uid==os.getuid() and
                 stat.S_IMODE(path.stat().st_mode)==0o700,"EXACT_SHORT_PRIVATE_NODE_SCRATCH")
    return path


def worker(output,rank,deadline,store_dir,store_pin):
    runtime=binding.from_scope(HERE/"sampler_v6.py")
    with open(os.devnull,"wb") as sink:
        os.dup2(sink.fileno(),1)
        os.dup2(sink.fileno(),2)
        try:
            path=Path(tempfile.mkdtemp(prefix=f"r{rank}-",dir=scratch_root(Path(os.environ[SCRATCH_ENV]))))
            for name in ("tmp","hf","ms"):
                (path/name).mkdir(mode=0o700)
            os.environ.update(TMPDIR=str(path/"tmp"),HF_DATASETS_CACHE=str(path/"hf"),MODELSCOPE_CACHE=str(path/"ms"))
            tempfile.tempdir=None
            sys.addaudithook(no_weight_reads)
        except Exception as error:
            runtime.mark(output,rank,"start","error",**runtime.failure(error))
            raise SystemExit(2) from None

        def draw(torch,_sampler,chosen_rank):
            import safetensors
            import safetensors.torch
            from datasets import Dataset

            torch.cuda._lazy_init=deny
            torch.load=torch.serialization.load=deny
            safetensors.safe_open=safetensors.torch.safe_open=safetensors.torch.load_file=deny
            frozen,reference,command=binding.load_members(HERE)
            validate_processor()
            cls,get_data=native_route.actual_swift(binding.pin)
            parsed,load=native_route.original_loader(command,binding.EXPOSURES,binding.PROCESSOR,path/"unused-output",
                                                      cls,get_data)
            mixin=importlib.import_module("swift.trainers.mixin")
            impl=importlib.import_module("swift.dataloader.shard")
            def clock():
                lineage.need(time.monotonic()<deadline,"COMMON_NATIVE_TWELVE_EPOCH_DEADLINE")

            epochs,order=native_route.observe_rank(torch,mixin.DataLoaderMixin,impl,Dataset,parsed,load,
                                                    chosen_rank,frozen,reference,str(binding.EXPOSURES),clock)
            runtime.write(output/f"rank-{chosen_rank}-arrow-private.json",
                          dict(rank=chosen_rank,order=order,epochs=epochs,physical_values_verified=64))
            # Frozen v6 lifecycle expects one 8-slot draw per child. Full epochs use separate evidence.
            return [row["logical"] for row in epochs[0]["rows"]]

        # Use the frozen v6 rank lifecycle and substitute only the CPU draw callback in this child.
        runtime.rank_draw=draw
        try:
            runtime.worker(output,rank,deadline,store_dir,store_pin)
        except Exception:
            raise SystemExit(2) from None


def report(runtime,output,native):
    result=dict(actual_arrow_mapping_verified=False,actual_native_metadata_shards_verified=0,
                production_workers=4,observed_loader_workers=0,metadata_collator=True,
                production_worker_behavior_verified=False,actual_template_collator_invoked=False,
                actual_forward_order_verified=False,training_ready=False,actual_optimizer_updates=0,
                planned_epochs=12,planned_all_epochs_slots=768,planned_continuation_slots=704,
                verified_continuation_slots=0)
    if native["native_logical_permutation_verified"] is not True:
        return result
    expected,_,_=binding.load_members(HERE)
    reference=continuation.epoch1_reference(binding.pin)
    reports=[]
    for rank in range(8):
        with (output/f"rank-{rank}-arrow-private.json").open("rb") as stream:
            raw=stream.read(32769)
        lineage.need(len(raw)<=32768,"BOUNDED_PRIVATE_TWELVE_EPOCH_RANK_RECEIPT")
        reports.append(lineage.strict_json(raw))
    plan,verified=continuation.assemble(reports,expected,reference,native["rank_logical_indices"])
    out=output/"steps4-25-plan.private.json"
    runtime.write(out,plan)
    result.update(actual_arrow_mapping_verified=True,actual_native_metadata_shards_verified=8,
                  actual_loader_constructions_per_rank=1,actual_set_epoch_calls_per_rank=12,
                  actual_arrow_logical_to_physical_SHA256=native_route.V4_ARROW_SHA,
                  epoch1_byte_matches_4726=True,verified_epochs=12,verified_all_epochs_slots=768,
                  verified_continuation_slots=704,verified_global_updates_covered=22,
                  start_global_step=3,start_epoch=1.5,nominal_max_steps=25,
                  continuation_plan_SHA256=lineage.digest(lineage.encode(plan)),
                  continuation_plan_bytes=out.stat().st_size,epochs=verified)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    args=parser.parse_args()
    lineage.need(re.fullmatch(r"[0-9a-f]{32}",args.request) and args.request not in (lineage.SAMPLER_REQUEST,*PREVIOUS_REQUESTS) and
                 args.output.is_relative_to(binding.ROOT) and
                 args.output.parent.resolve(strict=True)==args.output.parent,"EXACT_NEW_SHARED_ARROW_REQUEST")
    lineage.need(all(os.environ.get(key)=="-1" for key in MASKS) and
                 os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1","MASKED_OFFLINE")
    step=os.environ.get("SLURM_STEP_ID","")
    parent=os.environ.get("SLURM_JOB_ID","")
    lineage.need(parent=="1103606" and re.fullmatch(r"[0-9]+",step) and not os.environ.get("SLURM_STEP_GPUS") and
                 os.environ.get("SLURM_GPUS_ON_NODE") in (None,"","0") and
                 any({"job_"+parent,"step_"+step}<=set(line.split(":")[-1].split("/"))
                     for line in Path("/proc/self/cgroup").read_text().splitlines()),"EXACT_NUMBERED_ZERO_GRES_CONTEXT")
    need_scratch=scratch_root(args.scratch)
    lineage.need(not any(need_scratch.iterdir()),"EXACT_NEW_EMPTY_CPU_SCRATCH")
    binding.load_members(HERE)
    continuation.epoch1_reference(binding.pin)
    runtime=binding.from_scope(HERE/"sampler_v6.py")
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    os.environ.update(GLOO_SOCKET_IFNAME="lo",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",**{SCRATCH_ENV:str(need_scratch)})
    native=runtime.run(args.output,150,target=worker)
    try:
        summary=report(runtime,args.output,native)
    except (ValueError,KeyError,TypeError,OSError) as error:
        summary=dict(actual_arrow_mapping_verified=False,training_ready=False,independent_rank_error=runtime.failure(error))
    summary.update(request=args.request,actual_step=parent+"."+step,no_model=True,source_facts_SHA256=SOURCE_FACTS_SHA,
                   epoch_indices=list(native_route.EPOCHS),data_seed=native_route.DATA_SEED,
                   prior_4726_result_SHA256=continuation.EPOCH1_RESULT,
                   prior_4726_matrix_SHA256=continuation.EPOCH1_MATRIX,
                   prior_4726_archive_SHA256=continuation.EPOCH1_ARCHIVE,
                   original_CP3_native_SHA256=continuation.CP3_NATIVE,
                   original_CP3_independent_SHA256=continuation.CP3_INDEPENDENT,
                   prior_epoch0_result_SHA256=EPOCH0_RESULT_SHA,
                   prior_epoch0_archive_SHA256=EPOCH0_ARCHIVE_SHA,prior_CP2_CPU_archive_SHA256=CP2_CPU_ARCHIVE_SHA,
                   original_sampler_result_SHA256=lineage.SAMPLER_RESULT_SHA,planned_rank_occurrences=768,
                   previous_actual_failure_archive_SHA256=PREVIOUS_ARCHIVE_SHA,
                   previous_actual_failed_step=PREVIOUS_FAILED_STEP,previous_actual_result_SHA256=PREVIOUS_RESULT_SHA,
                   process_index_witness_SHA256=PROCESS_INDEX_WITNESS_SHA,
                   dataset_routing_method_SHA256=native_route.ROUTING_METHOD_SHA,dataset_routing_witness_SHA256=ROUTE_WITNESS_SHA,
                   cpu_argument_source_SHA256=native_route.CPU_ARGUMENT_SOURCE[1],
                   cpu_argument_witness_SHA256=CPU_ARG_WITNESS_SHA,
                   independent_zero_GRES_terminal_required=True,private_scratch_retained_for_owner=True,
                   native={key:value for key,value in native.items() if key!="rank_logical_indices"})
    summary.setdefault("planned_continuation_slots",704)
    summary.setdefault("verified_continuation_slots",0)
    runtime.write(args.output/"result.json",summary)
    raw=(args.output/"result.json").read_bytes()
    print(lineage.encode(dict(request=args.request,passed=summary["actual_arrow_mapping_verified"],
                              receipt_SHA256=lineage.digest(raw),receipt_bytes=len(raw))).decode(),end="")
    return 0 if summary["actual_arrow_mapping_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())

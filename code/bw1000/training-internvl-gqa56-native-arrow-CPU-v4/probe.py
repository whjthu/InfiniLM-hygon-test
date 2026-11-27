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

import binding
import lineage
import native_route


HERE=Path(__file__).resolve().parent
SCRATCH_ENV="GQA56_ARROW_PRIVATE_SCRATCH"
MASKS=("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")
SOURCE_FACTS_SHA="0b3939ed451bfd4bcc382cdc2039180b577df9705c724e82149d114fd5fec841"
PREVIOUS_REQUESTS=("b260bad52c3f473394ea2731bccc1530","8624c80413434f02a50670bb43a629a9",
                   "cc8b0d3f1cea4a88ae3fdf4d7322a3ad")
PREVIOUS_FAILED_STEP="1103606.4685"
PREVIOUS_ARCHIVE_SHA="437cf3d709295263fd06cf531fe04b7db56c65c88436256c26a93e637db1f4cd"
PREVIOUS_RESULT_SHA="53a99af5566497680bc6257231eeed6de27c70c785370b1dba835fe583810b60"
PROCESS_INDEX_WITNESS_SHA="b22499736fa84ca61476fa5be9f2a02fd062759ffe982cb08d88446c35558743"
ROUTE_WITNESS_SHA="3101d9405c13335d4f97bc4f58da5081a78fd89cd4b4c33a37425ec11e5b09df"
CPU_ARG_WITNESS_SHA="d7ba02cb6aa028d6f39f9fad342aa709f2e0585bf3f46e4b794d95d0f00a4d6f"


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
            result,order=native_route.observe_rank(torch,mixin.DataLoaderMixin,impl,Dataset,parsed,load,
                                                   chosen_rank,frozen,reference,str(binding.EXPOSURES))
            runtime.write(output/f"rank-{chosen_rank}-arrow-private.json",
                          dict(rank=chosen_rank,order=order,rows=result,physical_values_verified=64))
            return [row["logical"] for row in result]

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
                actual_forward_order_verified=False,training_ready=False,actual_optimizer_updates=0)
    if native["native_logical_permutation_verified"] is not True:
        return result
    expected,matrix,_=binding.load_members(HERE)
    reports=[]
    order=None
    for rank in range(8):
        with (output/f"rank-{rank}-arrow-private.json").open("rb") as stream:
            raw=stream.read(8193)
        lineage.need(len(raw)<=8192,"BOUNDED_PRIVATE_RANK_ARROW_RECEIPT")
        row=lineage.strict_json(raw)
        current=lineage.check_permutation(row["order"])
        lineage.need((order is None or order==current) and row["rank"]==rank and
                     row["physical_values_verified"]==64 and len(row["rows"])==8,
                     "EIGHT_NATIVE_ARROW_TABLES_AGREE")
        order=current
        want=[dict(rank=rank,update=offset//4+1,accumulation=offset%4+1,logical=logical,
                   **expected[order[logical]][1]) for offset,logical in enumerate(matrix[rank])]
        lineage.need(row["rows"]==want,"INDEPENDENT_NATIVE_RANK_LINEAGE_MATCH")
        reports.append(row["rows"])
    lineage.need(native["rank_logical_indices"]==matrix,"EXACT_ORIGINAL_4643_LOGICAL_TRACE")
    result.update(actual_arrow_mapping_verified=True,actual_native_metadata_shards_verified=8,
                  metadata_process_index_source="torch.distributed.get_rank",
                  actual_arrow_logical_to_physical_SHA256=lineage.digest(lineage.encode(order)),
                  **lineage.combine(reports,{val[1]["source_ordinal"] for val in expected.values()}))
    runtime.write(output/"physical-rank-matrix.private.json",reports)
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
                   original_sampler_result_SHA256=lineage.SAMPLER_RESULT_SHA,planned_rank_occurrences=64,
                   previous_actual_failure_archive_SHA256=PREVIOUS_ARCHIVE_SHA,
                   previous_actual_failed_step=PREVIOUS_FAILED_STEP,previous_actual_result_SHA256=PREVIOUS_RESULT_SHA,
                   process_index_witness_SHA256=PROCESS_INDEX_WITNESS_SHA,
                   dataset_routing_method_SHA256=native_route.ROUTING_METHOD_SHA,dataset_routing_witness_SHA256=ROUTE_WITNESS_SHA,
                   cpu_argument_source_SHA256=native_route.CPU_ARGUMENT_SOURCE[1],
                   cpu_argument_witness_SHA256=CPU_ARG_WITNESS_SHA,
                   independent_zero_GRES_terminal_required=True,private_scratch_retained_for_owner=True,
                   native={key:value for key,value in native.items() if key!="rank_logical_indices"})
    runtime.write(args.output/"result.json",summary)
    raw=(args.output/"result.json").read_bytes()
    print(lineage.encode(dict(request=args.request,passed=summary["actual_arrow_mapping_verified"],
                              receipt_SHA256=lineage.digest(raw),receipt_bytes=len(raw))).decode(),end="")
    return 0 if summary["actual_arrow_mapping_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())

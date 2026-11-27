"""Verify exact source and boundary logic with a tiny injected CPU tensor fixture."""
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import host
import worker

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
EXACT=REPO/"local-only/a100-vlm-original-publication-staging-v1/hygon-training-integration/generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"


def deny(action, code):
    try:
        action()
    except ValueError as error:
        assert str(error)==code,(str(error),code)
    else:
        raise AssertionError(code)


class Tensor:
    def __init__(self, value, shape=(1,2), dtype="BF16"):
        self.value=list(value)
        self.shape=shape
        self.dtype=dtype
        self.device=SimpleNamespace(type="cpu")
        self.is_meta=False
        self.requires_grad=True

    def numel(self):
        return len(self.value)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return bytes(self.value)

    def copy_(self, source, non_blocking):
        assert not non_blocking
        self.value=list(source.value)


def model_fixture():
    name="double_blocks.0.img_attn_q.lora_A.default.weight"
    dst=Tensor([0,0])
    src=Tensor([1,2])
    model=SimpleNamespace(named_parameters=lambda:((name,dst),))
    fake=SimpleNamespace(bfloat16="BF16",no_grad=nullcontext,
        isfinite=lambda _x:SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:True)),
        equal=lambda x,y:x.value==y.value)
    return fake,model,{name:src},name


def mock_query(*args, queue=False, wrong=False, memory=240000):
    if args[:3]==("scontrol","show","job"):
        return [f"JobId={host.PARENT} JobState=RUNNING UserId=test({os.getuid()}) NodeList={host.NODE}"]
    if args[:3]==("scontrol","show","hostnames"):
        return [args[3]]
    if args[:3]==("scontrol","show","step"):
        handle="1103606.4565" if wrong else host.STANDBY
        return [f"StepId={handle} Name={host.STANDBY_LABEL} State=RUNNING NodeList={host.NODE} TRES=cpu=8,gres/dcu=8,mem=16G,node=1"]
    if args[:2]==("squeue","--steps"):
        return [f"{host.STANDBY}|{host.NODE}"] if queue else []
    if args[0]=="squeue":
        return [f"{host.PARENT}|RUNNING|{host.NODE}"]
    if args[0]=="sacct":
        return [f"{host.STANDBY}|{host.STANDBY_LABEL}|RUNNING|0:0|300|{host.NODE}|cpu=8,gres/dcu=8,mem=16G,node=1|2026-11-27T12:00:00|Unknown"]
    if args[:3]==("scontrol","-d","show"):
        return [f"Nodes={host.NODE} CPU_IDs=0-63 Mem={memory}"]
    raise AssertionError(args)


def main():
    raw=EXACT.read_bytes()
    digest_fn,real_wrappers=worker.source_helpers(raw)
    assert callable(real_wrappers)
    cpu,model,state,name=model_fixture()
    expected=sha256(name.encode()+b"(1, 2)"+bytes([1,2])).hexdigest()
    result=worker.load_values(cpu,model,state,digest_fn,count=1,elements=2,expected=expected)
    assert result["copied_and_compared"]==1 and result["native_adapter_SHA256"]==expected
    deny(lambda:worker.adapter_pairs(model,{"unexpected":state[name]},"BF16",count=1,elements=2),"EXACT_ADAPTER_KEY_COVERAGE")
    deny(lambda:worker.adapter_pairs(model,{name:Tensor([1,2],shape=(2,1))},"BF16",count=1,elements=2),"EXACT_ADAPTER_SHAPE")
    deny(lambda:worker.adapter_pairs(model,{name:Tensor([1,2],dtype="FP32")},"BF16",count=1,elements=2),
        "EXACT_CPU_BF16_DESTINATION_AND_SOURCE")
    deny(lambda:worker.load_values(cpu,model,state,digest_fn,count=1,elements=2,expected="0"*64),"EXACT_ORIGINAL_ADAPTER_SHA")
    need_path=REPO/"results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-v3-4560-checkpoints-adapter-and-native-timing.json"
    evidence=json.loads(need_path.read_bytes())
    assert evidence["exported_adapter"]=={"bytes":worker.ADAPTER_BYTES,"SHA256":worker.ADAPTER_SHA}
    assert evidence["adapter_value_hash_after"]==worker.NAMED_SHA
    assert evidence["fresh_CPU_adapter_reader"]["tensors"]==216
    assert host.exact_tres("cpu=8,mem=64000M,node=1") and not host.exact_tres("cpu=8,mem=64000M,node=1,gres/dcu=1")
    with patch.object(os,"getuid",lambda:154459,create=True):
        assert host.parent_gate(mock_query)[0]==host.STANDBY
        assert host.parent_gate(lambda *a:mock_query(*a,queue=True))[0]==host.STANDBY
        deny(lambda:host.parent_gate(lambda *a:mock_query(*a,wrong=True)),"ONLY_EXACT_4564_CORESIDENT_OWNER")
        deny(lambda:host.parent_gate(lambda *a:mock_query(*a,memory=70000)),"CPU_MEMORY_AFTER_EXACT_STANDBY")
    request="63d76fd0153e420ab60397ae07d7afd1"
    here,label=host.scope(request)
    argv=host.job_argv(here,label,request,"f"*64)
    assert "--gres=none" in argv and "--overlap" not in argv and "--exclusive" not in argv and "--mem=64000M" in argv
    cmd=host.container(here,request,"f"*64)
    assert "--no-mount" in cmd and "home,cwd" in cmd
    assert "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1" in cmd[-1]
    print(json.dumps(dict(passed=True,original_backend_SHA256=worker.BACKEND_SHA,original_tensors=216,
                          negatives=6,scope="SYNTHETIC_MAPPING_AND_SCHEDULER_FIXTURES",actual_model_loads=0,actual_GPU=0),
                     sort_keys=True))


if __name__=="__main__":
    main()

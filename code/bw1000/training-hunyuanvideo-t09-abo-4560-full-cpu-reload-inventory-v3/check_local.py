"""Verify exact source and boundary logic with a tiny injected CPU tensor fixture."""
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import host
import worker

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
EXACT=REPO/"local-only/a100-vlm-original-publication-staging-v1/hygon-training-integration/generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"
FAILED_WATCH=REPO/"local-only/t09-4560-cpu-reload-original-watch-82c69b545e6145f899ce49649e9d0d32.json"
FAILED_WATCH_SHA="52703716fe340efd5784dcc62df824fdd6fbdf63519397aeca5b3959759348e6"


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


def actual_failed_accounting():
    raw=FAILED_WATCH.read_bytes()
    assert sha256(raw).hexdigest()==FAILED_WATCH_SHA
    envelope=json.loads(raw)
    assert envelope["authority_verified"] is envelope["payload_flushed"] is envelope["remote_received"] is True
    response=envelope["response"]
    actual,=response["accounting"]
    assert actual[0]=="1103606.4565" and actual[2:4]==["FAILED","1:0"]
    assert actual[6]=="cpu=8,mem=62.50G,node=1" and response["worker"]=={}
    assert response["terminal_reader_error"]["code"]=="EXACT_TERMINAL_ZERO_GRES_ACCOUNTING"
    with TemporaryDirectory() as folder:
        root=Path(folder)
        args=SimpleNamespace(request=response["original"])
        with patch.object(host,"scope",return_value=(root,actual[1])),patch.object(host,"accounting",return_value=[actual]):
            result=host.terminal(args)
            assert result["state"]=="ORIGINAL_CPU_TERMINAL" and result["qualified"] is False
            assert result["raw_accounting_AllocTRES"]==actual[6] and "controller_observation" not in result
            observed=host.controller_resources({"TRES":"node=1,mem=64000M,cpu=8"},{})
            doc=dict(request=args.request,step=actual[0],**observed)
            (root/"controller-tres-observation.json").write_bytes(json.dumps(doc).encode())
            # Separate current-schema observation fixture, not evidence from .4565.
            with patch.object(host,"read",lambda path,cap:path.read_bytes()):
                distinct=host.terminal(args)
            assert distinct["controller_observation"]==doc
            assert distinct["raw_accounting_AllocTRES"]==actual[6] and distinct["qualified"] is False
    return actual[6]


def inventory_cases():
    header=b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"p":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}'
    expected=worker.header_inventory(header,12,count=2,elements=3)
    assert expected=={"a":(2,),"p":(1,)}
    deny(lambda:worker.header_inventory(header.replace(b"F32",b"BF16",1),12,2,3),"EXACT_PUBLISHED_HEADER_F32_MEMBER")
    deny(lambda:worker.header_inventory(header.replace(b"[8,12]",b"[4,8]"),12,2,3),"EXACT_GAP_FREE_SAFETENSORS_PAYLOAD")
    deny(lambda:worker.header_inventory(b'{"x":1,"x":2}',12,2,3),"DUPLICATE_SAFETENSORS_JSON_KEY")
    deny(lambda:worker.header_inventory(header,12,2,4),"EXACT_PUBLISHED_HEADER_ELEMENTS")

    class LiveTensor(Tensor):
        def __init__(self, shape, ptr, requires_grad=True):
            super().__init__([0]*shape[0],shape=shape)
            self.ptr=ptr
            self.layout="strided"
            self.requires_grad=requires_grad

        def is_contiguous(self):
            return True

        def element_size(self):
            return 2

        def data_ptr(self):
            return self.ptr

        def storage_offset(self):
            return 0

        def stride(self):
            return (1,)

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda:self.ptr,nbytes=lambda:self.numel()*2)

    cpu=SimpleNamespace(bfloat16="BF16",strided="strided")
    parameter=LiveTensor((2,),16)
    persistent=LiveTensor((1,),64,False)
    ephemeral=LiveTensor((1,),128,False)
    ephemeral.dtype="F32"
    current_params={"a":parameter}
    current_buffers={"p":persistent,"ephemeral":ephemeral}
    keep_vars={"a":parameter,"p":persistent}

    class Model:
        def named_parameters(self,*,remove_duplicate):
            assert remove_duplicate is False
            return current_params.items()

        def named_buffers(self,*,remove_duplicate):
            assert remove_duplicate is False
            return current_buffers.items()

        def named_modules(self,*,remove_duplicate):
            assert remove_duplicate is False
            return [("",SimpleNamespace(_non_persistent_buffers_set={"ephemeral"},_buffers=current_buffers))]

        def state_dict(self,*,keep_vars: bool):
            assert keep_vars is True
            return state_dict

    model=Model()
    diag={}
    state_dict=keep_vars.copy()
    captured=worker.capture_base(cpu,model,expected,diag)
    assert diag["nonpersistent_buffers"]==diag["persistent_buffers"]==diag["parameters"]==1 and len(captured)==2
    state_dict.pop("p")
    deny(lambda:worker.capture_base(cpu,model,expected,{}),"EXACT_HEADER_STATE_REGISTRATION_BIJECTION")
    state_dict=keep_vars.copy()
    state_dict["p"]=LiveTensor((1,),160,False)
    deny(lambda:worker.capture_base(cpu,model,expected,{}),"EXACT_KEEP_VARS_OBJECT_BIJECTION")
    state_dict=keep_vars.copy()
    persistent.ptr=18
    deny(lambda:worker.capture_base(cpu,model,expected,{}),"ORIGINAL_DISTINCT_LIVE_TENSOR_STORAGE")
    persistent.ptr=64
    persistent.dtype="F32"
    deny(lambda:worker.capture_base(cpu,model,expected,{}),"ORIGINAL_MATERIALIZED_CPU_BF16_CONTIGUOUS")
    persistent.dtype="BF16"
    parameter.requires_grad=False
    current_params={"new_wrapper.a":parameter,"lora":LiveTensor((1,),128)}
    worker.check_injection(cpu,model,captured,{},adapter_count=1)
    parameter.requires_grad=True
    deny(lambda:worker.check_injection(cpu,model,captured,{},1),"EXACT_FROZEN_BASE_OBJECT_AFTER_LORA")
    parameter.requires_grad=False
    parameter.ptr=32
    deny(lambda:worker.check_injection(cpu,model,captured,{},1),"EXACT_BASE_STORAGE_AFTER_LORA")
    current_params["new_wrapper.a"]=LiveTensor((2,),16,False)
    deny(lambda:worker.check_injection(cpu,model,captured,{},1),"BASE_PARAMETER_OBJECT_COVERAGE_AFTER_LORA")


def main():
    inventory_cases()
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
    actual=actual_failed_accounting()
    assert host.exact_tres("cpu=8,mem=64000M,node=1") and host.exact_tres(actual)
    assert host.exact_tres("node=1,mem=62.50G,cpu=8")
    bad_tres=("cpu=8,mem=62.5G,node=1","cpu=8,mem=62.50g,node=1","cpu=8,mem=64000M,node=01",
              "cpu=8,mem=64000M,cpu=8","cpu=8,mem=64000M,node=1,node=1","cpu=8,mem=64000M,node=1,gres/gpu=0",
              "cpu=8,mem=62.50G,node=1,gres/dcu=1","cpu=7,mem=62.50G,node=1","cpu=8,mem=64000M"," cpu=8,mem=64000M,node=1")
    assert all(not host.exact_tres(value) for value in bad_tres)
    assert host.controller_resources({"TRES":actual},{"SLURM_STEP_GPUS":"0"})["step_GPUs_empty"] is False
    assert host.controller_resources({"TRES":actual},{"SLURM_STEP_GPUS":"(null)"})["step_GPUs_empty"] is False
    assert host.controller_resources({"AllocTRES":actual},{"SLURM_GPUS_ON_NODE":"1"})["GPUs_on_node_zero_or_unset"] is False
    deny(lambda:host.scope(host.ORIGINAL_FAILED_REQUEST),"NEW_CPU_RELOAD_REQUEST")
    deny(lambda:host.scope(host.TRES_V2_FAILED_REQUEST),"NEW_CPU_RELOAD_REQUEST")
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
                          negatives=32,actual_4565_accounting_TRES=actual,actual_4565_qualified=False,
                          scope="EXACT_FAILED_4565_AND_SYNTHETIC_BOUNDARIES",actual_model_loads=0,actual_GPU=0),
                     sort_keys=True))


if __name__=="__main__":
    main()

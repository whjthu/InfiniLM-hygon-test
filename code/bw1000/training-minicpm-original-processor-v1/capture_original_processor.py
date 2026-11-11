"""Capture original MiniCPM CPU template outputs without loading model weights."""
import datetime,hashlib,importlib.metadata,json,os,signal,time
from pathlib import Path
signal.alarm(180)
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
OUT=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-original-processor-v1")
assert not OUT.exists(),"Reconcile the original directory before any retry"
OUT.mkdir(mode=0o700,parents=True)
START=time.monotonic()
MODEL=Path("/mnt/data/nvidia-baseline-20261010/inputs/models/MiniCPM-V-4.6")
DATA=Path("/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl")
def sha(raw):return hashlib.sha256(raw).hexdigest()
def record(name,obj):
    raw=(json.dumps(obj,indent=2,ensure_ascii=False)+"\n").encode()
    with (OUT/name).open("xb") as f:f.write(raw)
    return {"bytes":len(raw),"SHA256":sha(raw)}
record("intent.json",{"claim_commit":CLAIM,"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"CPU_only":True,"load_model":False})
try:
    assert sha((MODEL/"config.json").read_bytes())=="72d83ac9089bebddf28131b9953908acaf0a5740b868263c8aaac7c40b97991c"
    raw=DATA.read_bytes()
    assert sha(raw)=="0e476394e3b0117d90cfad8f89ccfafe1800917692e86efed84eaf182c453811"
    rows=[json.loads(x) for x in raw.splitlines()]
    assert len(rows)==64
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    def forbid_device(*args,**kwargs):raise RuntimeError("GPU initialization forbidden in CPU processor qualification")
    torch.cuda._lazy_init=forbid_device
    from swift.model import get_model_processor
    from swift.template import get_template
    model,processor=get_model_processor(str(MODEL),model_type="minicpmv4_6",load_model=False,download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
    assert model is None
    template=get_template(processor,template_type="minicpmv4_6",max_length=4096,max_pixels=262144,padding_side="right",padding_free=False)
    template.set_mode("train")
    output=[];summary=[]
    def describe(value):
        if isinstance(value,torch.Tensor):
            assert value.device.type=="cpu"
            x=value.detach().contiguous()
            return dict(kind="tensor",dtype=str(x.dtype),shape=list(x.shape),SHA256=sha(x.view(torch.uint8).numpy().tobytes()))
        if isinstance(value,dict):return {str(k):describe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [describe(x) for x in value]
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise TypeError("Unexpected encoded input type "+type(value).__name__)
    for index,row in enumerate(rows):
        value=template.encode(row)
        assert isinstance(value,dict) and "input_ids" in value and "labels" in value
        assert len(value["input_ids"])==len(value["labels"])<=4096
        output.append(value)
        summary.append(dict(row_ordinal=index+1,encoded=describe(value)))
        record("progress-%03d.json"%(index+1),{"completed_rows":index+1,"planned_rows":64})
    payload=OUT/"encoded-original.pt"
    assert not payload.exists()
    torch.save({"rows":output,"model_config_SHA256":sha((MODEL/"config.json").read_bytes()),"dataset_SHA256":sha(raw)},payload)
    h=hashlib.sha256()
    with payload.open("rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    summary_identity=record("encoded-identities.json",summary)
    result=dict(state="CPU_ORIGINAL_PROCESSOR_COMPLETE",claim_commit=CLAIM,rows=64,
        torch=str(torch.__version__),Swift=importlib.metadata.version("ms-swift"),transformers=importlib.metadata.version("transformers"),
        model_loaded=False,device_initialized=torch.cuda.is_initialized(),payload=dict(path=str(payload),bytes=payload.stat().st_size,SHA256=h.hexdigest()),
        encoded_identities=summary_identity,wall_seconds=time.monotonic()-START,
        scope="Native CPU template encoding in original row order, before any model post-encode hook. Not the distributed sampler's first batch, zero-update forward, backward or training numerical acceptance.")
    record("receipt.json",result);print(json.dumps(result))
except BaseException as e:
    record("failure.json",{"state":"CPU_PROCESSOR_FAILED","error_type":type(e).__name__,"error_message":str(e)[:300],"wall_seconds":time.monotonic()-START,"claim_commit":CLAIM})
    raise

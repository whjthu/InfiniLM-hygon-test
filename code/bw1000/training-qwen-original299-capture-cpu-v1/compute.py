"""Independently consume the original eight-rank native reference without GPU work."""
import datetime, hashlib, json, os, signal, sys, time
from pathlib import Path
import torch
from reference_schema import expected_files,validate_capture
ROOT=Path('/original');AUDIT=Path('/evidence');sys.path.insert(0,'/framework')
epoch=float(os.environ['REFERENCE_READER_EPOCH']);deadline=float(os.environ['REFERENCE_READER_DEADLINE']);remaining=deadline-time.time();assert deadline==epoch+870 and 0<remaining<=870
signal.setitimer(signal.ITIMER_REAL,remaining)
from vlm.runtime.first_update_audit import fingerprint
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized()
def forbidden_device_initialization(*args,**kwargs):raise RuntimeError("GPU initialization forbidden in independent reader")
torch.cuda._lazy_init=forbidden_device_initialization
torch.set_num_threads(2)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for raw in iter(lambda:f.read(1024*1024),b""):h.update(raw)
    return h.hexdigest()
def load(p):return torch.load(p,map_location="cpu",weights_only=True)
def finite(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=="cpu"
        if value.is_floating_point():assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,(list,tuple)):
        for x in value:finite(x)
def named_digest(values):
    h=hashlib.sha256()
    for name,p in sorted(values.items()):
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def equal_named(a,b):
    return a.keys()==b.keys() and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)
def norm(values):
    return sum(torch.sum(x.double()*x.double()).item() for x in values.values() if isinstance(x,torch.Tensor))**.5
def main():
    started=time.monotonic()
    raw=(ROOT/"outcome.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()=="c149981c7a5e203323d54e248e15238031eaba85d45492d280b44db84b4c122f"
    assert hashlib.sha256(Path("/stage/QWEN_TARGET_NATIVE_BINDING.json").read_bytes()).hexdigest()=="9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
    outcome=json.loads(raw)
    assert outcome["keeper_passed"] is False and outcome["after"]["owned_processes_remaining"]==[]
    assert outcome["after"]["kfd_holder_count"]==0 and outcome["after"]["master_port_available"]
    native_outcome=outcome["native-outcome.json"]["value"]
    assert native_outcome["native_returncode"]!=0 and not native_outcome["normal_native_exit"]
    assert sorted(x.name for x in (ROOT/"capture").iterdir())==["rank-%02d"%i for i in range(8)]
    rows=[];initial_digests=[];updated_digests=[];all_artifacts=0
    for rank in range(8):
        folder=ROOT/"capture"/("rank-%02d"%rank)
        receipt=json.loads((folder/"receipt.json").read_bytes());validate_capture(receipt,rank)
        assert json.loads((folder/"first-two-scope-complete.json").read_bytes())=={"rank":rank,"global_step":2}
        assert not (folder/"native25-complete.json").exists()
        reference=json.loads(Path("/fix/A100_SUBSET.json").read_bytes())["ranks"][rank]
        assert reference["rank"]==rank
        owner=json.loads((folder/"owner.json").read_bytes())
        assert owner["rank"]==rank and owner["epoch"]==outcome["epoch"] and owner["deadline"]==native_outcome["native_deadline"]
        frozen=json.loads((folder/"frozen-boundaries.json").read_bytes())
        assert frozen=={"model.visual":{"tensors":351,"parameters":576388336}}
        lifecycle_raw=(folder/'checkpointing-lifecycle.json').read_bytes();lifecycle=json.loads(lifecycle_raw)
        bound=json.loads(Path('/stage/QWEN_TARGET_NATIVE_BINDING.json').read_bytes())
        assert lifecycle==dict(original_requested=True,native_Trainer_flag=False,language_checkpointing=True,
            language_checkpointed_layers=36,language_use_cache=False,vision_checkpointing=False,vision_checkpointing_flags=28,
            provider=bound['native_checkpointing_lifecycle']['provider'])
        provider_raw=(folder/'runtime-providers.json').read_bytes();providers=json.loads(provider_raw)
        assert providers['HIP']=='6.3.26113' and providers['CUDA'] is None
        assert len(providers['files'])>=1
        for name,pin in providers['files'].items():
            path=Path(name);assert path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256']==bound['model_source_SHA256']
        manifest={x["name"]:x for x in receipt["files"]}
        assert set(p.name for p in folder.glob("*.pt"))==expected_files()
        for name,item in manifest.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.stat().st_size==item["bytes"] and sha(path)==item["SHA256"]
        initial=load(folder/"initial.pt");finite(initial)
        assert initial["global_step"]==0 and sum(x.numel() for x in initial["parameters"].values())==43646976
        assert set(initial["parameters"])=={x["name"] for x in initial["manifest"]}
        initial_hash=named_digest(initial["parameters"]);initial_digests.append(initial_hash)
        assert initial_hash==reference["initial_named_parameter_SHA256"]
        assert initial_hash==owner['initial_trainables_SHA256']
        binding=json.loads(Path('/stage/QWEN_TARGET_NATIVE_BINDING.json').read_bytes());assert initial['manifest']==binding['trainable_manifest'] and owner['binding_SHA256']=='9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067'
        micro=[]
        for i in range(1,9):
            inp=load(folder/("forward-%02d-input.pt"%i));out=load(folder/("forward-%02d-output.pt"%i))
            finite(inp);finite(out)
            step=(i-1)//4
            input_fingerprint=fingerprint({"args":inp["args"],"kwargs":inp["kwargs"]})["SHA256"]
            assert input_fingerprint==binding["input_fingerprints_by_rank"][str(rank)][i-1]==reference["micro_forwards"][i-1]["input_content_SHA256"]
            assert inp["global_step"]==out["global_step"]==step and inp["micro_ordinal"]==out["micro_ordinal"]==i
            kwargs=inp["kwargs"];result=out["result"]
            labels=kwargs.get("labels")
            assert isinstance(labels,torch.Tensor) and labels.ndim==2 and labels.shape[0]==1 and 0<labels.shape[1]<=4096
            assert (labels!=-100).any()
            assert isinstance(result["loss"],torch.Tensor) and result["loss"].numel()==1
            logits=result.get('logits');logit_shape=None
            if logits is not None:
                assert isinstance(logits,torch.Tensor) and logits.ndim==3 and logits.shape[0]==1
                keep=kwargs.get('logits_to_keep',0)
                if isinstance(keep,int):expected_length=labels.shape[1] if keep==0 else min(keep,labels.shape[1])
                elif isinstance(keep,torch.Tensor):expected_length=int(keep.sum()) if keep.dtype==torch.bool else keep.numel()
                else:raise ValueError('unexpected native logit selection identity')
                assert logits.shape[1]==expected_length
                logit_shape=list(logits.shape)
            micro.append({"ordinal":i,"global_step":step,"label_tokens":int((labels!=-100).sum()),"label_sequence_length":labels.shape[1],
                          "input_sequence_length":int(kwargs['input_ids'].shape[-1]) if isinstance(kwargs.get('input_ids'),torch.Tensor) else int(kwargs['inputs_embeds'].shape[-2]) if isinstance(kwargs.get('inputs_embeds'),torch.Tensor) else None,
                          "loss":result["loss"].float().item(),"input_SHA256":manifest["forward-%02d-input.pt"%i]["SHA256"],
                          "output_SHA256":manifest["forward-%02d-output.pt"%i]["SHA256"],
                          "logit_shape":logit_shape,"input_content":fingerprint({'args':inp['args'],'kwargs':kwargs}),
                          "output_content":fingerprint(result),"RNG_content":fingerprint(inp['RNG_before_forward'])})
            del inp,out
        steps=[];previous=initial["parameters"]
        for update in (1,2):
            pre=load(folder/("update-%02d-preclip.pt"%update));post=load(folder/("update-%02d-postclip.pt"%update))
            before=load(folder/("update-%02d-before-optimizer.pt"%update));after=load(folder/("update-%02d-after-optimizer.pt"%update))
            for x in (pre,post,before,after):finite(x)
            assert pre["global_step"]==post["global_step"]==before["global_step"]==after["global_step_before_increment"]==update-1
            assert equal_named(previous,before["parameters"])
            assert set(pre["gradients"])==set(post["gradients"])==set(initial["parameters"])
            assert all((pre["gradients"][n] is None)==(post["gradients"][n] is None) for n in pre["gradients"])
            assert norm(pre["gradients"])>0
            assert norm(post['gradients'])<=norm(pre['gradients'])*(1+1e-5) and norm(post['gradients'])<=1.0002
            states=after["optimizer"]["state_by_name"]
            for name,gradient in pre["gradients"].items():
                if gradient is not None:
                    assert states[name] and states[name]["step"].item()==update
                    assert states[name]["exp_avg"].shape==initial["parameters"][name].shape
                    assert states[name]["exp_avg_sq"].shape==initial["parameters"][name].shape
            rates=[g["lr"] for g in before["optimizer"]["param_groups"]]
            differences={n:after["parameters"][n].double()-before["parameters"][n].double() for n in before["parameters"]}
            changed=sum(not torch.equal(before["parameters"][n],after["parameters"][n]) for n in before["parameters"])
            if update==1:assert max(rates)==0 and changed==0 and all(not x for x in before["optimizer"]["state_by_name"].values())
            if update==2:assert max(rates)>0 and changed>0
            steps.append({"update":update,"learning_rates":rates,"changed_parameter_tensors":changed,
                          "parameter_delta_l2":norm(differences),"preclip_gradient_l2":norm(pre["gradients"]),
                          "postclip_gradient_l2":norm(post["gradients"]),"reported_preclip_norm":float(post["reported_preclip_norm"]),
                          "optimizer_after_SHA256":manifest["update-%02d-after-optimizer.pt"%update]["SHA256"]})
            previous=after["parameters"]
            if update==2:updated_digests.append(named_digest(previous))
        rows.append({"rank":rank,"initial_named_parameter_SHA256":initial_hash,"updates":steps,"micro_forwards":micro,
                     "native_global_step":2,"parameter_after_second_update_SHA256":updated_digests[-1],"frozen_boundaries":frozen,
                     "artifact_count":len(manifest),"checkpointing_lifecycle":lifecycle,"checkpointing_lifecycle_SHA256":hashlib.sha256(lifecycle_raw).hexdigest(),"runtime_providers_SHA256":hashlib.sha256(provider_raw).hexdigest(),"runtime_provider_classes":len(providers["classes"])})
        all_artifacts+=len(manifest)
        partial=dict(planned_ranks=8,planned_artifacts=200,validated_ranks=len(rows),validated_artifacts=all_artifacts,ranks=rows)
        temp=AUDIT/'audit-progress.tmp';temp.write_text(json.dumps(partial));temp.replace(AUDIT/'audit-progress.json')
        del initial,previous,pre,post,before,after
    assert len(set(initial_digests))==1 and all_artifacts==200
    result={"state":"INDEPENDENT_HIP_CAPTURE_COMPLETE","claim_commit":outcome["claim_commit"],
            "outcome_SHA256":hashlib.sha256(raw).hexdigest(),"ranks":rows,"tensor_artifacts":all_artifacts,
            "original_gpu_step":"1103606.299","original_job_failed":True,"reference_updates":2,"micro_forwards":64,
            "initial_across_ranks_exact":True,"positive_update_across_ranks_exact":len(set(updated_digests))==1,
            "GPU_execution_by_reader":False,"cross_platform_numerics_qualified":False,"formal_performance":False}
    import compare
    result["A100_comparison"]=compare.scalar_comparison(rows,json.loads(Path("/fix/A100_SUBSET.json").read_bytes()))
    result["supplemental_policy_receipts"]=0
    result["epoch"]=epoch;result["deadline"]=deadline
    result["completed_UTC"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["wall_seconds"]=time.monotonic()-started
    result.update(passed=True,step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],GPU_cards=0)
    with (AUDIT/"compute.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"state":result["state"],"ranks":8,"artifacts":200,"reference_updates":2,
                      "micro_forwards":64,"A100_comparison":result["A100_comparison"]},allow_nan=False))


if __name__=="__main__":
    main()

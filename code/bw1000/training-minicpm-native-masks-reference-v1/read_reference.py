"""Independently consume the original eight-rank native reference without GPU work."""
import hashlib,json,os,signal,time
from pathlib import Path
import torch
from reference_schema import expected_files,validate_capture,validate_native25
ROOT=Path(__file__).resolve().parent.parent
signal.alarm(600)
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized()
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
    raw=(ROOT/"outcome.json").read_bytes();outcome=json.loads(raw)
    assert outcome["GPU_launched"] and outcome["native_returncode"]==0 and outcome["keeper_passed"]
    assert outcome["deadline"]-outcome["epoch"]==900 and outcome["native_deadline"]-outcome["epoch"]==820
    assert outcome["supervisor_stop"]-outcome["epoch"]==850
    assert outcome["cleanup"]["termination_needed"] is False and not outcome["post"]["compute_apps"]
    assert (ROOT/"capture").is_dir()
    assert sorted(x.name for x in (ROOT/"capture").iterdir())==["rank-%02d"%i for i in range(8)]
    rows=[];initial_digests=[];updated_digests=[];all_artifacts=0
    for rank in range(8):
        folder=ROOT/"capture"/("rank-%02d"%rank)
        receipt=json.loads((folder/"receipt.json").read_bytes());validate_capture(receipt,rank)
        assert json.loads((folder/"native25-complete.json").read_bytes())=={"rank":rank,"global_step":25}
        native=json.loads((ROOT/"train/nvidia-baseline"/("rank-%02d.json"%rank)).read_bytes())
        validate_native25(native,rank)
        owner=json.loads((folder/"owner.json").read_bytes())
        assert owner["rank"]==rank and owner["epoch"]==outcome["epoch"] and owner["deadline"]==outcome["native_deadline"]
        frozen=json.loads((folder/"frozen-boundaries.json").read_bytes())
        assert set(frozen)=={"model.vision_tower","model.merger"} and all(v["parameters"]>0 for v in frozen.values())
        manifest={x["name"]:x for x in receipt["files"]}
        assert set(p.name for p in folder.glob("*.pt"))==expected_files()
        for name,item in manifest.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.stat().st_size==item["bytes"] and sha(path)==item["SHA256"]
        initial=load(folder/"initial.pt");finite(initial)
        assert initial["global_step"]==0 and sum(x.numel() for x in initial["parameters"].values())==10822656
        assert set(initial["parameters"])=={x["name"] for x in initial["manifest"]}
        initial_hash=named_digest(initial["parameters"]);initial_digests.append(initial_hash)
        assert initial_hash==native["initial_trainable_sha256"]
        assert initial_hash=="9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16","Original initialization differs; retain evidence before any cross-platform promotion"
        micro=[]
        for i in range(1,9):
            inp=load(folder/("forward-%02d-input.pt"%i));out=load(folder/("forward-%02d-output.pt"%i))
            finite(inp);finite(out)
            step=(i-1)//4
            assert inp["global_step"]==out["global_step"]==step and inp["micro_ordinal"]==out["micro_ordinal"]==i
            kwargs=inp["kwargs"];result=out["result"]
            labels=kwargs.get("labels")
            assert isinstance(labels,torch.Tensor) and labels.ndim==2 and labels.shape[0]==1 and 0<labels.shape[1]<=4096
            assert (labels!=-100).any()
            assert isinstance(result["loss"],torch.Tensor) and result["loss"].numel()==1
            assert result["logits"].ndim==3 and tuple(result["logits"].shape[:2])==tuple(labels.shape)
            micro.append({"ordinal":i,"global_step":step,"label_tokens":int((labels!=-100).sum()),"sequence_length":labels.shape[1],
                          "loss":result["loss"].float().item(),"input_SHA256":manifest["forward-%02d-input.pt"%i]["SHA256"],
                          "output_SHA256":manifest["forward-%02d-output.pt"%i]["SHA256"]})
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
                     "native_global_step":25,"native_final_parameter_SHA256":native["final_trainable_sha256"],"frozen_boundaries":frozen,
                     "artifact_count":len(manifest)})
        all_artifacts+=len(manifest)
        del initial,previous,pre,post,before,after
    assert len(set(initial_digests))==1 and len(set(updated_digests))==1
    result={"state":"INDEPENDENT_NATIVE_REFERENCE_CAPTURE_COMPLETE","claim_commit":outcome["claim_commit"],
            "outcome_SHA256":hashlib.sha256(raw).hexdigest(),"ranks":rows,"tensor_artifacts":all_artifacts,
            "original25_completed_ranks":8,"reference_updates":2,"micro_forwards":64,
            "initial_across_ranks_exact":True,"positive_update_across_ranks_exact":True,
            "GPU_execution_by_reader":False,"cross_platform_numerics_qualified":False,"formal_performance":False}
    with (ROOT/"independent-reference.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({"state":result["state"],"ranks":8,"artifacts":all_artifacts,"reference_updates":2,"micro_forwards":64}))
if __name__=="__main__":main()

"""Independently consume the original eight-rank native reference without GPU work."""
import hashlib,json,os,signal,time,sys,datetime,struct,socket,subprocess
from pathlib import Path
import torch
from reference_schema import expected_files,validate_capture,validate_native25
ROOT=Path('/original');AUDIT=Path('/evidence');sys.path.insert(0,'/framework')
epoch=float(os.environ['REFERENCE_READER_EPOCH']);deadline=float(os.environ['REFERENCE_READER_DEADLINE']);remaining=deadline-time.time();assert deadline==epoch+870 and 0<remaining<=870
signal.setitimer(signal.ITIMER_REAL,remaining)
from vlm.runtime.first_update_audit import fingerprint
from transformers.trainer import safe_globals
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
    raw=(ROOT/"outcome.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()=="3b3d97688c6d71935fa4519f97dfb7e09a3bad5efe4871f29023ee3eff13fb58"
    assert hashlib.sha256(Path("/stage/QWEN_TARGET_NATIVE_BINDING.json").read_bytes()).hexdigest()=="9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
    outcome=json.loads(raw)
    assert outcome["keeper_passed"] and outcome["after"]["owned_processes_remaining"]==[]
    assert outcome["after"]["kfd_holder_count"]==0 and outcome["after"]["master_port_available"]
    native=outcome["native-outcome.json"]["value"]
    assert native["native_returncode"]==0
    assert sorted(x.name for x in (ROOT/"capture").iterdir())==["rank-%02d"%i for i in range(8)]
    rows=[];initial_digests=[];updated_digests=[];all_artifacts=0
    for rank in range(8):
        folder=ROOT/"capture"/("rank-%02d"%rank)
        receipt=json.loads((folder/"receipt.json").read_bytes());validate_capture(receipt,rank)
        assert json.loads((folder/"native25-complete.json").read_bytes())=={"rank":rank,"global_step":25}
        native=json.loads((ROOT/"train/nvidia-baseline"/("rank-%02d.json"%rank)).read_bytes())
        validate_native25(native,rank)
        owner=json.loads((folder/"owner.json").read_bytes())
        assert owner["rank"]==rank and owner["epoch"]==outcome["epoch"] and owner["deadline"]==native["native_deadline"]
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
        assert initial_hash==native["initial_trainable_sha256"]
        assert initial_hash==owner['initial_trainables_SHA256']
        binding=json.loads(Path('/stage/QWEN_TARGET_NATIVE_BINDING.json').read_bytes());assert initial['manifest']==binding['trainable_manifest'] and owner['binding_SHA256']=='9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067'
        micro=[]
        for i in range(1,9):
            inp=load(folder/("forward-%02d-input.pt"%i));out=load(folder/("forward-%02d-output.pt"%i))
            finite(inp);finite(out)
            step=(i-1)//4
            assert fingerprint({"args":inp["args"],"kwargs":inp["kwargs"]})["SHA256"]==binding["input_fingerprints_by_rank"][str(rank)][i-1]
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
                     "native_global_step":25,"native_final_parameter_SHA256":native["final_trainable_sha256"],"frozen_boundaries":frozen,
                     "artifact_count":len(manifest),"checkpointing_lifecycle":lifecycle,"checkpointing_lifecycle_SHA256":hashlib.sha256(lifecycle_raw).hexdigest(),"runtime_providers_SHA256":hashlib.sha256(provider_raw).hexdigest(),"runtime_provider_classes":len(providers["classes"])})
        all_artifacts+=len(manifest)
        partial=dict(planned_ranks=8,planned_artifacts=200,validated_ranks=len(rows),validated_artifacts=all_artifacts,ranks=rows)
        temp=AUDIT/'audit-progress.tmp';temp.write_text(json.dumps(partial));temp.replace(AUDIT/'audit-progress.json')
        del initial,previous,pre,post,before,after
    assert len(set(initial_digests))==1 and len(set(updated_digests))==1 and all_artifacts==200
    result={"state":"INDEPENDENT_HIP_CAPTURE_COMPLETE","claim_commit":outcome["claim_commit"],
            "outcome_SHA256":hashlib.sha256(raw).hexdigest(),"ranks":rows,"tensor_artifacts":all_artifacts,
            "original25_completed_ranks":8,"reference_updates":2,"micro_forwards":64,
            "initial_across_ranks_exact":True,"positive_update_across_ranks_exact":True,
            "GPU_execution_by_reader":False,"cross_platform_numerics_qualified":False,"formal_performance":False}
    result['checkpoints']=audit_checkpoints(outcome)
    result['all_native_final_parameters_across_ranks_exact']=len({r['native_final_parameter_SHA256'] for r in rows})==1
    assert result['all_native_final_parameters_across_ranks_exact']
    result['epoch']=epoch;result['deadline']=deadline;result['completed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['wall_seconds']=time.monotonic()-AUDIT_START
    result.update(passed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_cards=0)
    with (AUDIT/'compute.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({"state":result["state"],"ranks":8,"artifacts":all_artifacts,"reference_updates":2,"micro_forwards":64}))



def audit_checkpoints(outcome):
    checkpoints=[]
    for step in (10,20,25):
        folder=ROOT/'train'/('checkpoint-%d'%step)
        names=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json']+['rng_state_%d.pth'%i for i in range(8)]
        argument_files=[n for n in ('args.json','training_args.bin') if (folder/n).is_file()]
        assert argument_files;names+=argument_files;files={}
        for name in names:
            p=folder/name;assert p.is_file() and not p.is_symlink() and 0<p.stat().st_size<2*1024**3
            files[name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
        trainer=json.loads((folder/'trainer_state.json').read_bytes());assert trainer['global_step']==step and trainer['max_steps']==25
        optimizer=load(folder/'optimizer.pt');states=optimizer['state'];assert len(states)==504
        steps=[float(v['step']) for v in states.values()];assert min(steps)==max(steps)==step
        assert sum(v['exp_avg'].numel() for v in states.values())==43646976
        for state in states.values():
            assert set(state)=={'step','exp_avg','exp_avg_sq'} and state['exp_avg'].shape==state['exp_avg_sq'].shape
            finite(state)
        del optimizer,states
        scheduler=load(folder/'scheduler.pt');assert scheduler['last_epoch']==step
        ranks=[]
        for rank in range(8):
            with safe_globals():state=load(folder/('rng_state_%d.pth'%rank))
            assert set(state)=={'python','numpy','cpu','cuda'} and state['cpu'].dtype==torch.uint8
            device=state['cuda'];device=[device] if isinstance(device,torch.Tensor) else device
            assert device and all(x.device.type=='cpu' and x.dtype==torch.uint8 for x in device)
            ranks.append(dict(rank=rank,CPU_rng_bytes=state['cpu'].numel(),device_rng_bytes=[x.numel() for x in device],python_state_type=type(state['python']).__name__,numpy_state_type=type(state['numpy']).__name__))
        with (folder/'adapter_model.safetensors').open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];assert 0<length<=2*1024**2;header=json.loads(f.read(length))
        tensors={k:v for k,v in header.items() if k!='__metadata__'};assert len(tensors)==504
        assert sum(__import__('math').prod(v['shape']) for v in tensors.values())==43646976
        checkpoints.append(dict(global_step=step,required_state_files=files,optimizer_state_count=504,optimizer_step_min=min(steps),optimizer_step_max=max(steps),scheduler_last_epoch=scheduler['last_epoch'],rank_RNG=ranks,adapter_tensors=504,adapter_parameters=43646976,fresh_process_restore_qualified=False))
    return checkpoints


if __name__=='__main__':
    AUDIT_START=time.monotonic()
    try:
        main()
    except BaseException as error:
        p=AUDIT/'audit-progress.json';progress=json.loads(p.read_bytes()) if p.is_file() else dict(planned_ranks=8,planned_artifacts=200,validated_ranks=0,validated_artifacts=0)
        with (AUDIT/'failure.json').open('x') as f:json.dump(dict(error_type=type(error).__name__,planned_ranks=8,planned_artifacts=200,validated_ranks=progress['validated_ranks'],validated_artifacts=progress['validated_artifacts'],epoch=epoch,deadline=deadline),f,indent=2)
        raise

"""Native COCO20 continuous updates and fresh-process restart, using the deployed checkpoint API."""

from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import time

from coco20_inputs import load_dataset, sample
from coco20_step import exact, pinned_framework
from observe_update import Witness


PACKAGE = Path("/fusion-run")
FRAMEWORK = Path("/common-framework/generation/src/gencuda")


def need(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def stable(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data,out,indent=2,sort_keys=True,allow_nan=False)
        out.flush()
        os.fsync(out.fileno())


def checkpoint_sha(path):
    with path.open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    digest=sha256()
    for block in iter(lambda: stream.read(4<<20),b""):
        digest.update(block)
    return digest.hexdigest()


def framework_api(claim):
    for name,pin in claim["framework_checkpoint_modules"].items():
        exact(FRAMEWORK/name,pin)
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_training_resume_checkpoint as resume

    for module in (common,resume):
        path=Path(module.__file__).resolve()
        need(path==FRAMEWORK/path.name and path.name in claim["framework_checkpoint_modules"],
             "Checkpoint provider escaped the deployed framework")
    return common,resume


def schedule(mode):
    if mode=="continuous":
        return range(1,26)
    if mode=="resume20":
        return range(21,26)
    raise ValueError("Unrecognized training process")


def checkpoint_path(output, step):
    return output / f"checkpoint-{step:02d}.private.pt"


def run(model, native, torch, output, mode, claim, deadline, attention):
    from diffsynth.diffusion import loss

    common,resume=framework_api(claim)
    captures,backend=pinned_framework(claim)
    update=captures.instrument_update(backend._train_step)
    exact(Path(loss.__file__).resolve(),claim["loss_provider"])
    need(native.FlowMatchSFTLoss is loss.FlowMatchSFTLoss,"Original native loss identity differs")
    dataset,rows,original_manifest_present=load_dataset(PACKAGE,claim,deadline)
    first_result=json.loads((PACKAGE/"FIRST_UPDATE_EVIDENCE.json").read_bytes())["same_HIP_update"]
    first_norm=json.loads((PACKAGE/"TYPED_FIRST_UPDATE_EVIDENCE.json").read_bytes())["HIP"]["clip_norm_saved"]
    model.train()
    scheduler=model.pipe.scheduler
    scheduler.set_timesteps(1000,training=True)
    scheduler_contract=resume.scheduler_spec(scheduler)
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    frozen={name:p for name,p in model.named_parameters() if not p.requires_grad}
    versions={name:p._version for name,p in frozen.items()}
    need(len(named)==476 and sum(p.numel() for p in named.values())==9922560,"Native parameter boundary differs")
    optimizer=torch.optim.AdamW(list(model.trainable_modules()),lr=1e-4,weight_decay=.01)
    need({id(p) for group in optimizer.param_groups for p in group["params"]}=={id(p) for p in named.values()},
         "Native optimizer parameter set differs")
    progress=common.initial_optimizer_progress(model,optimizer)
    identity=dict(config=stable(claim),source=checkpoint_sha(PACKAGE/"SOURCE_FILES.json"),
                  data=claim["input_contract"]["SHA256"],model=claim["full_runtime"]["SHA256"],
                  runtime=stable(dict(torch=str(torch.__version__),hip=torch.version.hip,
                                      compatibility=claim["isolated_runtime_claim"]["SHA256"],
                                      providers=claim["framework_checkpoint_modules"],
                                      attention_policy=claim["attention_policy"])),
                  scheduler=common.tree_sha256(scheduler_contract))
    sample_order=[claim["input_contract"]["SHA256"]+":"+str(r["index"]) for r in rows]
    recipe=resume.build_contract(identity=identity,seed=claim["base_seed"],max_steps=25,
                                  sample_order=sample_order,device_ids=[0])
    write(output/"contract.json",recipe)
    restored=None
    if mode=="resume20":
        continuous=PACKAGE/"continuous"
        checkpoints=json.loads((continuous/"checkpoint-receipts.json").read_bytes())
        original=checkpoint_path(continuous,20)
        saved=next(v for v in checkpoints if v["global_step"]==20)
        need(saved["checkpoint_SHA256"]==checkpoint_sha(original),"Original step 20 checkpoint bytes changed")
        payload=resume.load_checkpoint(original)
        progress,restored=resume.restore_checkpoint(payload,recipe,model,optimizer,scheduler,scheduler_contract)
        resume.assert_restored_boundary(payload,model,optimizer,progress,scheduler,[0])
        need(restored["metadata"]["global_step"]==20 and restored["next"]["sample_cursor"]==0
             and restored["next"]["global_step"]==21,"Restored first use is not step 21 at cursor 0")
        write(output/"restore.json",{"restored_boundary_exact":True,"metadata":restored["metadata"],
                                     "next":restored["next"],"checkpoint_SHA256":saved["checkpoint_SHA256"]})
        del payload

    completed=[]
    receipts=[]
    begun=time.monotonic()
    with (output/"optimizer-journal.jsonl").open("x",encoding="utf-8") as journal, (output/"steps.jsonl").open("x",encoding="utf-8") as metrics:
        def append(stream, value):
            stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())

        for step in schedule(mode):
            deadline()
            ordinal=(step-1)%20
            data,input_identity=sample(dataset,rows,ordinal)
            witness=Witness(torch,named.values(),deadline,
                            lambda event: append(journal,dict(global_step=step,event=event)),
                            first_result if step==1 else None)
            previous_forced_calls = attention["forced_calls"]
            with captures.observe_native_loss(native,witness):
                value=update(model,optimizer,data,recipe["seed"]+step,1.,observer=witness)
            append(journal,dict(global_step=step,event="optimizer_returned"))
            witness.bind_return(value)
            forced_calls = attention["forced_calls"] - previous_forced_calls
            need(forced_calls > 0, "Native update did not use deterministic FlashAttention backward")
            progress=common.advance_optimizer_progress(progress,model,optimizer)
            need(all(p.grad is None and p._version==versions[name] for name,p in frozen.items()),
                 "A frozen parameter received a gradient or changed version")
            need(all(bool(torch.isfinite(p).all()) for p in named.values()),"Updated LoRA parameter is not finite")
            completed.append(step)
            append(metrics,dict(global_step=step,ordinal=ordinal,loss=value["loss"],preclip_norm=value["grad_norm"],
                                input_identity=input_identity,operands=witness.operands,
                                flash2_deterministic_calls=forced_calls,
                                default_first_step_preclip_norm=first_norm if step==1 else None))
            if step in claim["checkpoint_steps"]:
                deadline()
                saved=resume.make_payload(contract=recipe,global_step=step,model=model,optimizer=optimizer,
                    optimizer_progress=progress,scheduler=scheduler,expected_scheduler=scheduler_contract)
                path=checkpoint_path(output,step)
                resume.save_atomic(path,saved)
                receipts.append(dict(global_step=step,bytes=path.stat().st_size,checkpoint_SHA256=checkpoint_sha(path),
                    adapter_SHA256=saved["adapter_sha256"],state_SHA256=common.tree_sha256(saved["state"]),
                    optimizer_SHA256=common.tree_sha256(saved["state"]["optimizer"]),
                    diffusion_scheduler_SHA256=common.tree_sha256(saved["scheduler"]),
                    rng_SHA256=common.tree_sha256(saved["rng"]),integrity_SHA256=saved["integrity_sha256"],
                    next_data_cursor=saved["metadata"]["next_data_cursor"]))
                del saved
    need(completed==list(schedule(mode)) and [r["global_step"] for r in receipts]==([10,20,25] if mode=="continuous" else [25]),
         "Training and checkpoint denominators differ")
    write(output/"checkpoint-receipts.json",receipts)
    return dict(mode=mode,completed_global_steps=completed,optimizer_updates=len(completed),
                restored_exact=restored is not None,planned_steps=list(schedule(mode)),
                checkpoints=receipts,worker_training_section_seconds=time.monotonic()-begun,
                frozen_parameter_check="parameter version and no gradient",original_manifest_present=original_manifest_present,
                formal_A100=False,formal_performance=False)

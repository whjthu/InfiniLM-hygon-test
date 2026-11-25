"""Native Style18 continuous updates and fresh-process restart, using the deployed checkpoint API."""

from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import time

from style18_inputs import exact, forward, load_dataset, sample
from resume10_policy import prepare_checkpoint, confirm_checkpoint


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
    if mode=="resume10":
        return range(11,26)
    raise ValueError("Unrecognized training process")


def checkpoint_path(output, step):
    return output / f"checkpoint-{step:02d}.private.pt"


def run(model, torch, output, mode, claim, deadline):
    from diffsynth.diffusion import loss

    common,resume=framework_api(claim)
    dataset,rows=load_dataset(model,PACKAGE,deadline)
    first_result=json.loads((PACKAGE/"FIRST_UPDATE_EVIDENCE.json").read_bytes())["result"]["domain_update"]
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
                  data=claim["style18_inputs"]["SHA256"],model=claim["full_runtime"]["SHA256"],
                  runtime=stable(dict(torch=str(torch.__version__),hip=torch.version.hip,
                                      compatibility=claim["isolated_runtime_claim"]["SHA256"],
                                      providers=claim["framework_checkpoint_modules"])),
                  scheduler=common.tree_sha256(scheduler_contract))
    sample_order=[r["image"]["SHA256"] for r in rows]
    recipe=resume.build_contract(identity=identity,seed=claim["base_seed"],max_steps=25,
                                  sample_order=sample_order,device_ids=[0])
    write(output/"contract.json",recipe)
    restored=None
    if mode=="resume10":
        original=checkpoint_path(PACKAGE/"continuous",10)
        fd,identity,saved,recipe_pins=prepare_checkpoint(PACKAGE,recipe,deadline)
        try:
            payload=resume.load_checkpoint(original)
            confirm_checkpoint(fd,identity,PACKAGE)
        finally:
            os.close(fd)
        progress,restored=resume.restore_checkpoint(payload,recipe,model,optimizer,scheduler,scheduler_contract)
        resume.assert_restored_boundary(payload,model,optimizer,progress,scheduler,[0])
        need(restored["metadata"]["global_step"]==10 and restored["next"]["sample_cursor"]==10
             and restored["next"]["global_step"]==11,"Restored first use is not step 11 at cursor 10")
        write(output/"restore.json",{"restored_boundary_exact":True,"metadata":restored["metadata"],
                                     "next":restored["next"],"checkpoint_SHA256":saved["checkpoint_SHA256"],
                                     "recipe_pins":recipe_pins})
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
            ordinal=(step-1)%12
            data=sample(dataset,rows,ordinal)
            optimizer.zero_grad(set_to_none=True)
            objective,operands,_=forward(model,data,loss.FlowMatchSFTLoss,torch,recipe["seed"]+step,None)
            val=float(objective.detach())
            need(isfinite(val) and val>=0,"Native loss is nonfinite or negative")
            if step==1:
                need(abs(val-first_result["replayed_loss"])<=1e-6 and operands==first_result["noise_replay"],
                     "Step 1 differs from independently measured Style18 first update")
            objective.backward()
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in named.values()),
                 "Native backward has missing or nonfinite gradients")
            norm=torch.nn.utils.clip_grad_norm_(named.values(),1.)
            norm_value=float(norm)
            need(isfinite(norm_value) and norm_value>0 and all(bool(torch.isfinite(p.grad).all()) for p in named.values()),
                 "Native gradient norm or postclip result invalid")
            deadline()
            append(journal,dict(global_step=step,event="optimizer_intent"))
            optimizer.step()
            torch.cuda.synchronize()
            append(journal,dict(global_step=step,event="optimizer_returned"))
            progress=common.advance_optimizer_progress(progress,model,optimizer)
            need(all(p.grad is None and p._version==versions[name] for name,p in frozen.items()),
                 "A frozen parameter received a gradient or changed version")
            need(all(bool(torch.isfinite(p).all()) for p in named.values()),"Updated LoRA parameter is not finite")
            completed.append(step)
            append(metrics,dict(global_step=step,ordinal=ordinal,loss=val,preclip_norm=norm_value,operands=operands))
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
    need(completed==list(schedule(mode)) and [r["global_step"] for r in receipts]==[20,25],
         "Training and checkpoint denominators differ")
    write(output/"checkpoint-receipts.json",receipts)
    return dict(mode=mode,completed_global_steps=completed,optimizer_updates=len(completed),
                restored_exact=restored is not None,planned_steps=list(schedule(mode)),
                checkpoints=receipts,worker_training_section_seconds=time.monotonic()-begun,
                frozen_parameter_check="parameter version and no gradient",formal_A100=False,formal_performance=False)

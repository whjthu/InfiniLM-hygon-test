"""Render Style18 dev pairs from the exact initial and step-25 LoRA values."""

from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
import os
import time


PACKAGE = Path("/fusion-run")
TRAINED = Path("/recovery-checkpoint/continuous")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(4<<20),b""):
            h.update(data)
    return h.hexdigest()


def tensor_hash(torch, value):
    return sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def render(model,torch,output,claim,deadline):
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_training_resume_checkpoint as checkpoint
    from PIL import Image,ImageChops,ImageStat

    for entry in (common,checkpoint):
        path=Path(entry.__file__).resolve()
        pin=claim["framework_checkpoint_modules"][path.name]
        need(path.parent==Path("/common-framework/generation/src/gencuda")
             and path.stat().st_size==pin["bytes"] and digest(path)==pin["SHA256"],
             "Imported checkpoint reader differs from the common framework")
    plan_path=PACKAGE/"private-dev-plan.json"
    need(digest(plan_path)==claim["private_dev_plan"]["SHA256"],"Development plan identity differs")
    plan=json.loads(plan_path.read_bytes())
    samples=plan["samples"]
    need(len(samples)==len({s["seed"] for s in samples})==12 and plan["dev_images"]==6
         and not plan["training_development_overlap"] and not plan["blind_images_read"],"Development denominator differs")
    for i,s in enumerate(samples):
        need(s["ordinal"]==i//2 and s["replicate"]==i%2 and sha256(s["prompt"].encode()).hexdigest()==s["prompt_SHA256"],
             "Frozen sample identity differs")
    assert all(plan[k]==v for k,v in (("width",256),("height",256),("denoising_steps",8),("guidance_scale",1.0)))
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    initial={name:p.detach().cpu().clone() for name,p in named.items()}
    before=TRAINED/"INITIAL_ADAPTER_HASHES.private.json"
    need(digest(before)==claim["initial_training_hashes"]["SHA256"],"Original training initial adapter receipt differs")
    initial_hashes=json.loads(before.read_bytes())
    need(len(named)==476 and set(initial_hashes)=={n.removeprefix("pipe.dit.") for n in named},
         "Initial adapter parameter set differs")
    need(all(tensor_hash(torch,p)==initial_hashes[n.removeprefix("pipe.dit.")] for n,p in initial.items()),
         "Recreated initial adapter differs from the actual continuous run")
    path=TRAINED/"checkpoint-25.private.pt"
    need(path.stat().st_size==claim["continuous25_checkpoint"]["bytes"] and digest(path)==claim["continuous25_checkpoint"]["checkpoint_SHA256"],
         "Original continuous checkpoint body differs")
    raw=checkpoint.load_checkpoint(path)
    need(raw["metadata"]["global_step"]==25 and raw["metadata"]["next_data_cursor"]==1,
         "Candidate checkpoint is not the actual completed step 25")
    after=raw["state"]["trainable"]
    need(len(after)==476 and set(after)==set(initial) and common.tree_sha256(after)==claim["continuous25_checkpoint"]["adapter_SHA256"],
         "Trained adapter extent or identity differs")
    del raw
    states={"before":initial,"after":after}
    pipe=model.pipe
    model.eval()
    addresses={k:{} for k in ("before","after")}
    counts={"attempted":0,"completed":0,"failed":0}
    pairs=[]
    observations=output/"generations.jsonl"
    with observations.open("x",encoding="utf-8") as log:
        def log_row(data):
            log.write(json.dumps(data,sort_keys=True,allow_nan=False)+"\n")
            log.flush()
            os.fsync(log.fileno())

        for index,row in enumerate(samples):
            record={}
            order=("before","after") if index%2==0 else ("after","before")
            for arm in order:
                deadline()
                started=time.monotonic()
                counts["attempted"]+=1
                result=dict(sample_id=row["sample_id"],ordinal=row["ordinal"],replicate=row["replicate"],arm=arm,seed=row["seed"],
                            prompt_SHA256=row["prompt_SHA256"],reference_sha=row["reference"]["SHA256"],completed=False)
                original=pipe.generate_noise
                noises=[]
                def capture(*args,**kwargs):
                    noise=original(*args,**kwargs)
                    need(tuple(noise.shape)==(1,16,32,32) and noise.dtype==torch.bfloat16 and noise.device.type=="cuda",
                         "Native initial noise metadata differs")
                    noises.append(tensor_hash(torch,noise))
                    return noise
                try:
                    with torch.no_grad():
                        for name,value in named.items():
                            source=states[arm][name]
                            need(source.dtype==value.dtype and source.shape==value.shape,"Adapter tensor format differs")
                            value.copy_(source)
                        pipe.generate_noise=capture
                        image=pipe(prompt=row["prompt"],seed=row["seed"],rand_device="cuda",height=256,width=256,
                                   num_inference_steps=8,cfg_scale=1.0)
                    torch.cuda.synchronize()
                    need(len(noises)==1 and isinstance(image,Image.Image) and image.size==(256,256),"Native generation output differs")
                    image=image.convert("RGB")
                    target=output/f"{index:02d}-{arm}.private.png"
                    with target.open("xb") as f:image.save(f,format="PNG")
                    addresses[arm][index]=target
                    result.update(completed=True,initial_noise_SHA256=noises[0],image_bytes=target.stat().st_size,
                                  image_relative=target.name,image_SHA256=digest(target),
                                  RGB_SHA256=sha256(image.tobytes()).hexdigest(),seconds=time.monotonic()-started)
                    counts["completed"]+=1
                except Exception as exc:
                    result["error_type"]=type(exc).__name__
                    result["error"]=str(exc)[:210]
                    counts["failed"]+=1
                    if isinstance(exc,TimeoutError):
                        log_row(result)
                        raise
                finally:
                    pipe.generate_noise=original
                log_row(result)
                record[arm]=result
            if all(record[arm]["completed"] for arm in ("before","after")):
                a,b=record["before"],record["after"]
                with Image.open(addresses["before"][index]) as first, Image.open(addresses["after"][index]) as second:
                    pixel_mae=sum(ImageStat.Stat(ImageChops.difference(first,second)).mean)/3
                pairs.append(dict(sample_id=row["sample_id"],same_noise=a["initial_noise_SHA256"]==b["initial_noise_SHA256"],
                                  changed_pixels=a["RGB_SHA256"]!=b["RGB_SHA256"],pixel_mae=pixel_mae))
    return dict(planned_generations=24,planned_pairs=12,attempted=counts["attempted"],completed=counts["completed"],
                failed=counts["failed"],missing=24-counts["completed"]-counts["failed"],pairs=pairs,
                complete_same_noise_pairs=sum(item["same_noise"] for item in pairs),
                changed_pairs=sum(item["changed_pixels"] for item in pairs),
                initial_and_trained_adapter_identities_verified=True,optimizer_updates=0,formal_A100=False,
                formal_FID=False,CLIP_scored=False,passed=counts["completed"]==24 and all(row["same_noise"] for row in pairs))

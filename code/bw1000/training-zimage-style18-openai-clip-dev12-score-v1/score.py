"""Score the 24 fixed Style18 development generations with the pinned OpenAI CLIP release."""

from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import statistics
import time


BASE=Path("/clip-score")
GEN=Path("/paired-generation")
PROVIDER=Path("/clip-runtime/provider")
WEIGHT=Path("/clip-runtime/ViT-B-32.pt")


def need(condition,message):
    if not condition:
        raise RuntimeError(message)


def stream(path):
    h=sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(4<<20),b""):
            h.update(data)
    return h.hexdigest()


def read(path,max_bytes=1<<20):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=max_bytes,"Bounded score input unavailable")
    return path.read_bytes()


def main():
    started=time.monotonic()
    raw=read(BASE/"CLAIM.json")
    claim=json.loads(raw)
    need(os.environ["SLURM_JOB_ID"]==claim["parent"] and os.environ["SLURM_STEP_ID"].isdigit()
         and all(os.environ.get(key)=="" for key in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")),
         "Unexpected Slurm scope or accelerator mask")
    need(WEIGHT.stat().st_size==claim["weights"]["bytes"] and stream(WEIGHT)==claim["weights"]["SHA256"],
         "OpenAI pinned weight differs")
    manifest=read(PROVIDER/"PROVIDER_FILES.json")
    need(sha256(manifest).hexdigest()==claim["provider_manifest_SHA256"],"Official provider manifest changed")
    for name,pin in json.loads(manifest)["files"].items():
        path=PROVIDER/name
        content=read(path,2<<20)
        need(path.resolve().is_relative_to(PROVIDER) and len(content)==pin["bytes"]
             and sha256(content).hexdigest()==pin["SHA256"],"Official provider bytes changed")
    outcome=json.loads(read(GEN/"outcome.json"))
    worker_raw=read(GEN/"worker-result.json")
    worker=json.loads(worker_raw)
    need(outcome["passed"] and outcome["post_clean"] and worker["passed"]
         and outcome["request"]==worker["request"]==claim["generation_request"]
         and outcome["worker_result_SHA256"]==sha256(worker_raw).hexdigest(),
         "Original generation terminal identity differs")
    need(worker["development"]["completed"]==24 and worker["development"]["complete_same_noise_pairs"]==12,
         "No complete paired generation denominator")
    plan_raw=read(GEN/"private-dev-plan.json")
    need(sha256(plan_raw).hexdigest()==claim["private_plan_SHA256"],"Original development plan changed")
    plan=json.loads(plan_raw)["samples"]
    logs_raw=read(GEN/"development/generations.jsonl")
    records=[json.loads(line) for line in logs_raw.splitlines()]
    need(len(records)==24 and {r["sample_id"] for r in records}=={r["sample_id"] for r in plan},
         "Original image roster differs")

    import clip
    import torch
    from PIL import Image

    need(Path(clip.__file__).resolve()==PROVIDER/"clip/__init__.py" and not torch.cuda.is_initialized(),
         "Unexpected CLIP import location or device setup")
    torch.set_num_threads(4)
    model,preprocess=clip.load(str(WEIGHT),device="cpu",jit=False)
    model.eval()
    rows=[]
    with torch.inference_mode():
        prompts=clip.tokenize([s["prompt"] for s in plan],truncate=False)
        text=torch.nn.functional.normalize(model.encode_text(prompts).float(),dim=-1)
        tensors=[]
        index_rows=[]
        for index,item in enumerate(plan):
            pair=[row for row in records if row["sample_id"]==item["sample_id"]]
            need(len(pair)==2 and {row["arm"] for row in pair}=={"before","after"}
                 and all(row["completed"] and row["seed"]==item["seed"] and row["prompt_SHA256"]==item["prompt_SHA256"] for row in pair)
                 and pair[0]["initial_noise_SHA256"]==pair[1]["initial_noise_SHA256"],
                 "Native pair evidence differs")
            for row in sorted(pair,key=lambda row:row["arm"]):
                name=f'{index:02d}-{row["arm"]}.private.png'
                need(row["image_relative"]==name,"Generation file identity differs")
                path=GEN/"development"/name
                content=read(path)
                need(len(content)==row["image_bytes"] and sha256(content).hexdigest()==row["image_SHA256"],
                     "Generated image differs from its original receipt")
                with Image.open(path) as im:
                    need(im.mode=="RGB" and im.size==(256,256),"Generation image metadata differs")
                    tensors.append(preprocess(im))
                index_rows.append((index,row))
        image=[]
        for offset in range(0,24,4):
            image.append(torch.nn.functional.normalize(model.encode_image(torch.stack(tensors[offset:offset+4])).float(),dim=-1))
        embedded=torch.cat(image,dim=0)
        for rid,(index,row) in enumerate(index_rows):
            score=float(torch.dot(embedded[rid],text[index]))
            need(isfinite(score) and -1.00001<=score<=1.00001,"Nonfinite or invalid cosine")
            rows.append(dict(sample_id=row["sample_id"],arm=row["arm"],image_SHA256=row["image_SHA256"],
                             ordinal=row["ordinal"],replicate=row["replicate"],cosine=score))
    need(not torch.cuda.is_initialized(),"CLIP scoring initialized an accelerator")

    scores={arm:[next(row["cosine"] for row in rows if row["sample_id"]==s["sample_id"] and row["arm"]==arm) for s in plan]
            for arm in ("before","after")}
    differences=[a-b for a,b in zip(scores["after"],scores["before"],strict=True)]
    summary=dict(scoring="OpenAI CLIP ViT-B/32 normalized image-to-prompt cosine",requests=24,completed=24,
                 failed=0,paired_seeds=12,distinct_prompt_count=len({s["prompt_SHA256"] for s in plan}),
                 before_mean=statistics.fmean(scores["before"]),after_mean=statistics.fmean(scores["after"]),
                 mean_difference=statistics.fmean(differences),median_difference=statistics.median(differences),
                 improved=sum(x>0 for x in differences),declined=sum(x<0 for x in differences),equal=sum(x==0 for x in differences),
                 formal_A100=False,formal_FID=False,blind_set_used=False,actual_accelerator_allocated=0,
                 weight_SHA256=claim["weights"]["SHA256"],provider_manifest_SHA256=claim["provider_manifest_SHA256"],
                 original_generation_records_SHA256=sha256(logs_raw).hexdigest(),torch=str(torch.__version__),
                 elapsed_seconds=time.monotonic()-started,request=claim["request"],
                 Slurm_step=claim["parent"]+"."+os.environ["SLURM_STEP_ID"])
    with (BASE/"scores.private.json").open("x",encoding="utf-8") as out:
        json.dump({"summary":summary,"per_image":rows},out,allow_nan=False,sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(summary,allow_nan=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()

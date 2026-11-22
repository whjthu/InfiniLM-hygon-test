"""Exercise the original DiffSynth dataset and caller order with the exact COCO20 files."""

import csv
from hashlib import sha256
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time


BASE = Path("/coco20-inputs")
RUN = Path("/input-run")
NATIVE = Path("/native-source")
MODULE = "diffsynth.core.data"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(path, pin):
    need(path.is_file() and path.resolve(strict=True)==path, "Missing original input or provider")
    raw = path.read_bytes()
    need(len(raw)==pin["bytes"] and sha256(raw).hexdigest()==pin["SHA256"], "Original bytes differ")
    return raw


def main():
    result = dict(actual_DCU=0, actual_optimizer_updates=0, formal_effect=False,
                  original_manifest_present=False, training_ready=False)
    started = time.monotonic()
    claim = json.loads((RUN/"CLAIM.json").read_bytes())
    result["request"] = claim["request"]
    result["node"] = os.uname().nodename.split(".")[0]
    result["step"] = os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]
    need(result["node"]==claim["node"] and result["step"].startswith(claim["parent"]+".")
         and os.environ.get("SLURM_STEP_GPUS") in (None,"","none","N/A","(null)"), "Wrong zero-GRES CPU allocation")
    need(all(os.environ.get(k)=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")),
         "Accelerator visibility differed")
    contract = json.loads(checked(RUN/"COCO20_ORIGINAL.json",claim["input_contract"]))
    md = BASE/"metadata-zimage-20.csv"
    csv_bytes = checked(md,contract["metadata"])
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
    need(len(rows)==len(contract["row_order"])==20, "CSV count differs")
    for row,name,expected in zip(rows,contract["row_order"],contract["row_prompt_SHA256"],strict=True):
        need(row["image"]==name and sha256(row["prompt"].encode()).hexdigest()==expected, "Origin CSV row differs")
        checked(BASE/name,contract["images"][name])
    providers = {}
    for name,pin in claim["native_providers"].items():
        repo_path = NATIVE/"diffsynth/core/data"/(name+".py")
        checked(repo_path,pin)
        module = importlib.import_module(MODULE+"."+name)
        need(Path(module.__file__).resolve()==repo_path, "Imported provider was not pinned")
        providers[name] = pin["SHA256"]
    operators = importlib.import_module(MODULE+".operators")
    native = importlib.import_module(MODULE+".unified_dataset")
    need(native.ImageCropAndResize is operators.ImageCropAndResize and not operators.torch.cuda.is_initialized(),
         "Original PIL pipeline did not bind or accelerator initialized")
    ds = native.UnifiedDataset(base_path=str(BASE),metadata_path=str(md),repeat=1,data_file_keys=("image",),
            main_data_operator=native.UnifiedDataset.default_image_operator(
                    base_path=str(BASE),height=256,width=256,height_division_factor=16,width_division_factor=16))
    need(len(ds)==20, "Native input roster count differs")
    values = []
    for step in range(1,26):
        index=(step-1)%len(ds)
        item=ds[index]
        image=item["image"]
        prompt=sha256(item["prompt"].encode()).hexdigest()
        need(prompt==contract["row_prompt_SHA256"][index] and image.mode=="RGB" and image.size==(256,256),
             "Native decoded image or prompt differs")
        values.append(dict(step=step,index=index,RGB256_SHA256=sha256(image.tobytes()).hexdigest(),
                           prompt_SHA256=prompt))
    need(all(values[i]["RGB256_SHA256"]==values[20+i]["RGB256_SHA256"] and
             values[i]["prompt_SHA256"]==values[20+i]["prompt_SHA256"] for i in range(5)),
         "Repeated native inputs changed")
    need(not operators.torch.cuda.is_initialized(), "Accelerator initialized during native input check")
    result.update(passed=True, rows=20, decoded_occurrences=25, exact_repeats=5, provider_SHA256=providers,
                  torch=str(operators.torch.__version__),torchvision=str(operators.torchvision.__version__),
                  input_values=values, elapsed_s=time.monotonic()-started, A100_RGB256_compared=False)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()

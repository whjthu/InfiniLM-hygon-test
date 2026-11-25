"""Check pinned TextVQA/TextOCR street-scene rows with the original training processor."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path

from build_textvqa_street_scene import PROMPT
from check_qwen_native_inputs import CONFIG_SHA256, MAX_PIXELS, SOURCE_VERSIONS, TEMPLATE_SHA256, read_pinned, require


DATASET_SHA256 = "dd6c882665db32a47ff4736392b63f0d83e4ea0caa226534ec2f1093c6c04ecc"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def pin(value):
    return sha256(encode(value)).hexdigest()


def frozen_inputs(dataset, inventory, inventory_sha):
    path = dataset.resolve(strict=True)
    require(path == dataset.absolute() and path.name == "TRAIN_SWIFT.jsonl", "exact private training dataset")
    images = json.loads(read_pinned(inventory, inventory_sha))
    require(len(images) == 16, "original image denominator")
    for name, identity in images.items():
        absolute = path.parent / name
        require(absolute.parent == path.parent and absolute.resolve(strict=True) == absolute and name.endswith(".jpg"),
                "linked or unexpected training image")
        raw = read_pinned(absolute, identity["SHA256"])
        require(len(raw) == identity["bytes"], "original image length differs")
    ordered = []
    for line in read_pinned(dataset, DATASET_SHA256).splitlines():
        row = json.loads(line)
        first, second = row["messages"]
        image, = row["images"]
        require(set(row) == {"images", "messages"} and image in images and first["role"] == "user"
                and second["role"] == "assistant" and first["content"].startswith("<image>\n")
                and first["content"].count("<image>") == 1 and second["content"] and "<image>" not in second["content"],
                "invalid training row shape")
        task = "OCR" if first["content"][8:] == PROMPT else "VQA"
        ordered.append((task, {"messages":row["messages"],"images":[str(path.parent / image)]}))
    require(len(ordered)==41 and sum(task=="VQA" for task,_ in ordered)==26, "mixed training declaration differs")
    return ordered


def check(dataset, inventory, inventory_sha, model_dir):
    require(os.environ.get("SLURM_STEP_GPUS") in (None,"","(null)"), "GPU assigned to the native input probe")
    versions = {name:importlib.metadata.version(name) for name in SOURCE_VERSIONS}
    require(versions == SOURCE_VERSIONS, "original Swift processor provider versions")
    inputs = frozen_inputs(dataset,inventory,inventory_sha)
    config = json.loads(read_pinned(model_dir/"config.json",CONFIG_SHA256))
    marker = config["image_token_id"]

    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.templates import qwen

    require(not torch.cuda.is_available(), "device runtime visible to zero-GRES input check")
    read_pinned(Path(qwen.__file__),TEMPLATE_SHA256)
    model,processor = get_model_processor(str(model_dir),model_type="qwen3_vl",load_model=False,
                                          download_model=False,device_map="cpu")
    require(model is None, "unexpected model materialization")
    template = get_template(processor,template_type="qwen3_vl",max_length=4096,max_pixels=MAX_PIXELS,
                            padding_side="right",padding_free=False,truncation_strategy="raise")
    template.set_mode("train")
    evidence, failures, distinct_lengths = [], [], {}
    for index,(task,row) in enumerate(inputs):
        try:
            encoded = template.encode(row)
            ids,labels = encoded["input_ids"],encoded["labels"]
            require(0 < len(ids)==len(labels)<=4096,"invalid training sequence")
            image = [i for i,token in enumerate(ids) if token==marker]
            trained = [i for i,label in enumerate(labels) if label!=-100]
            require(image and trained and trained[0]>image[-1] and all(labels[i]==-100 for i in image)
                    and all(ids[i]==labels[i] for i in trained),"image or loss mask differs")
            pixels,grid=encoded["pixel_values"],encoded["image_grid_thw"]
            require(pixels.device.type==grid.device.type=="cpu" and tuple(grid.shape)==(1,3)
                    and pixels.numel()>0 and bool(torch.isfinite(pixels).all()),"bad native image preprocessing")
            evidence.append(dict(row=index,task=task,tokens=len(ids),image_tokens=len(image),supervised_tokens=len(trained),
                                 input_ids_SHA256=pin(ids),labels_SHA256=pin(labels),grid=grid.tolist()))
            if len(distinct_lengths)<2:
                distinct_lengths.setdefault(len(ids),encoded)
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:
            failures.append(dict(row=index,task=task,error=type(exc).__name__,detail_SHA256=pin(str(exc))))
    report = dict(planned=41,processed=len(evidence),failures=failures,provider_versions=versions,rows=evidence,
                  dataset_SHA256=DATASET_SHA256,media_inventory_SHA256=inventory_sha,template_SHA256=TEMPLATE_SHA256,
                  model_materialized=False,optimizer_updates=0,GPU=0,collator_passed=False,
                  state="NATIVE_PREPROCESSOR_INCOMPLETE")
    if failures:
        return report
    require(len(distinct_lengths)==2,"need distinct encoded lengths for native padding")
    with torch.device("meta"):
        template.dummy_model,_=get_model_processor(str(model_dir),model_type="qwen3_vl",return_dummy_model=True,
                                                   download_model=False)
    parameters=list(template.dummy_model.parameters())
    require(parameters and all(p.device.type=="meta" for p in parameters),"meta-model materialized parameters")
    batch=template.data_collator(list(distinct_lengths.values()))
    labels,attention=batch["labels"],batch["attention_mask"].bool()
    lengths=list(distinct_lengths)
    require(tuple(labels.shape)==tuple(attention.shape)==(2,max(lengths)) and attention.sum(dim=1).tolist()==lengths
            and bool((labels[~attention]==-100).all()) and tuple(batch["image_grid_thw"].shape)==(2,3),
            "native collator attention or loss mask differs")
    report.update(state="ACTUAL_NATIVE_TRAINING_INPUTS_CHECKED",collator_passed=True,collated_lengths=lengths)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--inventory-sha",required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    args=parser.parse_args()
    result=check(args.dataset,args.inventory,args.inventory_sha,args.model_dir)
    print(json.dumps(result,sort_keys=True))
    return int(not result["collator_passed"])


if __name__=="__main__":
    raise SystemExit(main())

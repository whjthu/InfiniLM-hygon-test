"""Render the original target Qwen recipe for the exact staged street-scene training bytes."""

from hashlib import sha256
import inspect
import json
from pathlib import Path, PurePosixPath

from evidence import (DATASET, DATA_SHA, NATIVE_OUT, NATIVE_PUBLIC_SHA, NATIVE_REPORT_SHA, NATIVE_REQUEST,
                      STAGE, TRANSFER_SHA, ZIP_SHA, dataset_rows, fingerprints, native_report, need, pin, read)


ORIGINAL_SHA = "28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400"
RECIPE_SHA = "c251e21dbaa7d4258add5275f1a959af2536128cac7d86374d1bf7de83bb5806"
BINDING_SHA = "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
ASSETS_SHA = "3abef5782032d9c6fe41dcc876c27320b0ff9314771259b4b5eec563b4a2599f"
TERMINAL_220_SHA = "a9eed82cb336b2024bdbb126e9c68054aeda0c1f33dd21456ad863ce36f945d7"
INIT_SHA = "2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f"
INIT_NAMED_SHA = "086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137"
PLUGIN = "street_updates.py"
CALLBACK = "qwen_street_scene_first_two"
BASELINE = "swift_callback_nvidia_baseline_v2.py"
REQUIRED = {"--per_device_train_batch_size":"1","--gradient_accumulation_steps":"4","--max_length":"4096",
            "--max_pixels":"262144","--torch_dtype":"bfloat16","--freeze_vit":"true","--lora_rank":"16",
            "--lora_alpha":"32","--lora_dropout":"0.05","--max_steps":"25","--learning_rate":"1e-4",
            "--lr_scheduler_type":"cosine","--optim":"adamw_torch","--warmup_ratio":"0.03",
            "--seed":"42","--data_seed":"42","--dataset_shuffle":"true","--dataloader_drop_last":"false",
            "--split_dataset_ratio":"0","--packing":"false","--padding_free":"false","--padding_side":"right",
            "--attn_impl":"sdpa","--gradient_checkpointing":"true"}


def value(argv, option):
    need(argv.count(option)==1 and argv.index(option)+1<len(argv)
         and not argv[argv.index(option)+1].startswith("--"), "EXACT_ORIGINAL_OPTION_"+option)
    return argv.index(option)+1


def group(argv, option):
    at=value(argv,option)
    end=at
    while end<len(argv) and not argv[end].startswith("--"):
        end+=1
    return at,end


def history(public_native, terminal_220, assets, binding):
    native=json.loads(read(public_native,8192,NATIVE_PUBLIC_SHA))
    old=json.loads(read(terminal_220,65536,TERMINAL_220_SHA))
    model=json.loads(read(assets,65536,ASSETS_SHA))["model"]
    init=json.loads(read(binding,2*1024**2,BINDING_SHA))
    need(native["request"]==NATIVE_REQUEST and native["exact_original_C0"]=="1194900.1251"
         and native["native_report_SHA256"]==NATIVE_REPORT_SHA and native["native_collator_pass"] is True
         and native["input_rows"]==native["processed_rows"]==41 and native["tags"]=={"OCR":15,"TextVQA":26}
         and native["new_DCU_steps"]==native["optimizer_updates"]==0
         and old["observation"]["rows"][1][:4]==["1087656.220","training-qwen-native25-hip-v3","COMPLETED","0:0"]
         and old["observation"]["model_training_steps"]==25 and len(old["observation"]["ranks"])==8
         and len(model)==14 and init["initial_state"]["bytes"]==174821899
         and init["initial_state"]["SHA256"]==INIT_SHA and init["initial_trainables_SHA256"]==INIT_NAMED_SHA,
         "EXACT_TARGET_NATIVE_INPUTS_AND_220_MODEL_IDENTITY")
    return {"model_files":model,"model_manifest_SHA256":pin(model),"model_member_count":14,
            "initial_state":init["initial_state"],"initial_trainables_SHA256":INIT_NAMED_SHA}


def stage_proof():
    staged=json.loads(read(Path(STAGE)/"staged.json",2048,"0e070b110cbf3a951f4ad6107c16d18a59e71ac4ba01173d29715a5511430f60"))
    need(staged["request"]==NATIVE_REQUEST and staged["zip_SHA256"]==ZIP_SHA and staged["members"]==22
         and staged["transfer_receipt_SHA256"]==TRANSFER_SHA and staged["state"]=="EXACT_PRIVATE_INPUTS_INSTALLED",
         "EXACT_PRIOR_PRIVATE_STAGE")
    dataset=Path(DATASET)
    rows=dataset_rows(dataset,dataset.parent/"MEDIA_PINS.json")
    return fingerprints(native_report(Path(NATIVE_OUT)),rows)


def prepare(original_recipe, original, binding, framework, output, plugin, native_port, *, epoch, deadline, maximum):
    """Build eight-rank argv and the data cwd for the original process owner."""
    need(sha256(original).hexdigest()==ORIGINAL_SHA and
         sha256(read(inspect.getfile(original_recipe),16384)).hexdigest()==RECIPE_SHA,"EXACT_ORIGINAL_220_RECIPE")
    read(binding,2*1024**2,BINDING_SHA)
    framework, output, plugin=map(PurePosixPath,(framework,output,plugin))
    need(all(path.is_absolute() and ".." not in path.parts and len(path.parts)>1 for path in (framework,output,plugin))
         and output.is_relative_to(STAGE.parent) and not output.is_relative_to(STAGE)
         and plugin.name==PLUGIN,"EXACT_PRIVATE_TRAINING_OUTPUT_AND_PLUGIN_PATH")
    launch=original_recipe(original,binding,framework/"vlm/runtime/swift_entry.py",framework,output,native_port,
                          epoch=epoch,deadline=deadline,max_seconds=maximum)
    argv=launch["command"]
    need(all(argv[value(argv,key)]==required for key,required in REQUIRED.items()),
         "ORIGINAL_220_MATH_AND_SAMPLER_CONFIGURATION")
    need(not any(key in argv for key in ("--resume_from_checkpoint","--resume_only_model","--val_dataset",
                                        "--test_dataset","--eval_dataset","--save_strategy")),
         "FRESH_TRAIN_ONLY_NO_CHECKPOINT_PROMOTION_ENTRY")
    argv[value(argv,"--dataset")]=str(DATASET)
    first,last=group(argv,"--callbacks")
    need(argv[first:last]==["native_initial_trainables","native_first_updates","nvidia_baseline"],
         "ORIGINAL_INITIALIZATION_AND_OBSERVER_ORDER")
    argv[first:last]=["native_initial_trainables",CALLBACK]
    first,last=group(argv,"--external_plugins")
    plugins=argv[first:last]
    need([PurePosixPath(name).name for name in plugins]==[BASELINE,"native_initial_trainables.py","native_first_updates.py"],
         "ORIGINAL_220_TRAINING_PLUGIN_IDENTITY")
    argv[first:last]=[plugins[1],str(plugin)]
    argv.extend(["--save_strategy","no"])
    launch["required_assets"].update(dataset=str(DATASET),plugins=[plugins[1],str(plugin)])
    launch["environment_overrides"].update(DATASET=str(DATASET),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    launch.update(cwd=str(DATASET.parent),dataset_SHA256=DATA_SHA,native_input_report_SHA256=NATIVE_REPORT_SHA,
                  source_native_step="1194900.1251",planned_optimizer_updates=2,nominal_scheduler_steps=25,
                  configured_global_batch=32,initial_trainables_SHA256=INIT_NAMED_SHA,
                  checkpoint_promotion=False,Model_GO=False)
    return launch

"""New domain plan through the frozen T05 native launcher, never a new train loop."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
MODEL=ROOT/'training-minicpm-original-model-v1/MiniCPM-V-4.6'
TRANSLATOR_SHA='71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2'
ORIGINAL_SHA='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789'
DATA_SHA='0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def translator(path):
    raw=Path(path).read_bytes();need(sha(raw)==TRANSLATOR_SHA,'common translator differs')
    old=b'''        if (case != "T05.sft" or len(observers) != 1
                or observers[0]["callback"] != "native_resume_witness"
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("MiniCPM full resume requires its single dedicated native witness")'''
    new=b'''        if (case != "T05.sft" or len(observers) != 1
                or observers[0]["callback"] != "minicpm_usgs_resume_witness"
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("MiniCPM domain resume requires its single dedicated native witness")'''
    need(raw.count(old)==1,'original dedicated T05 resume guard differs')
    candidate=raw.replace(old,new,1)
    ns={'__name__':'minicpm_domain_common_translator','__file__':str(path)}
    exec(compile(candidate,str(path),'exec'),ns)
    return ns['build_launch'],sha(candidate)


def command_plan(translator_file, original, run_root, dataset_sha, arm, master_port):
    """Pure command construction; this is not GPU admission or permission to run."""
    need(sha(original)==ORIGINAL_SHA and arm in ('continuous','resumed'),'original T05 recipe or arm differs')
    run_root=PurePosixPath(run_root)
    need(run_root.is_absolute() and '..' not in run_root.parts and len(dataset_sha)==64,'exact Linux root and view pin required')
    build,source_sha=translator(translator_file)
    framework=ROOT/'training-qwen-a68952c-swift-initial-CPU-v1/framework'
    plugins=ROOT/'training-vlm-original-assets-v1/plugins'
    checkpoint=run_root/'continuous/train/checkpoint-20' if arm=='resumed' else None
    callback='minicpm_usgs_resume_witness' if checkpoint else 'minicpm_usgs_continuous_witness'
    plugin=run_root/'src/minicpm_usgs_witness.py'
    plan=build('T05.sft',original,framework/'vlm/runtime/swift_entry.py',MODEL.parent,
               run_root/'view/training800.jsonl',plugins,run_root/arm/'train',master_port,
               diagnostic_plugin=plugin,diagnostic_callback=callback,diagnostic_callback_position='before',
               resume_from_checkpoint=checkpoint)
    argv=plan['command']
    if checkpoint is None:
        need(argv.count('nvidia_baseline')==1,'original callback count differs')
        argv.remove('nvidia_baseline')
    else:
        need(argv.count('nvidia_baseline')==0,'resume builder kept original callback')
    baseline=str(plugins/'swift_callback_nvidia_baseline_v2.py')
    need(argv.count(baseline)==1,'original baseline plugin differs')
    argv.remove(baseline);plan['required_assets']['plugins'].remove(baseline)
    # Resume builder removes the original baseline itself; exactly the new witness remains.
    need('nvidia_baseline' not in argv,'unexpected original callback in new domain task')
    for option,value in {
      '--model_type':'minicpmv4_6','--template':'minicpmv4_6','--attn_impl':'sdpa',
      '--gradient_checkpointing':'false','--freeze_vit':'true','--freeze_aligner':'true',
      '--max_length':'4096','--max_steps':'25','--save_steps':'10','--warmup_steps':'5',
      '--per_device_train_batch_size':'1','--gradient_accumulation_steps':'4',
      '--lora_rank':'16','--lora_alpha':'32','--lora_dropout':'0.05',
      '--optim':'adamw_torch','--lr_scheduler_type':'cosine','--torch_dtype':'bfloat16',
    }.items():
        need(argv.count(option)==1 and argv[argv.index(option)+1]==value,'original MiniCPM option differs: '+option)
    need(argv.count('--model')==1,'ambiguous model')
    argv[argv.index('--model')+1]=str(MODEL)
    env=plan['environment_overrides']
    need(env['NPROC_PER_NODE']=='8' and env['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7','original eight-rank topology differs')
    plan['required_assets']['model_directory']=str(MODEL)
    plan.update(case='minicpmv4_6.usgs88-exposure25-resume20',source_recipe_case='T05.sft',arm=arm,
                source_dataset_SHA256=DATA_SHA,dataset_SHA256=dataset_sha,unique_source_rows=88,physical_rows=800,
                planned_optimizer_updates=25 if arm=='continuous' else 5,
                checkpoint_steps=[10,20,25] if arm=='continuous' else [25],
                candidate_translator_SHA256=source_sha,actual_optimizer_updates=0,
                formal_training_qualified=False,training_GO=False,GPU_GO=False)
    return plan


def require_native88(raw,expected_sha):
    need(isinstance(expected_sha,str) and len(expected_sha)==64 and sha(raw)==expected_sha,'actual native88 independent pin required')
    r=json.loads(raw)
    need(r.get('state')=='MINICPM_USGS88_NATIVE_CPU_INDEPENDENT_METADATA_PASS' and r.get('passed') is True
         and r.get('planned_rows')==r.get('completed_rows')==88 and r.get('missing_rows')==r.get('failed_rows')==0
         and r.get('actual_encode_calls')==176 and r.get('actual_collator_calls')==88
         and r.get('suffix_EOS_and_real_separator_verified') is True and r.get('dev_or_blind_read') is False,
         'native88 actual full-mask/suffix/provider qualification incomplete')
    return r

"""TL5 .1171 worker derivative. Source rendering has no process or Slurm effects."""
import json
from pathlib import Path, PurePosixPath
import re

from scope_contract import INPUT_ROOT, need, sha

ORIGINAL_WORKER_SHA = "6c9437969121b53e735611131fbd05e9dbb1b4bf18d4714002bb6e7d121b2348"
CORE_SHA = "dcf5356c2879bb2c61b1de1f0f037d8fd312542775e46e6556c6f6d89dc0ee82"
FRAMEWORK_SHA = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"


def check_scope(scope):
    request = scope["request"]
    need(type(request) is str and re.fullmatch(r"[0-9a-f]{32}", request)
         and request not in {"fa68d03ea04f4588ac9c21459097d50e", "1fde0c1744ce40898ed9d7a2724f0fac"}
         and scope["label"] == "training-minicpm-off-native2-" + request[:12]
         and re.fullmatch(r"[0-9]+", scope["parent"]) and re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", scope["node"])
         and type(scope["port"]) is int and 1024 <= scope["port"] < 65536 and scope["port"] != 29929,
         "OFF_NEW_NATIVE_SCOPE")
    return scope


def change(source, old, new):
    need(source.count(old) == 1, "OFF_ORIGINAL_SOURCE_ANCHOR")
    return source.replace(old, new, 1)


def command_plan(recipe, translator, original, run, view_sha, port):
    run = PurePosixPath(run)
    plan = recipe.command_plan(translator, original, str(run), view_sha, "continuous", port)
    argv = plan["command"]
    need(argv[0].endswith("/training-qwen-a68952c-swift-initial-CPU-v1/framework/vlm/runtime/swift_entry.py")
         and argv.count("minicpm_usgs_continuous_witness") == 1
         and argv.count(str(run / "src/minicpm_usgs_witness.py")) == 1
         and argv[argv.index("--dataset")+1] == str(run / "view/training800.jsonl")
         and argv[argv.index("--max_steps")+1] == "25" and argv[argv.index("--warmup_steps")+1] == "5"
         and not any(x == "--resume_from_checkpoint" for x in argv), "OFF_ORIGINAL_CONTINUOUS_PLAN")
    argv[0] = str(run / "framework/vlm/runtime/swift_entry.py")
    argv[argv.index("minicpm_usgs_continuous_witness")] = "minicpm_off_two_updates"
    old_plugin, new_plugin = (str(run / "src" / x) for x in ("minicpm_usgs_witness.py", "off_callback.py"))
    argv[argv.index(old_plugin)] = new_plugin
    plugins = plan["required_assets"]["plugins"]
    plugins[plugins.index(old_plugin)] = new_plugin
    plan.update(case="minicpmv4_6.off6-exposure800-prefix2", unique_source_rows=6, physical_rows=800,
                planned_optimizer_updates=2, checkpoint_steps=[2], nominal_max_steps=25)
    del plan["source_dataset_SHA256"]
    return plan


def render_worker(original, scope, stage_sha):
    check_scope(scope)
    need(sha(original) == ORIGINAL_WORKER_SHA and re.fullmatch(r"[0-9a-f]{64}", stage_sha), "OFF_ORIGINAL_WORKER_OR_STAGE")
    source = original.decode()
    source = change(source, "LABEL='training-minicpm-usgs-exposure25-resume20-target-v2'", f"LABEL={scope['label']!r}")
    source = change(source, "REQUEST='fa68d03ea04f4588ac9c21459097d50e'", f"REQUEST={scope['request']!r}")
    source = change(source, "PARENT,NODE,PORT='1103606','f13r1n19',29929", f"PARENT,NODE,PORT={scope['parent']!r},{scope['node']!r},{scope['port']}")
    source = change(source, "MINICPM_USGS_TARGET_WORKER_CONFIG_V1", "MINICPM_OFF_TARGET_WORKER_CONFIG_V1")
    a = source.index("def source_and_data(config):")
    b = source.index("def checkpoint_inventory(", a)
    source = source[:a] + f'''def source_and_data(config):
    sys.path.insert(0,str(RUN/'src'))
    from scope_contract import exact_4479, view_bytes, INPUT_SHA, INPUT_ROOT
    from off_inputs import read_inputs
    from render_worker import command_plan
    need(config['asset_stage']['SHA256']=={stage_sha!r},'OFF_STAGE_PIN')
    stage=checked_record(config['asset_stage'],RUN)
    need(stage['request']==REQUEST and (stage['parent'],stage['node'],stage['port'])==(PARENT,NODE,PORT)
         and stage['source_rows']==6 and stage['train_media']==4 and stage['input_manifest_SHA256']==INPUT_SHA,'OFF_STAGE_SCOPE')
    need(file_pin(RUN/'src/SOURCE_FILES.json',2*1024**2)['SHA256']==CORE_SHA,'OFF_COMMON_CORE_MANIFEST')
    for name,pin in read(RUN/'src/SOURCE_FILES.json')['files'].items():
        need(Path(name).name==name and file_pin(RUN/'src'/name,2*1024**2)==pin,'OFF_COMMON_CORE_MEMBER')
    for name,pin in stage['overlay'].items():
        need(name in ('scope_contract.py','off_callback.py','off_inputs.py','render_worker.py','reload_cp2.py')
             and file_pin(RUN/'src'/name,2*1024**2)==pin,'OFF_OVERLAY_MEMBER')
    need(set(stage['overlay'])=={{'scope_contract.py','off_callback.py','off_inputs.py','render_worker.py','reload_cp2.py'}},'OFF_OVERLAY_MEMBERS')
    raw,ledger=view_bytes(read_inputs(Path(INPUT_ROOT),INPUT_SHA))
    view=RUN/'view/training800.jsonl'
    need(view.read_bytes()==raw and read(RUN/'view/exposure-ledger-private.json')==ledger,'OFF_EXACT_TARGET_800')
    framework=stage['framework']
    need(framework['files']==393 and framework['base_manifest_SHA256']==BASE_MANIFEST_SHA,'OFF_FRAMEWORK_BASELINE')
    files=checked_record(framework['manifest'],RUN,2*1024**2)
    need(len(files)==393,'OFF_FRAMEWORK_SIZE')
    for name,pin in files.items():
        rel=Path(name)
        need(not rel.is_absolute() and '..' not in rel.parts and file_pin(RUN/'framework'/rel,2*1024**2)==pin,'OFF_FRAMEWORK_FILE')
    recipe=load_module('minicpm_worker_recipe',RUN/'src/minicpm_domain_recipe.py')
    expected=command_plan(recipe,RUN/'framework/tools/hygon_vlm_recipe.py',(RUN/'src/original-launch.json').read_bytes(),
                         RUN,ledger['view_SHA256'],PORT)
    plan=checked_record(stage['plan'],RUN)
    need(plan==expected,'OFF_EXACT_NATIVE_PLAN')
    return {{'continuous':plan}}


''' + source[b:]
    source = change(source, "expected=100 if arm=='continuous' else 20", "expected=8")
    source = change(source, "steps==list(range(1,26) if arm=='continuous' else range(21,26))", "steps==[1,2]")
    source = change(source, "value['completed_global_step']==25", "value['completed_global_step']==2")
    source = change(source, "value['saved_checkpoints']==([10,20,25] if arm=='continuous' else [25])", "value['saved_checkpoints']==[2] and value['request']==REQUEST")
    source = source.replace("25 if arm=='continuous' else 5", "2").replace("100 if arm=='continuous' else 20", "8")
    source = change(source, "witness=dict(asset_binding=str(run/'src/ASSETS.json'))", "witness=dict(asset_binding=str(run/'src/ASSETS.json'),request=REQUEST)")
    source = change(source, "planned_total_optimizer_updates=30,planned_arms=['continuous','resumed']", "planned_total_optimizer_updates=2,planned_native_forwards=64,planned_arms=['continuous']")
    source = change(source, "for arm in ('continuous','resumed'):", "for arm in ('continuous',):")
    source = change(source, "for step in ([10,20,25] if arm=='continuous' else [25]):checkpoint_inventory(RUN,arm,step,budget)",
                    "checkpoint_inventory(RUN,arm,2,budget)")
    a = source.index("        compare=load_module('minicpm_target_compare'")
    b = source.index("        outcome['passed']=True", a)
    source = source[:a] + '''        from scope_contract import audit_ranks
        records=[read(RUN/'continuous/train/minicpm-usgs-witness'/('rank-%02d'%i)/'result.json') for i in range(8)]
        evidence=audit_ranks(records,REQUEST)
        evidence['checkpoint_inventory']=file_pin(RUN/'continuous-checkpoint-2-files.json')
        evidence['independent_CPU_reload_required']=True
        save(RUN/'off-first-two.json',evidence)
''' + source[b:]
    source = change(source, "need(socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']==PARENT,'wrong original Slurm node/parent')",
                    "need(socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']==PARENT and os.environ.get('SLURM_GPUS_ON_NODE') in (None,'8'),'OFF_JOB_OR_STEP_DEVICE_COUNT')")
    compile(source, "worker.py", "exec")
    return source.encode()

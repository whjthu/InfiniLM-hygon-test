"""Check the complete new CPU scope before helper/SSH/Slurm or runtime imports."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

D = Path(__file__).resolve().parent


def verify(root=D, *, prepared=False):
    def need(ok, code):
        if not ok:
            raise ValueError(code)
    def obj(name):
        return json.loads((root / name).read_bytes())
    files = obj('SOURCE_FILES.json')['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'SOURCE_FILES.json'}
    need(set(files) == actual, 'EXACT_NEW_SOURCE_SCOPE')
    for name, expected in files.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'SOURCE_PATH')
        p = root / name
        raw = p.read_bytes()
        need(not p.is_symlink() and len(raw) == expected['bytes'] and
             hashlib.sha256(raw).hexdigest() == expected['SHA256'], 'SOURCE_PIN:' + name)
        if name.endswith('.py'):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    plan, claim = obj('PLAN.json'), obj('CLAIM.json')
    need(type(prepared) is bool,'EXPLICIT_PREPARATION_MODE')
    expected_GO=not prepared
    need(claim['root_GO'] is expected_GO and claim['CPU_GO'] is expected_GO and
         all(claim[k] is False for k in ('GPU_GO', 'Model_GO', 'training_GO', 'formal17_GO',
                                        'performance_GO', 'effect_GO')), 'CPU_SCOPE_ONLY')
    need(plan['label'] == claim['label'] == 'training-internvl-USGS72-native-CPU-v1' and
         plan['case'] == claim['case'] == 'T03.sft', 'EXACT_CASE_SCOPE')
    need(plan['whole_seconds'] == claim['budget']['single_epoch_seconds'] == 300 and
         plan['work_seconds'] == 210 and plan['Slurm_max_seconds'] == 240 and
         plan['cleanup_reserve_seconds'] == 45, 'ONE_BUDGET')
    for key in ('work_seconds_max', 'Slurm_seconds_max', 'cleanup_accounting_reserve_seconds'):
        need(key not in plan, 'STALE_OLD_BUDGET_FIELD')
    need(plan['GPU'] == claim['resources']['GRES'] == 0 and plan['CPU'] == 8 and
         plan['memory_MiB'] == 12288 and plan['model_view_basename'] == 'InternVL3_5-8B-HF',
         'ORIGINAL_HF_METADATA_CPU_VIEW')
    need(plan['framework_root'] == claim['source_dependencies']['framework_root'] ==
         '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-a68952c-swift-initial-CPU-v1/framework',
         'KNOWN_COMMON393_PHYSICAL_ROOT')
    need(len(obj('FRAMEWORK_FILES.json')['files']) == 393 and
         hashlib.sha256((root / 'FRAMEWORK_FILES.json').read_bytes()).hexdigest() ==
         'e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41', 'QUALIFIED_COMMON393_IDENTITY')
    need(len(obj('MODEL_METADATA.json')['files']) == 13 and len(obj('INPUT_ROWS.json')['rows']) == plan['rows'] == 72, 'FULL13_NEW72_DENOMINATORS')
    need(plan['request_id']==claim['request_id']=='8110b01cd24846f8b6fcd08f0abda5a0' and plan['unbounded_max_length']==1000000,'NEW72_FIXED_REQUEST_FULL_INPUT')
    need('INPUT_REFERENCE.json' not in files and 'ORIGINAL_INPUT_FILES.json' not in files,'OLD64_VALUE_REFERENCE_REMOVED')
    contract=obj('DATA_CONTRACT.json')
    need(len(contract['files'])==14 and contract['target_data']==plan['assets_root']==plan['RUN']+'/data','EXACT14_TRAIN_ONLY_ASSETS')
    need(plan['dataset']==plan['assets_root']+'/train-swift.jsonl' and contract['files']['train-swift.jsonl']==dict(bytes=plan['dataset_bytes'],SHA256=plan['dataset_SHA256']),'NEW72_TRAIN_DATASET')
    need([r['ordinal'] for r in obj('INPUT_ROWS.json')['rows']]==list(range(1,73)) and len({r['source_ID_SHA256'] for r in obj('INPUT_ROWS.json')['rows']})==72,'NEW72_ORDERED_IDENTITIES')
    binding = obj('NATIVE_BINDING.json')
    need(binding['spec']['case'] == 'T03.sft' and len(binding['trainable_manifest']) == 504 and
         sum(v['numel'] for v in binding['trainable_manifest']) == 43646976, 'ORIGINAL_META_PEFT')
    target=obj('TARGET_MATH_SOURCES.json')
    need(target['schema']=='REVIEWED_TARGET_MATH_SOURCE_CONTRACT_V1' and target['case']=='T03.sft'
         and target['reference_binding']==files['NATIVE_BINDING.json']
         and target['reference_model_source_SHA256']==binding['model_source_SHA256']
         and target['numerical_equivalence_proven'] is target['GPU_GO'] is target['Model_GO'] is False,'REFERENCE_NOT_REWRITTEN_OR_PROMOTED')
    need(target['target_sources']=={
        'transformers.models.internvl.':{'bytes':37559,'SHA256':'632fc3b219cd1d08c322e786fa25d7dbf07059f6ccb6fe22dd1c45592415ad9a'},
        'transformers.models.qwen3.':{'bytes':23353,'SHA256':'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab'}},'EXACT_REVIEWED_PLATFORM_CONTRACT')
    need(claim['target_math_contract']['SHA256']==files['TARGET_MATH_SOURCES.json']['SHA256'],'CLAIM_TARGET_CONTRACT_PIN')
    expected_helpers = {
        'held_child.py': '06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351',
        'owned_processes.py': '5a3dd9b7530526f324b230ef0fe5be6ff2953f1331718d6520a1362bca229817',
        'slurm_accounting.py': '15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b'}
    need(all(files[name]['SHA256'] == expected for name, expected in expected_helpers.items()),
         'QUALIFIED_HELD_TREE_ACCOUNTING_HELPERS')
    need(plan['memory_control']['metric']=='STEP_CGROUP_CHARGED_MEMORY' and
         plan['memory_control']['soft_limit_bytes']==8*1024**3 and
         plan['memory_control']['supported_controllers']==[1,2] and
         plan['memory_control']['RSS_sum_equivalence_claimed'] is False,'EXPLICIT_STEP_MEMORY_CONTRACT')
    need(all(n in files for n in ('step_memory.py','step_tree.py','step_control.py','native_release.py','cgroup_evidence.py')),
         'COMPLETE_NEW_CONTROL_AND_CONSUMER_SOURCE')
    terminal = obj('TERMINAL_READ_CLAIM.json')
    need(terminal['root_GO'] is expected_GO and terminal['READ_GO'] is expected_GO and terminal['GPU_GO'] is False
         and terminal['Model_GO'] is False and 'read_terminal.py' in files, 'PRE_REGISTERED_TERMINAL_CONSUMER')
    need(plan['container_launcher'] == claim['launcher']['path'] ==
         '/public/software/apps/apptainer/1.3.4/bin/apptainer' and claim['launcher']['CPU_rocm_flag'] is False
         and claim['launcher']['readonly_system_alias']['actual_source'] == '/usr/local/hyhal', 'QUALIFIED_CPU_FRONTEND')
    launch_tree=ast.parse((root/'launch_once.py').read_bytes())
    entry=next(n for n in launch_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    need([a.arg for a in entry.args.kwonlyargs]==['transport_start'] and len(entry.args.kw_defaults)==1
         and isinstance(entry.args.kw_defaults[0],ast.Constant) and entry.args.kw_defaults[0].value is None,'EXPLICIT_ROOT_CALLBACK_NO_DEFAULT')
    need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
         and n.func.value.id=='subprocess' and n.func.attr=='Popen' for n in ast.walk(launch_tree)),'NO_LOCAL_TRANSPORT_CONSTRUCTOR_OR_FALLBACK')
    need('glm-paired-readonly-connection.py' not in (root/'launch_once.py').read_text(encoding='utf-8') and 'accounting_contract.py' in files,'NEW_TRANSPORT_ACCOUNTING_SCOPE')
    total = sum(p['bytes'] for p in files.values()) + (root / 'SOURCE_FILES.json').stat().st_size
    need(total + sum(v['bytes'] for v in contract['files'].values()) <= 2 * 1024**2, 'BOUNDED_NEW_SOURCE_AND_DATA_PAYLOAD')
    return dict(source_members=len(files), source_bytes=total, framework_members=393,
                metadata_files=13, new_train_rows=72, old64_replayed=False, stdlib_only=True, Torch_imported=False,
                target_or_GPU_called=False, fixture_tests=0)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',action='store_true')
    options=parser.parse_args()
    print(json.dumps(verify(prepared=options.prepared), sort_keys=True))

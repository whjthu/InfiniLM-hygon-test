"""Check this complete new bundle, inherited stager delta and exact native recipe."""
from pathlib import Path,PurePosixPath
import ast,base64,gzip,hashlib,json,subprocess,sys
D=Path(__file__).resolve().parent;B=D.parent;F=B/'hygon-training-integration';S=D/'publication';prefix='code/bw1000/'+D.name;source=S/prefix
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
def sha(raw):return hashlib.sha256(raw).hexdigest()
fm=json.loads((source/'FRAMEWORK_FILES.json').read_bytes());manifest=json.loads((source/'SOURCE_FILES.json').read_bytes());assert len(fm['files'])==350
archive=(source/'FRAMEWORK_BUNDLE.json.gz').read_bytes();raw=gzip.decompress(archive);assert len(raw)==fm['bundle_uncompressed_bytes'] and sha(raw)==fm['bundle_uncompressed_SHA256']
bundle=json.loads(raw);assert set(bundle)==set(fm['files']);py_count=0
for name,pin in fm['files'].items():
 raw=base64.b64decode(bundle[name],validate=True);assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
 if name.endswith('.py'):ast.parse(raw);py_count+=1
for name,pin in manifest['files'].items():
 raw=(source/name).read_bytes();assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
 if name.endswith('.py'):ast.parse(raw)
old=B/'training-internvl-A100-first-updates-v1'
for name in ('owned_processes.py','deadline_contract.py','capture_first_updates.py'):
 assert (D/name).read_bytes()==(old/name).read_bytes()==(B/'training-qwen-A100-first-updates-v2'/name).read_bytes()
old_stager=(old/'receive_reference.py').read_text(encoding='utf8')
expected=old_stager.replace('hygon-internvl-A100-first-updates-v1','hygon-llava-A100-first-updates-v1').replace('==349','==350').replace('framework_files=349','framework_files=350')
assert (D/'receive_reference.py').read_text(encoding='utf8')==expected
# No new stager failure fixtures: underlying logic is byte-identical except
# namespace/count constants, and the complete new350-file bundle was verified.
sys.path.insert(0,str(F))
from tools.hygon_vlm_recipe import build_launch,validate_diagnostic_outputs
from vlm.runtime.native_recipe_contract import native_recipe_spec
W=PurePosixPath('/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1');framework=W/'framework';src=W/'src'
original=(source/'original-launch.json').read_bytes();native=json.loads(original);model=json.loads((source/'MODEL_FILES.json').read_bytes())
launch=build_launch('T04.sft',original,PurePosixPath('/home/whj/nvidia-baseline/env-vlm/bin/swift'),PurePosixPath(model['model_root']).parent,PurePosixPath('/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl'),PurePosixPath('/home/whj/nvidia-baseline'),W/'train',29903,diagnostic_plugin=src/'capture_first_updates.py',diagnostic_callback='native_first_updates',diagnostic_callback_position='before')
loader='/home/whj/nvidia-baseline/llava_onevision.py';assert launch['command'].count(loader)==1
launch['command'][launch['command'].index(loader)]=str(framework/'vlm/plugins/llava_onevision.py')
launch['required_assets']['plugins']=[str(framework/'vlm/plugins/llava_onevision.py') if p==loader else p for p in launch['required_assets']['plugins']]
binding=json.loads((source/'LLAVA_NATIVE_BINDING.json').read_bytes());assert native_recipe_spec(launch)==binding['spec']
argv=launch['command'];reverted=argv.copy();reverted.remove(str(src/'capture_first_updates.py'));reverted.remove('native_first_updates')
reverted[reverted.index('--output_dir')+1]=native['command'][native['command'].index('--output_dir')+1]
reverted[reverted.index(str(framework/'vlm/plugins/llava_onevision.py'))]='/mnt/data/nvidia-baseline-20261010/cuda_source_r3/vlm/plugins/llava_onevision.py'
assert reverted==native['command']
assert '--template' not in argv and '--use_logits_to_keep' not in argv and argv[argv.index('--attn_impl')+1]=='eager'
assert launch['environment_overrides']['NPROC_PER_NODE']=='8' and launch['environment_overrides']['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
validate_diagnostic_outputs(launch,dict(FIRST_UPDATE_CAPTURE=str(W/'capture')))
from importlib.util import module_from_spec,spec_from_file_location
spec=spec_from_file_location('published_OneVision_deadline',D/'deadline_contract.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
assert mod.schedule(1000,1001)==dict(epoch=1000,native_deadline=1820,supervisor_stop=1850,deadline=1900)
task=json.loads((S/('coordination/tasks/'+D.name+'.json')).read_bytes())
assert task['GPU_GO'] and task['actual_GPU_launches']==0 and task['max_native_GPU_launches']==1 and all(n in manifest['files'] or n=='SOURCE_FILES.json' for n in task['remote_files'])
for name,pin in task['source_SHA256'].items():assert sha((source/name).read_bytes())==pin
binding_description=binding['capture_budget_description'];assert binding_description['within_recorder_limits'] and binding_description['input_truncation_allowed'] is False and binding_description['output_truncation_allowed'] is False
assert binding_description['per_rank_total_upper_bytes']==4807148264 and binding_description['per_rank_total_limit']==8*1024**3 and binding_description['per_artifact_limit']==2*1024**3-1
v=dict(passed=True,framework_commit=fm['framework_commit'],framework_files=350,framework_Python_ASTs=py_count,package_manifest_files=len(manifest['files']),complete_bundle_verified=True,
 inherited_stager_logic_unchanged_except_namespace_and_349_to350=True,qualified_owned_deadline_helpers_identical=True,exact_native_original_argv_preserved_except_output_plugin_and_observer=True,
 original_shared_loader_SHA256=fm['files']['vlm/plugins/llava_onevision.py']['SHA256'],recipe_case='T04.sft',GPU_cards_in_recipe=8,native_updates=25,captured_updates=2,master_port=29903,
 no_local_fixture_replays=True,actual_remote_install=False,actual_GPU_launches=0,actual_model_loaded=False,actual_native_training_qualified=False,scope='Local canonical bundle/source and exact launch binding verification;not actual LinuxHOST install or loaded model/device qualification.')
with (D/'local-package-checks.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))

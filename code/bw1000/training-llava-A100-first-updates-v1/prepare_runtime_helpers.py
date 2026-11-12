"""Reuse owned/deadline machinery; bind OneVision and its actual shared loader."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;B=D.parent;old=B/'training-internvl-A100-first-updates-v1'
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
binding=(G/'coordination/training/llava-native-first-updates-binding-v1.json').read_bytes()
for name in ('owned_processes.py','deadline_contract.py','capture_first_updates.py'):
 raw=(old/name).read_bytes();ast.parse(raw);(D/name).write_bytes(raw)
s=(old/'run_reference_once.py').read_text(encoding='utf8')
for a,b in [('InternVL','OneVision'),('INTERNVL_NATIVE_BINDING.json','LLAVA_NATIVE_BINDING.json'),('T03.sft','T04.sft'),('29901','29903'),('349','350'),('b715b6b52f28b24738b41388ca3d68a0e15e301f','b641edb0ad0f5104f18ba4ab45a247fbe1f7c495'),('7fefdc16f5519a998a94fdef703c2dd326b3c15e2e31d5a3735f6c4558269c67','849e72c9dde3d3e4a7bd40a4c1ebc700854af944fa6fbf86317f8447f55f7c9b'),('2ff3b19fc433f38b2b9053787f1b458e70ada2457e1d1092f5beab3ae769a076',hashlib.sha256(binding).hexdigest())]:
 assert a in s,a;s=s.replace(a,b)
s=s.replace('tests=20,framework_files_verified=350','tests=24,framework_files_verified=350').replace('source_CPU_tests_reused=20','source_CPU_tests_reused=24').replace('fresh_metadata_files_SHA_verified=13','fresh_metadata_files_SHA_verified=14')
begin=s.index(" lifecycle=native_binding['native_checkpointing_lifecycle']")
end=s.index("\n assert file_sha(Path('/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py'))",begin)
s=s[:begin]+''' assert native_binding['spec']['training']['gradient_checkpointing'] is False
 assert native_binding['resolved_template_type']=='llava_onevision1_5'
 assert native_binding['capture_budget_conditions']==dict(case='T04.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False,native_logits_to_keep=False)
 assert native_binding['capture_budget_description']['per_rank_total_upper_bytes']==4807148264 and native_binding['capture_budget_description']['per_rank_total_limit']==8*1024**3
 assert native_binding['capture_budget_description']['within_recorder_limits'] and not native_binding['actual_GPU_serialized_artifact_bytes_qualified']
 assert native_binding['runtime_guard_framework_commit']==fm['framework_commit']
 # Source-only prerequisites do not assert that values have already been loaded.
 assert not native_binding['actual_weight_loading_qualified'] and not native_binding['materialized_RoPE_repair_qualified']
 assert file_sha(FRAMEWORK/'vlm/plugins/llava_onevision.py')==file_sha(Path('/mnt/data/nvidia-baseline-20261010/cuda_source_r3/vlm/plugins/llava_onevision.py'))
''' + s[end:]
needle=" argv=launch['command'];overrides=launch['environment_overrides'];"
assert needle in s
s=s.replace(needle,""" # Rebind the original compatibility plugin to its identical bytes in this
 # complete common framework; keep the original baseline callback external.
 loader='/home/whj/nvidia-baseline/llava_onevision.py'
 assert launch['command'].count(loader)==1
 launch['command'][launch['command'].index(loader)]=str(FRAMEWORK/'vlm/plugins/llava_onevision.py')
 launch['required_assets']['plugins']=[str(FRAMEWORK/'vlm/plugins/llava_onevision.py') if p==loader else p for p in launch['required_assets']['plugins']]
 assert '--template' not in launch['command'] and '--use_logits_to_keep' not in launch['command']
"""+needle)
assert 'native_checkpointing_lifecycle' not in s and 'INTERNVL' not in s
ast.parse(s);(D/'run_reference_once.py').write_bytes(s.encode())
for name in ('receive_reference.py','preflight_once.py','launch_reference.py'):
 s=(old/name).read_text(encoding='utf8').replace('training-internvl-A100-first-updates-v1',D.name).replace('hygon-internvl-A100-first-updates-v1','hygon-llava-A100-first-updates-v1').replace('InternVL','OneVision').replace('29901','29903').replace('==349','==350').replace('framework_files=349','framework_files=350').replace('model_files=17','model_files=18')
 if name=='preflight_once.py':
  needle=" phase='original-inputs'";assert needle in s
  s=s.replace(needle," phase='original-shared-loader'\n pin=BOUND['PRIOR_QUALIFICATIONS.json']['original_loader'];p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']\n"+needle)
 ast.parse(s);(D/name).write_bytes(s.encode())
print(json.dumps(dict(owned_deadline_and_recorder_unchanged=True,framework_files=350,master_port=29903,actual_GPU_launches=0)))

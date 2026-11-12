"""Reuse the actual qualified GPU supervisor, changing only model bindings."""
from pathlib import Path
import ast,hashlib
D=Path(__file__).resolve().parent;old=D.parent/'training-qwen-A100-first-updates-v2'
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
for name in ('owned_processes.py','deadline_contract.py','capture_first_updates.py'):
 raw=(old/name).read_bytes();ast.parse(raw);(D/name).write_bytes(raw)
s=(old/'run_reference_once.py').read_text(encoding='utf8')
for a,b in [('Qwen','InternVL'),('QWEN_NATIVE_BINDING.json','INTERNVL_NATIVE_BINDING.json'),('T01.sft','T03.sft'),('29895','29901'),('342','349'),('e306d8e2e8f8d2bfd21421f1943e6102c673d02e','b715b6b52f28b24738b41388ca3d68a0e15e301f'),('28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400','7fefdc16f5519a998a94fdef703c2dd326b3c15e2e31d5a3735f6c4558269c67')]:
 assert a in s,a;s=s.replace(a,b)
binding=(G/'coordination/training/internvl-native-first-updates-binding-v1.json').read_bytes()
assert '78ca35c6bf9a93790c8f4a09bb8e1c9b1bb1779d14e0e8ffbf00eaa8d6fc3d2c' in s
s=s.replace('78ca35c6bf9a93790c8f4a09bb8e1c9b1bb1779d14e0e8ffbf00eaa8d6fc3d2c',hashlib.sha256(binding).hexdigest())
s=s.replace("dict(tests=13,framework_files_verified=349", "dict(tests=20,framework_files_verified=349")
s=s.replace("source_CPU_tests_reused=13", "source_CPU_tests_reused=20").replace('fresh_metadata_files_SHA_verified=10','fresh_metadata_files_SHA_verified=13')
needle="assert native_binding['spec']['training']['gradient_checkpointing'] is True"
assert needle in s
s=s.replace(needle,needle+"\n assert native_binding['capture_budget_conditions']==dict(case='T03.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False)\n assert native_binding['capture_budget_plan']['per_rank_total_upper_bytes']==3933355752 and native_binding['capture_budget_plan']['per_rank_total_limit']==8*1024**3\n assert not native_binding['actual_GPU_serialized_artifact_bytes_qualified']")
ast.parse(s);(D/'run_reference_once.py').write_bytes(s.encode())
print('Prepared actual qualified owned-GPU helpers for original InternVL bindings')

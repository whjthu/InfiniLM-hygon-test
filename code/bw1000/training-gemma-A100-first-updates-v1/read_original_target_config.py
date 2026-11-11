from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-gemma-A100-first-updates-v1';assert not (D/'original-target-regex-read-intent.json').exists();(D/'original-target-regex-read-intent.json').write_bytes(b'{"scope":"Read only native LoRA target configuration from the original failed run; no model replay"}\n')
remote=r'''from pathlib import Path
import ast,hashlib,json,re
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-A100-first-updates-v1');p=W/'native.private.log';patterns=[]
with p.open() as f:
 for line in f:
  if 'lora_config:' not in line:continue
  match=re.search(r"target_modules=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",line)
  if match:
   value=ast.literal_eval(match.group(1));assert isinstance(value,str) and len(value)<2000;patterns.append(value)
expected=json.loads((W/'src/GEMMA_NATIVE_BINDING.json').read_bytes());names=sorted({re.sub(r'\.lora_[AB]\.default\.weight$','',x['name'].removeprefix('base_model.model.')) for x in expected['trainable_manifest']})
print(json.dumps(dict(original_native_target_patterns=sorted(set(patterns)),bound_meta_target_pattern=expected['target_modules'],expected_target_names_count=len(names),each_actual_pattern_matches_all_bound_targets=[all(re.fullmatch(p,n) for n in names) for p in sorted(set(patterns))],native_log_SHA256=hashlib.sha256((W/'native.private.log').read_bytes()).hexdigest(),qualification='Original emitted configuration and its matches on the bound238 target modules only. Full native trainable values were not captured.')))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');a=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(a,input=remote.encode(),capture_output=True,timeout=25);(D/'original-target-regex-stdout.private.json').write_bytes(r.stdout);(D/'original-target-regex-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(D/'original-target-regex-config.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))

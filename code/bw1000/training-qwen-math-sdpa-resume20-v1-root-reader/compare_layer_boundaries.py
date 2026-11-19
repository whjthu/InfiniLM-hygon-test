import hashlib,json,runpy
from pathlib import Path
parent=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
control=parent/'training-qwen-continuous-branches21-hip-v2'
manifest=json.loads((control/'src/FRAMEWORK.json').read_bytes())
name='vlm/runtime/qwen_loss_boundary_trace.py'
path=control/'framework'/name
raw=path.read_bytes()
assert len(raw)==manifest['files'][name]['bytes'] and hashlib.sha256(raw).hexdigest()==manifest['files'][name]['SHA256']
fn=runpy.run_path(str(path))
roots=(parent/'training-qwen-math-sdpa-recovery-pair-v1'/'continuous/train/continuous-tail',parent/'training-qwen-math-sdpa-resume20-v1'/'resumed/train/resume-witness')
out={}
for rank in (1,4):
 pin=[]
 docs=[]
 for root in roots:
  f=root/f'rank-{rank:02d}'/'loss-21.json'
  raw=f.read_bytes(); assert 0<len(raw)<=65536
  pin.append(dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()))
  docs.append(json.loads(raw))
 pairs=fn['compare_loss_boundaries'](*docs)
 first=fn['first_suffix_gradient_difference'](pairs)
 first_forward=next((dict(module=name,boundary=key) for name, row in pairs.items() for key in ('input','output') if not row[key]['logical_sha_equal']),None)
 internal={name:{key:v for key,v in row.items() if key in ('input','output','grad_input','grad_output','recompute_input','recompute_output')}
           for name,row in pairs.items() if name in ('decoder_32.self_attn','decoder_31.self_attn','decoder_30.self_attn','lm_head','final_norm') or first and name.startswith(first['module']+'.')}
 out[str(rank)]=dict(micro=docs[0]['ordinal'],pins=pin,compared_modules=len(pairs),first_forward=first_forward,first_suffix_gradient=first,
                      changed_top_level=[name for name,row in pairs.items() if '.' not in name and any(v['logical_sha_equal'] is False for v in row.values())],
                      internal=internal)
print(json.dumps(out))

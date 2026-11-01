"""Fixed CPU618 evidence and same-project read-only alias; no GPU imports."""
import pathlib,json,hashlib
WRAPPER_SHA='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d'
CPU_SHA='873fdaa88a67d2a47b0d0cf87acced50fcbccac938cad58c64c9a01726dd2766'
MF_SHA='6e3e2251ab313e374f387ec95aa5985866e638b5ea2f857da42463e498121cea'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify_namespace(root,directory):
 root=pathlib.Path(root);directory=pathlib.Path(directory)
 assert sha(directory/'resource_wave_host_alias_outer.sh')==WRAPPER_SHA
 original=root/'incoming/resource_wave_local_r3097_outer.sh'
 assert sha(original)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
 needle='--bind "$TASK_ROOT:/refactor"';old=original.read_text()
 assert old.count(needle)==1 and (directory/'resource_wave_host_alias_outer.sh').read_text()==old.replace(needle,needle+' --bind "$TASK_ROOT:$TASK_ROOT:ro"')
 cp=root/'candidates/container-host-alias-r295/CPU-summary.json';assert sha(cp)==CPU_SHA
 c=json.loads(cp.read_bytes())
 assert c['completed']and c['passed']and c['actual_CPU']=='910407.618'and c['CPU_only']
 assert all(c[k]==0 for k in ['GPU','Model','HTTP','Native_dlopen','weights'])
 assert c['source_manifest_SHA256']==MF_SHA and c['new_wrapper_SHA256']==WRAPPER_SHA
 phases={x['phase']:x for x in c['phases']};assert set(phases)=={'control','candidate'}and all(x['rc']==0 for x in phases.values())
 proofs={}
 for phase in ['control','candidate']:
  p=cp.with_name('CPU-'+phase+'.json');assert sha(p)==phases[phase]['receipt_SHA256'];v=json.loads(p.read_bytes())
  assert v['completed']and v['passed']and v['source_manifest_SHA256']==MF_SHA and v['actual_step']=='910407.618'
  assert all(v[k]==0 for k in ['GPU','Model','HTTP','Native_dlopen','weights_opened'])
  proofs[phase]=v
 assert len(proofs['control']['path_cases'])==2 and all(x['expected_namespace_failure']for x in proofs['control']['path_cases'])
 v=proofs['candidate'];assert len(v['path_cases'])==2 and v['GLM_actual_static_chain']['ancestry_files_checked']==2181
 assert v['Deep_actual_binding']['objects']==48 and v['Deep_actual_binding']['bindings']==5
 assert v['Deep_CPU_preflight_counts_actual']=={'baseline_files_SHA256':68,'overlay_files_SHA256':9}
 assert v['Deep_original_preweight_reached_first_Torch_import']and v['expected_blocked_imports']==['torch']
 return {'wrapper_SHA256':WRAPPER_SHA,'CPU618_summary_SHA256':CPU_SHA,'scope':'Static dependency chains and Deep pre-first-Torch checks only; full Model/ABI remains unqualified'}
def terminal_failure(row):
 assert len(row)>=4
 return row[2].split()[0].rstrip('+')!='COMPLETED' or row[3]!='0:0'

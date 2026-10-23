import pathlib,json,hashlib,sys,subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
p=r/'results'/f'deep-causal-teacher-r7238-{tag}.json';q=json.loads(p.read_text());assert q['completed']and q['engine_worker_shutdown_completed']and q['causal_diagnostic_collected']and q['no_serving_TPS']and not q['performance_adopted']
assert q['library_sha256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
launch=json.loads((r/'results/Deep-causal-teacher-r7238-launch.json').read_text())
for n,h in launch['sources'].items():assert sha(r/'incoming'/n)==h
for key in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:
 a=q[key];assert a['verified']and a['width']==4;owners=a['actual_owner_report']['owners'];assert sorted(o['device']for o in owners)==list(range(8));masks=[]
 for o in owners:
  assert o['changes']==1;mask=list(map(int,o['verified_mask'].split(',')));assert len(mask)==4;masks+=mask
 assert len(masks)==len(set(masks))==32
arms=q['causal_teacher_r7238'];assert len(arms)==3
assert [(c['label'],c['prefill_B'],c['decode_M'],c['projection_tile'],c['C4_tile'],c['prefill_mode'],c['decode_mode'])for c in arms]==[(n,32,32,p,c,[0,0,0],[0,0,0])for n,p,c in [('A-projection16-C4-16',16,16),('P-projection0-C4-16',0,16),('C-projection16-C4-0',16,0)]]
assert all(s['ID_exact_frozen']for s in arms[0]['steps'])
frozen=next(c for c in json.loads((r/'results/deep-attention-r3509-910406.102.json').read_text())['attention_F64_cases']if c['phase']=='natural-l0-r1')['generated_ids'][0]
summary=[]
for c in arms:
 assert len(c['steps'])==36
 for s,z in enumerate(c['steps']):
  assert z['step']==s and z['teacher_input_from_frozen']and z['frozen_ID']==frozen[s]and z['ID_exact_frozen']==(z['ID']==frozen[s]);assert z['top2_IDs'][0]==z['ID']and z['margin']==z['top2_logits'][0]-z['top2_logits'][1]
  if 'raw_file_remote_only'in z:
   b=pathlib.Path(z['raw_file_remote_only'].replace('/refactor',str(r))).read_bytes();assert hashlib.sha256(b).hexdigest()==z['raw_SHA256']==z['logits_SHA256'];v=(np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32);assert np.isfinite(v).all();idx=np.lexsort((np.arange(len(v)),-v))[:2];assert idx.tolist()==z['top2_IDs']and v[idx].tolist()==z['top2_logits']
 summary.append({'label':c['label'],'frozen_mismatch_steps':[z['step']for z in c['steps']if not z['ID_exact_frozen']],'step35':c['steps'][35]})
for idx,name in [(1,'P'),(2,'C')]:
 for s,z in enumerate(arms[idx]['steps']):assert z['argmax_equal_A']==(z['ID']==arms[0]['steps'][s]['ID'])
out={'completed':True,'source_report_SHA256':sha(p),'normal8owners':True,'summary':summary,'performance_adopted':False,'scope':'Only diagnostic frozen-prefix causal comparison, not final ordinary serving qualification. No bitwise quality gate.'}
p=r/'results'/f'deep-causal-teacher-r7238-{tag}-audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))

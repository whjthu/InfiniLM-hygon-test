"""Finite causal diagnostic on frozen historical prefixes; never serving TPS."""
import hashlib,json,ctypes,time
import numpy as np

def validate_natural_inputs(natural,reference):
 current=[c['input_ids']for c in natural['cases']]
 historical=[c['input_ids']for c in reference['natural_input_provenance']['cases']]
 assert current==historical, 'fresh_natural_prompts_differ_from_frozen_historical_source'
 assert len(current)==2 and len(current[0])==512, 'natural0_must_be_exactly512_for_tail_dispatch'
 return current

def run(g):
 r,tag,lib,ctx,cfg,forward,check,report,path=[g[k]for k in ['r','tag','lib','ctx','cfg','forward','check','report','path']]
 natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text())
 source=r/'results/deep-attention-r3509-910406.102.json'
 assert hashlib.sha256(source.read_bytes()).hexdigest()=='91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9'
 reference=json.loads(source.read_text())
 validate_natural_inputs(natural,reference)
 frozen=next(c for c in reference['attention_F64_cases']if c['phase']=='natural-l0-r1')['generated_ids']
 assert len(frozen)==8 and all(len(x)>=36 for x in frozen)
 report.update(causal_teacher_r7238=[],performance_adopted=False,serving_whole_candidate_qualified=False,
  scope='Three single-factor tile arms, identical B32 prefill and M32 frozen prefix through index35. Diagnostic only. Original 0e051 DSO and SDK unchanged. No F64GPU or new C++ hooks. Each arm resets, no cache row reindex/recycle. Intermediate bits diagnostic; final ordinary qualification not attempted.')
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 def policy(mode,projection,c4):
  g['layout_mode'],g['HC_mode'],g['tail_SUM_mode']=mode
  check(lib.modern_deep_layout_policy_r7052(mode[0]));check(lib.deep_hc_canonical_policy_r7130(mode[1]));check(lib.deep_moe_tail_set_r7211(mode[2]))
  g['projection_tile_mode']=projection;g['c4_tile_mode']=c4;g['precision_mode']=5
  lib.modern_deep_precision_policy_r3443.argtypes=[ctypes.c_int];check(lib.modern_deep_precision_policy_r3443(5))
 def invoke(inputs,rows,past):
  widths=list(map(len,inputs));positions=sum([list(range(p,p+w))for p,w in zip(past,widths)],[])
  code,bits,ids=forward(sum(inputs,[]),positions,rows,past,[p+w for p,w in zip(past,widths)],[0]+np.cumsum(widths).tolist());check(code)
  v=(bits.astype(np.uint32)<<16).view(np.float32)
  assert np.isfinite(v).all() and v.argmax(-1).tolist()==ids.tolist()
  free=[g['torch'].cuda.mem_get_info(d)[0]for d in range(8)];assert min(free)>=6*1024**3
  return bits,v,ids.tolist()
 arms=[('A-projection16-C4-16',32,32,16,16,(0,0,0),(0,0,0)),
       ('P-projection0-C4-16',32,32,0,16,(0,0,0),(0,0,0)),
       ('C-projection16-C4-0',32,32,16,0,(0,0,0),(0,0,0))]
 baseline={};matched={};first_single={}
 for label,B,M,projection,c4,prefill_mode,decode_mode in arms:
  check(lib.modern_native_engine_reset(ctx));policy(prefill_mode,0,0)
  row={'label':label,'prefill_B':B,'decode_M':M,'prefill_mode':prefill_mode,'decode_mode':decode_mode,'projection_tile':projection,'C4_tile':c4,'steps':[]}
  report['current_causal_arm_r7238']=row;save()
  prompts=[natural['cases'][i%2]['input_ids']for i in range(B)]
  for step in range(36):
   if step:policy(decode_mode,projection,c4)
   active=list(range(B))if step==0 or M==32 else[0]
   current=None
   for a in range(0,len(active),4 if step==0 else M):
    rows=active[a:a+(4 if step==0 else M)]
    inputs=[prompts[i]if step==0 else[frozen[i%8][step-1]]for i in rows]
    past=[0 if step==0 else len(prompts[i])+step-1 for i in rows]
    bits,values,ids=invoke(inputs,rows,past)
    if 0 in rows:
     j=rows.index(0);current=(bits[j].copy(),values[j].copy(),ids[j])
   assert current is not None
   bits,v,token=current
   # Stable top2 with lowest-index tie break, matching first-argmax semantics.
   top=np.lexsort((np.arange(len(v)),-v))[:2];assert int(top[0])==token
   if label.startswith('A-'):baseline[step]=bits.copy()
   if label.startswith('P-'):matched[step]=bits.copy()
   if label.startswith('C-'):first_single[step]=bits.copy()
   z={'step':step,'teacher_input_from_frozen':True,'ID':token,'frozen_ID':frozen[0][step],'ID_exact_frozen':token==frozen[0][step],
      'top2_IDs':top.tolist(),'top2_logits':v[top].tolist(),'margin':float(v[top[0]]-v[top[1]]),'logits_SHA256':hashlib.sha256(bits.tobytes()).hexdigest(),
      'bits_equal_A_diagnostic':np.array_equal(bits,baseline[step]),'argmax_equal_A':token==int(((baseline[step].astype(np.uint32)<<16).view(np.float32)).argmax())}
   for key,bank in [('B',matched),('C',first_single)]:
    if step in bank:z['argmax_equal_'+key]=token==int(((bank[step].astype(np.uint32)<<16).view(np.float32)).argmax());z['bits_equal_'+key+'_diagnostic']=np.array_equal(bits,bank[step])
   if step in [0,25,35]:
    raw=r/'results'/f'deep-causal-teacher-r7238-{tag}-{label}-s{step}.bin';assert not raw.exists();raw.write_bytes(bits.tobytes());z['raw_file_remote_only']=str(raw);z['raw_SHA256']=hashlib.sha256(raw.read_bytes()).hexdigest()
   row['steps'].append(z);save()
  if label.startswith('A-'):assert all(z['ID_exact_frozen']for z in row['steps']), 'A_control_must_reproduce_frozen36_before_attribution'
  report['causal_teacher_r7238'].append(row);report.pop('current_causal_arm_r7238');save()
 policy((0,0,0),0,0);check(lib.modern_native_engine_reset(ctx))
 report['causal_diagnostic_collected']=len(report['causal_teacher_r7238'])==3;report['no_serving_TPS']=True;save()

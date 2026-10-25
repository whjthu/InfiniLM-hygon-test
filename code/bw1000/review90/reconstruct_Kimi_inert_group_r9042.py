import pathlib,json,hashlib,array,struct,statistics
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');raw=r/'results/Kimi-layer0-raw-r9042-910375.693';reports=[json.loads((r/f'results/Kimi-layer0-real-r9042-910375.693-node{node}.json').read_text())for node in range(2)];q={'handle':'910375.693','CPU_only':True,'new_GPU':False,'new_capture':False,'status':'INERT_SELECTOR_HARNESS_NEGATIVE','original_Slurm':'FAILED1','original_watch_audit':'FAILED_PRESERVED','group_candidate_measured':False,'group_invocations':'0inferredfromactualdriverwritesoldOPROJenv/newDSOreadsROWGROUPenvandnoexpectedgroup-pathtracewithactiveKDAtrace; noindependentnativecounterinstrumented','workers':[],'journal_audit':{'journals':0,'records':0,'nonfinite':0,'required_group_point_records':0},'boundaries':[]}
def records(arm,rank,call):return{z['point']:z for z in[json.loads(x)for x in(raw/arm/f'stage{rank//8}-rank{rank%8}-layer0-kda-call{call}.jsonl').read_text().splitlines()]}
def data(arm,z):return(raw/arm/z['file']).read_bytes()
def diff(a,b):
 aa=array.array('H',a);bb=array.array('H',b);dec=lambda x:struct.unpack('<f',struct.pack('<I',int(x)<<16))[0];assert len(aa)==len(bb);return{'different_elements':sum(x!=y for x,y in zip(aa,bb)),'max_abs':max(abs(dec(x)-dec(y))for x,y in zip(aa,bb)),'A_SHA256':hashlib.sha256(a).hexdigest(),'B_SHA256':hashlib.sha256(b).hexdigest()}
for d in reports:
 for w in d['workers']:
  n=sum(w[a]['native_calls']for a in ['A','B','C'])+26*len(w['timing_trials']);assert n==345 and w['normal_close']and'error'not in w;assert len(w['timing_trials'])==10;q['workers'].append({'rank':w['rank'],'normal_close':True,'native_calls':n,'trials':10,'actual_timed_groups':'bothlabelsareoriginalnativebaseline','unique_output_SHA256_all10':len({z['output_SHA256']for z in w['timing_trials']})})
for arm,n in [('A',33),('B',26),('C',26)]:
 paths=list((raw/arm).glob('*.jsonl'));assert len(paths)==16*n
 for p in paths:
  rows=[json.loads(x)for x in p.read_text().splitlines()];q['journal_audit']['journals']+=1;q['journal_audit']['records']+=len(rows);q['journal_audit']['nonfinite']+=sum(z['nonfinite']for z in rows);q['journal_audit']['required_group_point_records']+=sum(z['point']in['r9042_o_proj_local','r9042_o_proj_reduced']for z in rows)
for rank in range(16):
 aa=[records('A',rank,25),records('A',rank,26)];bb=records('B',rank,25);cc=records('C',rank,25)
 for j,pos in enumerate([536,537]):
  for point in ['kda_output','gated','output']:
   a=data('A',aa[j][point]);n=len(a);b=data('B',bb[point])[j*n:(j+1)*n];c=data('C',cc[point])[j*n:(j+1)*n]
   for name,x,y in [('A_vs_B',a,b),('A_vs_C',a,c),('B_vs_C',b,c)]:q['boundaries'].append({'rank':rank,'position':pos,'point':point,'comparison':name,**diff(x,y)})
q['all16_normal_close']=all(z['normal_close']for z in q['workers']);q['native_calls_total']=sum(z['native_calls']for z in q['workers']);q['canary_readonly']='everyactualCAPIcallassertionscompletedbefore16normalclose; nooriginalworkererrors';q['timing_interpretation']='10trialsarebaseline-only; no groupoverhead/candidategainclaim';p=r/'results/Kimi-inert-group-CPU-reconstruction-r9042.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))

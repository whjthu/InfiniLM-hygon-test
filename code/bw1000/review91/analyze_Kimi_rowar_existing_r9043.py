import pathlib,json,hashlib,array,struct
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');raw=r/'results/Kimi-layer0-raw-r9043-910375.701';q={'CPU_only':True,'GPU':False,'new_capture':False,'original_handle':'910375.701','original_Slurm':'FAILED1_preserved','status':'EXISTING_RAW_NUMERIC_DIAGNOSTIC_ONLY_PERFORMANCE_UNKNOWN','boundaries':[],'journal_counts':{},'nonfinite':0,'warm_counter_deltas':[]}
def records(arm,rank,call):return{z['point']:z for z in[json.loads(x)for x in(raw/arm/f'stage{rank//8}-rank{rank%8}-layer0-kda-call{call}.jsonl').read_text().splitlines()]}
def data(arm,z):return(raw/arm/z['file']).read_bytes()
def diff(a,b):
 aa=array.array('H',a);bb=array.array('H',b);assert len(aa)==len(bb);dec=lambda x:struct.unpack('<f',struct.pack('<I',int(x)<<16))[0];return{'different_elements':sum(x!=y for x,y in zip(aa,bb)),'max_abs':max(abs(dec(x)-dec(y))for x,y in zip(aa,bb)),'A_SHA256':hashlib.sha256(a).hexdigest(),'B_SHA256':hashlib.sha256(b).hexdigest()}
for arm,n in [('W_C',26),('W_D',26),('A',33),('B',26),('C',26),('D',26)]:
 files=list((raw/arm).glob('*.jsonl'));q['journal_counts'][arm]={'actual':len(files),'expected':n*16,'complete':len(files)==n*16}
 for p in files:q['nonfinite']+=sum(z['nonfinite']for z in[json.loads(x)for x in p.read_text().splitlines()])
for rank in range(16):
 aa=[records('A',rank,25),records('A',rank,26)];others={a:records(a,rank,25)for a in ['B','C','D']}
 for j,pos in enumerate([536,537]):
  for point in ['kda_output','gated','r9042_o_proj_local','r9042_o_proj_reduced','output']:
   a=data('A',aa[j][point]);n=len(a);values={'A':a,**{arm:data(arm,rr[point])[j*n:(j+1)*n]for arm,rr in others.items()}}
   for left,right in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]:q['boundaries'].append({'rank':rank,'position':pos,'point':point,'comparison':left+'_vs_'+right,**diff(values[left],values[right])})
for node in range(2):
 p=r/f'results/Kimi-layer0-real-r9043-910375.701-node{node}.json';d=json.loads(p.read_text());q['warm_counter_deltas'].extend({'rank':w['rank'],**w['activation_counter_deltas']}for w in d['workers']);q.setdefault('workers',[]).extend({'rank':w['rank'],'normal_close':w.get('normal_close'),'normal_close_after_error':w.get('normal_close_after_error'),'error':w.get('error'),'A_calls':w.get('A',{}).get('native_calls'),'B_calls':w.get('B',{}).get('native_calls'),'C_calls':w.get('C',{}).get('native_calls'),'D_calls':w.get('D',{}).get('native_calls'),'timing_trials':len(w.get('timing_trials',[]))}for w in d['workers'])
q['summary']=[]
for comp in ['A_vs_B','A_vs_C','A_vs_D','B_vs_C','B_vs_D','C_vs_D']:
 for pos in [536,537]:
  for point in ['kda_output','gated','r9042_o_proj_local','r9042_o_proj_reduced','output']:
   v=[z for z in q['boundaries']if z['comparison']==comp and z['position']==pos and z['point']==point];q['summary'].append({'comparison':comp,'position':pos,'point':point,'equal_ranks':sum(z['different_elements']==0 for z in v),'different_min':min(z['different_elements']for z in v),'different_max':max(z['different_elements']for z in v),'max_abs':max(z['max_abs']for z in v)})
p=r/'results/Kimi-rowAR-existing-CPU-r9043.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))

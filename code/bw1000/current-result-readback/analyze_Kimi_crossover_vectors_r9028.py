"""Remote-only comparison of actually recorded boundaries, never token qualification."""
import pathlib,json,array,struct,math,hashlib,sys
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
run='TP16PP2-I1928669zh-8e319d252bb323d5';source=r/'results'/f'kimi-crossover-r9026-{run}-node2.json'
packet=json.loads(source.read_bytes());calls=packet.get('crossover_calls',[]);done={v['arm'] for v in packet.get('crossover_arms',[])}
assert 'A' in done,'ordinary A not complete'
ORDER={'attn_norm':0,'attn_output':1,'exit':2,'decoder':3,'final_norm':4}
def host_path(s):
 p=r/s[len('/refactor/'):] if s.startswith('/refactor/') else pathlib.Path(s)
 assert str(p).startswith(str(r/'results/Kimi-crossover-private-r9026-')),str(p)
 return p
def collect(arm):
 last=[c for c in calls if c['arm']==arm][-1];rows={}
 for owner in last['owners']:
  if owner['node'] not in (0,2):continue
  trace=owner['trace_after'];assert trace['finite'] and trace['arm']==arm
  for rec in trace['records']:
   selected=[(j,p) for j,p in enumerate(rec['positions']) if p in (536,537)]
   if rec['point'] not in ORDER or not selected:continue
   assert rec['shape']==[1,len(rec['positions']),7168]
   raw=host_path(rec['remote_owned_buffer']).read_bytes();assert len(raw)==rec['bytes']==len(rec['positions'])*7168*2
   for j,pos in selected:
    key=(rec['layer'],ORDER[rec['point']],rec['point'],pos,rec['node']);data=raw[j*14336:(j+1)*14336]
    if key in rows:assert rows[key][0]==data,('ambiguous repeated actualboundary',key)
    rows[key]=(data,rec)
 return rows
def floats(b):
 a=array.array('H');a.frombytes(b)
 if sys.byteorder!='little':a.byteswap()
 return [struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in a]
def compare(a,b):
 aa=array.array('H');aa.frombytes(a);bb=array.array('H');bb.frombytes(b)
 different=sum(x!=y for x,y in zip(aa,bb))
 if not different:return {'BF16_different_elements':0,'relative_RMS':0.,'maxabs':0.}
 x=floats(a);y=floats(b);assert all(math.isfinite(t) for t in x+y)
 mse=sum((v-w)**2 for v,w in zip(x,y));den=sum(v*v for v in x)
 return {'BF16_different_elements':different,'relative_RMS':math.sqrt(mse/max(den,1.e-30)),'maxabs':max(abs(v-w) for v,w in zip(x,y))}
base=collect('A');result={'scope':'Earliest RECORDED boundary differences; unrecorded operations may precede them. No intermediate bitexact acceptance gate or Model/serving speed claim. Raw remainsremote.','source':str(source),'source_snapshot_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'arms_complete':sorted(done),'A_recorded_boundaries':len(base),'comparisons':[]}
for arm in 'BCDE':
 if arm not in done:continue
 candidate=collect(arm);shared=sorted(set(base)&set(candidate));different=[];rowchecks=[]
 for key in shared:
  a,ra=base[key];b,rb=candidate[key];m=compare(a,b)
  if m['BF16_different_elements']:different.append({'layer':key[0],'point':key[2],'input_position':key[3],'node':key[4],**m})
  sa=ra.get('recurrent_rows_at_boundary');sb=rb.get('recurrent_rows_at_boundary')
  if sa is not None and sb is not None:
   rowchecks.append({'layer':key[0],'point':key[2],'input_position':key[3],'node':key[4],'ordinary_metadata_read_write':[ra['actual_init_state_indices'],ra['actual_final_state_indices']],'candidate_metadata_read_write':[rb['actual_init_state_indices'],rb['actual_final_state_indices']],'ordinary_state':sa,'candidate_state':sb})
 result['comparisons'].append({'arm':arm,'shared_recorded_boundaries':len(shared),'different_boundary_count':len(different),'first_recorded_differences':different[:24],'first_recorded_by_position':{str(pos):next((v for v in different if v['input_position']==pos),None) for pos in (536,537)},'recurrent_boundary_states':rowchecks,'arm_final_ID_observation':next(v for v in packet['crossover_arms'] if v['arm']==arm)})
out=r/'results/Kimi-crossover-vector-analysis-r9028.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({'written':str(out),'SHA256':hashlib.sha256(out.read_bytes()).hexdigest(),'comparisons':[{k:v for k,v in x.items() if k!='recurrent_boundary_states'} for x in result['comparisons']]}))

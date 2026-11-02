"""Read only original723 NPZ/JSON. Recompute numeric/carry; preserve missing second commit."""
from pathlib import Path
import argparse,hashlib,json,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;CPU=D.parent/'cpu-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def analyze(observation,rawdir):
 q=json.loads(Path(observation).read_bytes());r=q['report'];w=q['watch'];assert r['actual_step']=='910371.723'and not r['completed']and not r['passed']
 canonical=(json.dumps(r,indent=2)+'\n').encode();assert hashlib.sha256(canonical).hexdigest()==q['report_SHA256']=='a15d0ff6572430ff539bb8f78094148899e63292e15aaa075b4268d0cba93001'
 assert w['terminal'][2:] == ['FAILED','1:0','a02r3n16']and w['CPU_keeper']=='910407.781'
 assert w['post']['actual_step']=='910371.724'and not w['post']['kfd_pids']and not w['post']['matching_component_processes']
 assert len(r['owners'])==8 and all(o['normal_close']for o in r['owners'])
 assert len(r['hardware'])==8 and len({v['PCI']for v in r['hardware']})==8
 assert len(r['raw_archives'])==24 and len(r['steps'])==1 and r['steps'][0]['past']==8575 and r['steps'][0]['commit_checked']
 delta=json.loads((D/'SOURCE_DELTA.json').read_bytes());assert sha(CPU/'consumer.py')==delta['original_SHA256'];assert sha(D/'consumer_statebytes.py')==delta['corrected_SHA256']
 assert sha(CPU/'reference.py')=='fa49666e1684d642055b69f5bcab00efbe9dd2d064917bf28c99dad3d943ebae'
 consumer=module('consumer390fixed',D/'consumer_statebytes.py');oracle=module('oracle390',CPU/'reference.py')
 groups={8575:[dict(rank=i)for i in range(8)],8576:[dict(rank=i)for i in range(8)]};manifest=[]
 for a in r['raw_archives']:
  npz=Path(rawdir)/Path(a['npz']).name;meta=Path(rawdir)/Path(a['metadata']).name
  assert sha(npz)==a['npz_SHA256']and sha(meta)==a['metadata_SHA256'];v=json.loads(meta.read_bytes());packet=groups[a['past']][a['rank']]
  for k,x in v.items():
   if k.startswith('identity_'):
    if k in packet:assert packet[k]==x
    packet[k]=x
  with np.load(npz,allow_pickle=False)as arrays:
   for stage,fields in v['fields'].items():
    packet[stage]={}
    for k,f in fields.items():
     if f is None:packet[stage][int(k)]=None;continue
     raw=arrays[f['array']];assert raw.dtype==np.uint8 and raw.ndim==1 and hashlib.sha256(raw.tobytes()).hexdigest()==f['SHA256']
     packet[stage][int(k)]=dict(raw=raw,shape=f['shape'],dtype=f['dtype'])
  manifest.append(dict(rank=a['rank'],past=a['past'],phase=a['phase'],npz_SHA256=sha(npz),metadata_SHA256=sha(meta),remote_npz=a['npz'],remote_metadata=a['metadata']))
 first=consumer.validate_same_cache(groups[8575],oracle,committed=True)
 second=consumer.validate_same_cache(groups[8576],oracle,groups[8575],committed=False)
 assert first['passed']and second['passed']and second['nextstep_committed_cache_and_state_consumed']
 state=[]
 for rank in range(8):
  a=groups[8575][rank]['post_commit'][11];b=groups[8576][rank]['before'][8]
  assert consumer.equal(a,b)and a['shape']==[8,512]and a['dtype']==9 and len(a['raw'])==8192
  assert 'post_commit'not in groups[8576][rank]and groups[8576][rank]['identity_after_paged']['active']
  state.append(dict(rank=rank,shape=a['shape'],dtype_enum=a['dtype'],bytes=len(a['raw']),bytewise_equal=True,SHA256=hashlib.sha256(a['raw'].tobytes()).hexdigest()))
 metrics=[v for step in [first,second]for owner in step['metrics']for v in owner['scores'].values()]
 return dict(state='ORIGINAL_GPU_ENGINEERING_FAIL_OFFLINE_SAMECACHE_AND_FIRST_COMMIT_CARRY_PASS_SECOND_COMMIT_UNKNOWN',original_GPU='910371.723 FAILED1/59s',CPU_keeper='910407.781 FAILED1/62s',GPU_submission_count=1,original_report_SHA256=q['report_SHA256'],watch_SHA256=q['watch_SHA256'],original_GPU_manifest_SHA256=w['manifest_SHA256'],source_publication_commit=w['publication_commit'],Native_SHA256=r['current_Native_SHA256'],bridge_SHA256=r['bridge_SHA256'],driver_wall_s=r['driver_wall_s'],steps=[first,second],numeric_summary=dict(owner_steps=16,three_score_paths=48,max_CPU_FP8_NRMSE=max(v['CPU_FP8_NRMSE']for v in metrics),maxabs=max(v['maxabs']for v in metrics),minimum_cutoff_margin=min(v['cutoff_margin_512_513']for v in metrics),all_finite_legal_unique_TopK=True),state_carry=state,actual_second_commit=False,second_commit_output_and_state='UNKNOWN:notexecuted; no fabricatedpostcommit',first_commit=True,next8576_actual_original_prepare_consumed_committed_slot2143=True,normal_Runtime_owners=8,unique_PCI_count=8,post=w['post'],archives=manifest,offline_consumer_delta=delta,performance_trials=0,Model_gain='UNKNOWN',original684_numeric_qualification='Stillnotretrospectivelyqualified',GPU_replay=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--rawdir',required=True);p.add_argument('--output');a=p.parse_args();q=analyze(a.observation,a.rawdir);text=json.dumps(q,indent=2)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf-8',newline='\n')
 else:print(text)
if __name__=='__main__':main()

"""Independent CPU-only reference on original20 BF16 outputs and driver samples."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import argparse,json,hashlib,math,statistics
import numpy as np
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CASES={'past0':(0,512),'tile3072':(3071,512),'maxpast':(32255,512),'continuation1':(32767,1),'reset_reuse':(0,512)}
def words(shape,seed):
 a=np.arange(math.prod(shape),dtype=np.uint32);a=a*np.uint32(1664525)+np.uint32(seed);x=((a>>24).astype(np.int16)-128).astype(np.float32)/np.float32(256);return(x.view(np.uint32)>>16).astype(np.uint16).reshape(shape)
def f32(a):return(a.astype(np.uint32)<<16).view(np.float32)
def metric(a,b):
 a=a.astype(np.float64);b=b.astype(np.float64);delta=a-b;nrms=float(np.sqrt(np.mean(delta*delta)))/max(1e-20,float(np.sqrt(np.mean(b*b))));maxabs=float(np.abs(delta).max());return dict(NRMS=nrms,maxabs=maxabs,passed=nrms<=.03 and maxabs<=.05)
def run(owner_path,raw_dir):
 root=json.loads(owner_path.read_bytes());packet=json.loads(root['out'])if'out'in root else root;o=packet['owner'];w=packet['watch'];assert o['completed']and o['actual_step']=='910380.664'and o['source_manifest_SHA256']=='e8c8e00c5c2a8c2754494bb89a1060503aeb0de272f230f75f978027e03b9956';assert w['completed']and w['passed']and w['post']['passed'];assert len(o['outputs'])==20
 raw=[words((32768,3,192),1013904223),words((32768,3,192),12345),words((32768,3,128),98765)]
 for data,r in zip(raw,o['inputs']):assert hashlib.sha256(data.tobytes()).hexdigest()==r['SHA256']and list(data.shape)==r['shape']
 Q,K,V=map(f32,raw);del raw;refs={};sample_rows={}
 for name,(past,count)in CASES.items():
  rows=sorted(set([0,min(1,count-1),count//3,max(0,count-2),count-1]));ref=np.zeros((len(rows),3,128),np.float64)
  for head in range(3):
   keys=K[:past+count,head].astype(np.float64);values=V[:past+count,head].astype(np.float64)
   for j,row in enumerate(rows):
    total=past+row+1;score=keys[:total]@Q[past+row,head].astype(np.float64)/math.sqrt(192.);prob=np.exp(score-score.max());prob/=prob.sum();ref[j,head]=prob@values[:total]
  refs[name]=ref;sample_rows[name]=rows
 outs={};records=[]
 for r in o['outputs']:
  phase=r['phase'];name=r['case'];past,count=CASES[name];assert r['mode']==[0,1,1,0][phase]and r['past']==past and r['count']==count and r['native_rc']==0 and r['output_CPU_canary'];p=raw_dir/Path(r['path']).name;assert H(p)==r['SHA256']and p.stat().st_size==r['bytes']==count*3*128*2;data=f32(np.fromfile(p,dtype='<u2').reshape(count,3,128));assert np.isfinite(data).all();assert(phase,name)not in outs;outs[phase,name]=data
 assert len(outs)==20 and sum(v.size//(3*128)for v in outs.values())==8196
 for phase in range(4):
  for name in CASES:
   a=outs[phase,name];ref=metric(a[sample_rows[name]],refs[name]);paired=metric(a,outs[0,name]);reset=metric(a,outs[phase,'past0'])if name=='reset_reuse'else None;assert ref['passed']and paired['passed']and(reset is None or reset['passed']);records.append(dict(phase=phase,mode=[0,1,1,0][phase],case=name,fixed_independent_rows=sample_rows[name],CPU_F64_reference=ref,full_BF16_vs_first_OFF=paired,reset_vs_samephase_start=reset))
 hp=raw_dir/Path(o['driver_history']['path']).name;assert H(hp)==o['driver_history']['SHA256'];history=json.loads(hp.read_bytes());assert all(r[2]==0 and r[4]==68702699520 for r in history);assert all(b[0]>=a[0]for a,b in zip(history,history[1:]));mins={};phase_memory=[]
 for phase,mode in enumerate([0,1,1,0]):
  start=next(e for e in o['events']if e['phase']==phase and e['kind']=='phase.allocated');end=next(e for e in o['events']if e['phase']==phase and e['kind']=='phase.complete');vals=[r[3]for r in history if start['monotonic']<=r[0]<=end['monotonic']];assert vals;mins[phase]=min(vals);census=start['value'];storage=[census[16+i*8:16+(i+1)*8]for i in range(24)];assert len(set(r[0]for r in storage))==24 and all(r[1]==(64880640 if mode else 77856768)for r in storage);phase_memory.append(dict(phase=phase,mode=mode,observed_min_free=min(vals),samples=len(vals),storage_payload=sum(r[1]for r in storage)))
 pairs=[dict(OFF=off,ON=on,observed_delta_free_bytes=mins[on]-mins[off])for off,on in [(0,1),(3,2)]];assert pairs==[{k:v for k,v in p.items()if k in ['OFF','ON','observed_delta_free_bytes']}for p in pairs]
 original={(r['OFF'],r['ON']):r['observed_ON_minus_OFF_minfree']for r in o['observed_capacity_pairs']};assert all(p['observed_delta_free_bytes']==original[p['OFF'],p['ON']]for p in pairs);assert mins[2]-mins[3]>=295698432
 points=[r['free']for r in o['driver_initialization_points']if r['rc']==0]+[r[3]for r in history];assert min(points)>=4294967296;assert o['close']['rc']==0 and o['close']['values'][2]==1 and not w['post']['KFD_pids']and not w['post']['expected_PIDs_still_exist'];timings=[]
 for name in CASES:
  values={r['phase']:r['seconds']for r in o['outputs']if r['case']==name};timings.append(dict(case=name,raw_seconds_by_phase=values,warm_ON2_over_OFF3_CABI=values[2][2]/values[3][2],warm_ON2_over_OFF3_dispatch_sync=values[2][0]/values[3][0]))
 gaps=[b[0]-a[0]for a,b in zip(history,history[1:])];return dict(passed=True,actual_GPU='910380.664',original_owner_wrapper_SHA256=H(owner_path),original_report_SHA256=packet['owner_SHA256'],reader_scope='Independent original20rawBF16 SHA+fullpaired+differentfixedrows CPUF64 reference, driverhistory recomputation; no Native/SDK/GPU/Model or new componentquery',original_input_SHA_verified=True,planned_queries=20,actual_queries=20,query_tokens=8196,independent_numeric=records,CPU_F64_only=True,phase_memory=phase_memory,observed_memory_pairs=pairs,warm_delta_bytes=mins[2]-mins[3],old643_deficit_bytes=295698432,component_surplus_bytes=(mins[2]-mins[3])-295698432,driver_observed_min=min(points),driver_samples=len(history),sample_gap_median_s=statistics.median(gaps),sample_gap_max_s=max(gaps),normal_Runtime_destroy_weak_expired=True,post_PID_KFD_clean=True,component_timings=timings,memory_scope='Component clear+trim eachphase; unresetdriver/Torch history. Initialization/sub5ms peaks UNKNOWN. Observednet is not Model fit/driver upper bound.',timing_scope='CABI includes H2D/metadata/dispatch/sync and outputD2H plus ALLguards. Dispatchsync not exclusiveGPUbusy; no multiplication into Model speed.',Model_GO=False,Model_fit=False,whole_Model_performance='UNKNOWN')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--owner',type=Path,required=True);p.add_argument('--raw-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();q=run(a.owner,a.raw_directory);a.output.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k]for k in ['passed','actual_GPU','planned_queries','actual_queries','query_tokens','warm_delta_bytes','component_surplus_bytes','normal_Runtime_destroy_weak_expired','Model_fit']}))

"""Stdlib-only original BF16/artifact/numeric reader; no SDK/Model execution."""
from pathlib import Path,PurePosixPath
import argparse,array,gzip,hashlib,io,json,math,struct,sys,tarfile
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
H=lambda b:hashlib.sha256(b).hexdigest();blob=Path(a.archive).read_bytes();raw=gzip.decompress(blob)
with tarfile.open(fileobj=io.BytesIO(raw))as t:
 members=t.getmembers();assert all(v.isfile()and not PurePosixPath(v.name).is_absolute()and '..'not in PurePosixPath(v.name).parts for v in members)
 files={v.name:t.extractfile(v).read()for v in members}
m=json.loads(files['MANIFEST.json']);assert m['actual_GPU']=='910380.648'and len(m['files'])==34 and len(files)==35
for v in m['files']:assert len(files[v['member']])==v['bytes']and H(files[v['member']])==v['SHA256']
watch=json.loads(files['results/Kimi461-mixed-prefill-watch-v1.json'])
assert watch['completed']and watch['passed']and watch['actual_step']=='910380.648'and watch['last_accounting'][2:4]==['COMPLETED','0:0']
owner=json.loads(files['results/Kimi461-mixed-910380.648.json'])
assert H(files['results/Kimi461-mixed-910380.648.json'])==watch['owner_report']['SHA256']
assert owner['completed']and owner['passed']and owner['normal_destroy']and owner['driver']['cleanup_return']==0
assert owner['actual_step']=='910380.648'and owner['visible_count']==1 and owner['physical_PCI']=='0000:09:00.0'
assert owner['baseline_stats_before']==[0]*5 and owner['baseline_stats_after']==[6,15360,154183680,132167680,0]
assert owner['entry_affinity']==owner['final_affinity']and len(owner['entry_affinity'])>=4
assert owner['driver']['counts']==[3]*4 and len(owner['driver']['records'])==12 and len(owner['driver']['inspections'])==4
assert not owner['Model']and not owner['real_checkpoint_weights']and owner['synthetic_geometry_component']
for phase in ['before','post']:
 v=watch[phase];assert v['passed']and not v['KFD_pids']and not v['matching_GPU_driver_PIDs']and not v['expected_PIDs_still_exist']and len(set(v['physical_PCI']))==8
assert watch['post']['expected_owner_PIDs']==[owner['PID']]
assert owner['physical_PCI'].lower()in[v.lower()for v in watch['before']['physical_PCI']]
assert owner['HIP_VISIBLE_DEVICES']==owner['SLURM_STEP_GPUS']and owner['SLURM_STEP_GPUS']in list('01234567')
for name,v in owner['symbols'].items():
 assert v['SHA256']==('04a6a1a86c97d176e4304b9bb4c1d6fcb8550784beaa47e0f5841fb7603ffabf'if name=='modern_dense_prefill_stats_r2475'else'22121d93c0e7c8cbb16de643b86a7f44626bd51c4e8793407bfd8f86243d072a')
assert len(owner['symbols'])==9
for v in owner['driver']['inspections']:
 g=v['geometry'];mm,k,n=[(4608,7168,1),(512,7168,3584)][g];data=v['values']
 assert data[0]==0 and data[1]>0 and data[2]==1 and data[3]==g
 assert data[4:16]==[mm,k,k,1,n,k,k,1,k,n,1,k]and data[19:]==[1]*6
assert len({v['values'][1]for v in owner['driver']['inspections']})==1
# BF16 table: values are exact F32 expansions. Metric reductions use CPU F64.
lut=[struct.unpack('<f',struct.pack('<I',i<<16))[0]for i in range(65536)]
def words(b):
 v=array.array('H');v.frombytes(b)
 if sys.byteorder!='little':v.byteswap()
 return v
def metric(values,reference):
 n=0;delta2=0.;reference2=0.;ma=0.;mr=0.
 for x,y in zip(values,reference):
  assert math.isfinite(x)and math.isfinite(y)
  n+=1;d=x-y;delta2+=d*d;reference2+=y*y;ma=max(ma,abs(d));mr=max(mr,abs(y))
 nrms=math.sqrt(delta2/max(n,1))/max(math.sqrt(reference2/max(n,1)),1e-20);scaled=ma/max(1.,mr)
 return dict(elements=n,NRMS=nrms,maxabs=ma,maxabs_scaled=scaled,agreement=nrms<=.02 and scaled<=.04)
def sample_ref(mm,k,n,rows,cols):
 def numerator(i,seed):return((((i*1664525+seed)&0xffffffff)>>24)-128)
 # Independent exact integer dot on actual generated BF16 fractions /256.
 # Final cast to F32; not a claim to reproduce the provider's reduction order.
 return [struct.unpack('<f',struct.pack('<f',sum(numerator(r*k+j,1013904223)*numerator(c*k+j,12345)for j in range(k))/65536.))[0]for r in rows for c in cols]
summary=[];performance=[];total=0
for gid,(mm,k,n)in enumerate([(4608,7168,1),(512,7168,3584)]):
 records=[r for r in owner['driver']['records']if r['geometry']==gid];assert[r['mode']for r in records]==[0,1,0,1,1,0]
 src=owner['sources'][gid];rows=src['CPU_sample_rows'];cols=src['CPU_sample_columns'];ref=sample_ref(mm,k,n,rows,cols)
 off=None
 for r in records:
  v=r['output'];member=PurePosixPath(v['path']).relative_to('/refactor').as_posix();b=files[member]
  assert len(b)==v['bytes']==mm*n*2 and H(b)==v['SHA256']and v['shape']==[mm,n]and v['dtype']=='BF16'and v['endianness']=='little'
  w=words(b);assert all((x&0x7f80)!=0x7f80 for x in w);total+=len(b)
  if off is None:off=w
  full=metric((lut[x]for x in w),(lut[x]for x in off))
  cpu=metric((lut[w[i*n+j]]for i in rows for j in cols),ref)
  reported=r['CPU_sample_reference']['vs_original_OFF'];assert math.isclose(full['NRMS'],reported['NRMS'],rel_tol=1e-8,abs_tol=1e-12)
  summary.append(dict(geometry=gid,ordinal=r['ordinal'],mode=r['mode'],raw_SHA256=v['SHA256'],CPU_exact_fraction_F32_sample=cpu,vs_original_OFF=full))
 formal=records[2:];result=dict(geometry=gid,formal_OFF=2,formal_ON=2)
 for name,index in [('dispatch_drain',0),('entire_CAPI',2)]:
  a0=[r['metrics'][index]for r in formal if r['mode']==0];b0=[r['metrics'][index]for r in formal if r['mode']==1]
  av=sum(a0)/2;bv=sum(b0)/2;drift=abs(a0[-1]-a0[0])/max(a0[0],1e-30)
  result[name]=dict(OFF_mean_s=av,ON_mean_s=bv,reduction_percent=100*(av-bv)/av,OFF_endpoint_drift_percent=100*drift,sample_size_small=True)
 result['cold_warm_trim_release_bytes']=[r['component_only_after_warm_trim'][2]-r['component_only_after_warm_trim'][0]for r in records[:2]]
 performance.append(result)
agreement=all(v['CPU_exact_fraction_F32_sample']['agreement']and v['vs_original_OFF']['agreement']for v in summary)
assert agreement==owner['numeric_agreement']==watch['numeric_agreement']
out=dict(passed=True,actual_GPU='910380.648',actual_keeper='910407.1064',actual_before='910380.647',actual_post='910380.649',archive=dict(bytes=len(blob),SHA256=H(blob),members=34),BF16_outputs=12,BF16_raw_bytes=total,normal_Runtime_destroy=True,owner_TID=owner['driver']['inspections'][0]['values'][1],physical_PCI=owner['physical_PCI'],operational_pass=True,numeric_agreement=agreement,numerical=summary,performance=performance,minimum_observed_driver_free_bytes=owner['observed_floor_bytes'],capacity_decision='not enough evidence to repair643295698432B deficit; no Model reload',memory_limit='trim-release observations are cached/free driver points. Down carries earlier geometry cache; do not add geometries or call it instantaneous pool/BLAS peak.',old_post645='FAILED remains; this no-communication component does not requalify oldports',Model_GO=False,Model_speed_or_fit=False)
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['numerical']}))

"""Pure stdlib audit of sealed GPU1498 synthetic HC metadata component originals."""
from pathlib import Path,PurePosixPath
import argparse,base64,gzip,hashlib,json,math,statistics,struct
SEAL='ce8e8cd311b5fac89e1f323cd120490699bca428c9129b80e741a92001e9d267'
H=lambda b:hashlib.sha256(b).hexdigest()
OPS=['Mul','Add','Sigmoid','Add','Mul','Add','Sigmoid','Mul','Mul','Add','Softmax','Add']
def summary(a):
 assert a and all(math.isfinite(v)and v>=0 for v in a)
 return dict(n=len(a),min=min(a),median=statistics.median(a),max=max(a))
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def reference():
 mix=[(i-12)*.03125 for i in range(24)];base=[(i%7-3)*.015625 for i in range(24)];eps=f32(1e-6)
 pre=[1/(1+math.exp(-(mix[i]*.5+base[i])))+eps for i in range(4)]
 post=[2/(1+math.exp(-(mix[i+4]*.75+base[i+4])))for i in range(4)]
 comb=[]
 for i in range(4):
  row=[math.exp(mix[8+4*i+j]*.25+base[8+4*i+j])for j in range(4)];den=sum(row);comb.extend(x/den+eps for x in row)
 return dict(pre=pre,post=post,comb=comb)
def audit(path):
 raw=Path(path).read_bytes();assert H(raw)==SEAL;s=json.loads(gzip.decompress(raw));fs={}
 for f in s['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert len(b)==f['bytes']and H(b)==f['SHA256']and f['path']not in fs;fs[f['path']]=b
 assert len(fs)==56
 def one(tail):
  a=[(p,b)for p,b in fs.items()if p.endswith(tail)];assert len(a)==1,(tail,len(a));return a[0]
 def read(tail):return json.loads(one(tail)[1])
 pins={}
 for name,mf,pin in [('hc-metadata-component-v3','SOURCE_MANIFEST.json','aefb4c62f7b973b34877f0bfc9232604ffa2901cf6c0047a37743344c528426e'),('hc-metadata-GPU-v2','GPU_MANIFEST.json','5ba26348257dcb06b56b7df59be663a4a9ac27c106ee0aae10f9fd83a2e882da'),('hc-metadata-launch-v4','SOURCE_MANIFEST.json','b0845edc6f5f7e8f393ee4bc4649bc8c304254a10c80c3760b992c4ae6ed06f6')]:
  p,b=one('/'+name+'/'+mf);assert H(b)==pin;pins[p]=pin
  for n,h in json.loads(b)['files_SHA256'].items():assert H(fs[str(PurePosixPath(p).parent/n)])==h
 cpu=read('/CPU-result.json');assert H(one('/CPU-result.json')[1])=='e981f62b1e363ccee4cc6973838f7286d5e0cfd20271d77b7ce76113e2089fd2';assert cpu['completed']and cpu['passed']and cpu['actual_CPU']=='910407.1055'and cpu['new_HOST']==1 and cpu['new_GPU']==0 and cpu['fresh_M_equals_MD']and not cpu['probe_executed']and not cpu['Runtime']
 assert cpu['source_manifest_SHA256']==list(pins.values())[0]and cpu['executable_SHA256']=='bb7fdad13f280bb12b1b1ca2e20291ab35d2c63fc058fc86178f45f54a5ee7b7'
 w=read('-watch.json');worker=read('/worker.json');phase=read('-HOST-v4-phases.json');assert w['completed']and w['passed']and w['GPU']=='910374.1498'and w['CPU_keeper']=='910407.1056'and w['GPU_srun_rc']==0 and w['epoch']==1793881295.0230434;assert not any(w.get(x)for x in ['error','post_error','forced_TERM']);assert phase['completed']and phase['phase']=='keeper_submitted'and len(phase['commands'])==5 and all(c['rc']==0 for c in phase['commands'])
 assert worker['completed']and worker['passed']and worker['actual_GPU']==w['GPU']and worker['gpu_epoch']==w['epoch']and len(worker['owners'])==8 and len(worker['allowed_CPUs'])>=32;assert not any(worker[x]for x in ['weights','Model','HTTP'])
 assert H(one('/worker.json')[1])==w['worker_SHA256'];allowed=worker['allowed_CPUs'];assert allowed==sorted(set(allowed));allframes=[];rankrows=[];ref=reference();masks=[];maxerror=0
 libraries={cpu['executable']:cpu['executable_SHA256'],**{p:h for p,h in cpu['parent_libraries'].items()if not PurePosixPath(p).name.startswith('libDeep_')}}
 for rank,r in enumerate(worker['owners']):
  assert rank==r['rank']and r['returncode']==0 and not r.get('TERM')and not r.get('KILL');p,b=one('/rank'+str(rank)+'.json');assert H(b)==r['report_SHA256'];z=json.loads(b);assert z['completed']and z['passed']and z['normal_Runtime_expired']and z['rank']==rank and not any(z[x]for x in ['Model','weights','HTTP']);assert z['entry_CPUmask']==allowed and len(z['frames'])==7
  assert r['actual_provider_maps']==libraries==worker['map_observations'][str(rank)];assert z['metadata']==r['metadata']and z['Mul_Torch8_hypothesis']and r['Mul_Torch8_hypothesis'];md=z['metadata'];assert md['Mul']['implementation_index']==8 and 'ELm8E'in md['Mul']['provider_RTTI']and md['Add']['implementation_index']==md['Sigmoid']['implementation_index']==0 and all(v['workspace_bytes']==0 for v in md.values())
  f0=z['frames'][0];mask=allowed[rank*4:(rank+1)*4];assert f0['mask_observed_after_original_completion']and f0['actual_CPUmask_after_first_completed_frame']==r['actual_SDK_bound_mask']==r['expected_mask']==mask;assert f0['owner_TID']==r['actual_owner_TID']==r['PID']and type(f0['owner_TID'])is int and f0['owner_pthread'];masks+=mask
  for name in ['memory_before_create','memory_after_create','memory_after_frames']:assert z[name]['free']>=16*1024**3 and z[name]['total']>=z[name]['free']
  for i,f in enumerate(z['frames']):
   assert f['frame']==i and f['cold']==(i==0)and f['warm']==(i in [1,2])and f['finite_CPUref_passed']and [o['op']for o in f['ops']]==OPS
   error=0
   for name,expected in ref.items():
    got=f['output'][name];assert len(got)==len(expected)and all(math.isfinite(x)for x in got);error=max(error,max(abs(x-y)for x,y in zip(got,expected)))
   assert error<=2e-6 and abs(error-f['max_abs_CPUref'])<1e-12;maxerror=max(maxerror,error)
   for o in f['ops']:
    for k in ['allocation_wall_s','allocation_thread_s','submit_wall_s','submit_thread_s']:assert math.isfinite(o[k])and o[k]>=0
   row=dict(rank=rank,frame=i,mode='cold'if i==0 else'warm'if i in [1,2]else'measure',whole_ms=1000*f['whole_to_original_completion_s'],thread_ms=1000*f['thread_to_completion_s'],final_drain_ms=1000*f['final_drain_s'],view_ms=1000*f['narrow_view_wall_s'],allocations_ms=1000*sum(o['allocation_wall_s']for o in f['ops']),submits_ms=1000*sum(o['submit_wall_s']for o in f['ops']),allocation_thread_ms=1000*sum(o['allocation_thread_s']for o in f['ops']),submit_thread_ms=1000*sum(o['submit_thread_s']for o in f['ops']),max_abs_CPUref=error)
   row['unassigned_instrumented_enclosure_ms']=row['whole_ms']-sum(row[k]for k in ['final_drain_ms','view_ms','allocations_ms','submits_ms']);assert row['unassigned_instrumented_enclosure_ms']>=-1e-5
   row['op_submit_ms']={name:1000*sum(o['submit_wall_s']for o in f['ops']if o['op']==name)for name in ['Mul','Add','Sigmoid','Softmax']};allframes.append(row)
  rankrows.append(dict(rank=rank,PID=r['PID'],startticks=r['startticks'],entry_CPUmask=z['entry_CPUmask'],actual_bound_CPUmask=mask,owner_TID=f0['owner_TID'],report_SHA256=H(b),normal_Runtime_expired=z['normal_Runtime_expired'],metadata=md,memory={k:z[k]for k in ['memory_before_create','memory_after_create','memory_after_frames']},formal_whole_ms=summary([f['whole_ms']for f in allframes if f['rank']==rank and f['mode']=='measure'])))
 assert len(masks)==32 and len(set(masks))==32 and len(allframes)==56
 posts={}
 for which,step in [('before','910374.1497'),('post','910374.1499')]:
  lines=one('-'+which+'-'+step+'.log')[1].decode().splitlines();q=json.loads(next(x for x in reversed(lines)if x.startswith('{')));assert q['actual_step']==step and q['KFD_available']and not q['kfd_pids']and not q['matching_component_processes']and not q['expected_PIDs_still_exist']and q['ports_bindable']['43882']and len(set(q['physical_PCI']))==8;posts[which]=q
 assert w['post']['rc']==0 and w['fresh']['rc']==0;assert b'32768M' in one('/hc-metadata-GPU-v2/controller.py')[1]
 rows=s['accounting']['rows'];assert s['accounting']['rc']==0 and len(rows)==4
 for step in ['910374.1497','910374.1498','910374.1499','910407.1056']:assert any(x.split('|')[0]==step and x.split('|')[2:4]==['COMPLETED','0:0']for x in rows)
 group={}
 for mode in ['cold','warm','measure']:
  a=[x for x in allframes if x['mode']==mode];group[mode]={k:summary([r[k]for r in a])for k in ['whole_ms','thread_ms','final_drain_ms','view_ms','allocations_ms','submits_ms','allocation_thread_ms','submit_thread_ms','unassigned_instrumented_enclosure_ms']};group[mode]['op_submit_sum_ms']={k:summary([r['op_submit_ms'][k]for r in a])for k in ['Mul','Add','Sigmoid','Softmax']}
 return dict(schema='Deep-HC-metadata1498-original-audit-v1',passed=True,scope='Eight independent device owners; synthetic HC T1 same-op component, NOT Model/exclusive-GPU attribution',archive=dict(path=str(path),SHA256=H(raw),bytes=len(raw),files=len(fs),decoded_bytes=sum(map(len,fs.values()))),source_manifests=pins,CPU=dict(actual=cpu['actual_CPU'],receipt_SHA256=H(one('/CPU-result.json')[1]),executable_SHA256=cpu['executable_SHA256'],fresh_M_equals_MD=True,new_HOST=1,new_GPU=0),GPU=w['GPU'],keeper=w['CPU_keeper'],epoch=w['epoch'],accounting=rows,post=posts,normal8=True,all8_maps=libraries,owners=rankrows,frames=allframes,frame_counts=dict(cold=8,warm=16,measure=32,total=56),submits_per_owner=84,total_submits=672,max_abs_CPUref_recomputed=maxerror,timing_distributions_ms=group,known_original_failure='1494 OUT_OF_MEMORY + premature prepin remains FAIL; no old result rewritten',Model=False,weights=False,HTTP=0,exclusive_GPUbusy=False,exclusive_CPUbusy=False,Model_gain='UNKNOWN',timing_contract=['One frame includes 12alloc/12submit/12narrow/1view+original completion fence. Subintervals are host enclosures, not kernel events.','D2H/reference, frame0 selfmask, JSON persistence and existing Work destructor drain are outside sampled whole. Their cost remains in rank/worker lifetime, not a separately measured phase.','Metadata Make precedes frame0. Cold means first timed Call sequence, not process/ELF cold. Formal distributions are 4frames x8owners, not a repeated Model/ABBA gain.','Eight owners execute concurrently. Do not sum8 owner time as latency. Different op disjoint host enclosures can sum only within a frame.','Same Model HC branch has 86HC perowner perdecode(43layers x2), same F32 geometry/op order; actual Model inputs, allocator residency/cache history, event scopes, surrounding work and synchronization differ. No component ratio is a Model saving.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(passed=q['passed'],files=q['archive']['files'],GPU=q['GPU'],normal8=q['normal8'],submits=q['total_submits'],formal=q['frame_counts']['measure'],max_abs_CPUref=q['max_abs_CPUref_recomputed'],measure_ms=q['timing_distributions_ms']['measure']['whole_ms'],output_SHA256=H(Path(a.output).read_bytes()))))

"""Audit original1494 FAIL and actual provider metadata; no device/source imports."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json
SEAL='fac71a2841a60c1e381d0cd04c79bbba77909a919228a40287c85e35756214e1'
H=lambda b:hashlib.sha256(b).hexdigest()
def audit(path):
 raw=Path(path).read_bytes();assert H(raw)==SEAL;s=json.loads(gzip.decompress(raw));fs={}
 for f in s['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert len(b)==f['bytes']and H(b)==f['SHA256']and f['path']not in fs;fs[f['path']]=b
 assert len(fs)==52
 def one(tail):
  a=[(p,b)for p,b in fs.items()if p.endswith(tail)];assert len(a)==1,(tail,len(a));return a[0]
 def read(tail):return json.loads(one(tail)[1])
 for name,mf,pin in [('hc-metadata-GPU-v1','GPU_MANIFEST.json','0e82853e2c4540a7f126a116b7e8ab2b5ae5445faac007e24e22f43a0e2e8df2'),('hc-metadata-launch-v3','SOURCE_MANIFEST.json','cbc0f08f2fe6cb6e036ad8f7c2ee4d2543428342431f1d27adfe2563439fa631'),('hc-metadata-component-v2','SOURCE_MANIFEST.json','deb468c46e3c544d192491e80dd2d3bd34f695b73a6c9911860765e4279f7b47')]:
  p,b=one('/'+name+'/'+mf);assert H(b)==pin
  for n,h in json.loads(b)['files_SHA256'].items():assert H(fs[str(Path(p).parent).replace('\\','/')+'/'+n])==h
 w=read('-watch.json');worker=read('/worker.json');phase=read('-HOST-v3-phases.json');assert w['completed']and not w['passed']and w['GPU']=='910374.1494'and w['CPU_keeper']=='910407.1050'and w['GPU_srun_rc']==1 and w['epoch']==1793875369.2821825;assert phase['completed']and phase['phase']=='keeper_submitted'and len(phase['commands'])==5 and all(c['rc']==0 for c in phase['commands'])
 assert worker['completed']and not worker['passed']and len(worker['owners'])==8 and len(worker['allowed_CPUs'])==32
 for r in worker['owners']:assert r['actual_mask']==r['expected_mask']==worker['allowed_CPUs'][r['rank']*4:(r['rank']+1)*4]
 ranks=[];missing=[]
 for rank in range(8):
  paths=[(p,b)for p,b in fs.items()if p.endswith('/rank'+str(rank)+'.json')]
  if not paths:missing.append(rank);continue
  p,b=paths[0];q=json.loads(b);assert q['completed']and not q['passed']and q['rank']==rank and q['error']=='2392 owner mask too narrow for eight disjoint requested subsets'and q['frames']==[]
  m=q['metadata'];assert m['Mul']['implementation_index']==8 and 'ELm8E'in m['Mul']['provider_RTTI']and m['Add']['implementation_index']==m['Sigmoid']['implementation_index']==0 and all(v['workspace_bytes']==0 for v in m.values());assert q['Mul_Torch8_hypothesis'];ranks.append(dict(rank=rank,report_SHA256=H(b),metadata=m,memory_before_create=q['memory_before_create'],memory_after_create=q['memory_after_create'],error=q['error'],frames=0,normal_Runtime_expired=q.get('normal_Runtime_expired')))
 assert [r['rank']for r in ranks]==[0,2,3,4,6]and missing==[1,5,7];gpu_log=one('-910374.1494.log')[1].decode();assert 'Detected 4 oom_kill events'in gpu_log and'Out Of Memory'in one('/Deep-HC-metadata-r467-device-v1-srun.log')[1].decode();post_lines=one('-post-910374.1495.log')[1].decode().splitlines();post=json.loads(next(x for x in reversed(post_lines)if x.startswith('{')));assert post['actual_step']=='910374.1495'and post['KFD_available']and not post['kfd_pids']and not post['matching_component_processes']and not post['expected_PIDs_still_exist']and post['ports_bindable']['43882']and len(set(post['physical_PCI']))==8
 rows=s['accounting']['rows'];assert s['accounting']['rc']==0
 for step,state,rc in [('910374.1494','OUT_OF_MEMORY','0:125'),('910407.1050','FAILED','1:0'),('910374.1495','COMPLETED','0:0')]:assert any(v.split('|')[0]==step and v.split('|')[2:4]==[state,rc]for v in rows)
 return dict(schema='Deep-HC-metadata1494-original-audit-v1',passed=True,scope='Offline actualFAIL/providermetadata/post audit; NOT componentperformance/lifecyclePASS',archive=dict(path=str(path),SHA256=H(raw),files=len(fs),decoded_bytes=sum(map(len,fs.values()))),actual_GPU='910374.1494',CPU_keeper='910407.1050',epoch=w['epoch'],original_accounting=rows,original_watch_passed=False,GPU_srun_rc=1,normal8_Runtime_exit_qualified=False,post=post,metadata_rank_reports=ranks,missing_rank_reports=missing,observed_timed_frames=0,observed_formal_frames=0,finite_CPUref_device_qualified=False,actual_Mul_provider='Torch implementation8 confirmed in5actualrankmetadata; no timing or Modelattribution',root_causes=[dict(kind='HOSTcaller_affinity_order',evidence='Eachchild preexec reduced common32CPU mask to4 beforefirstSDKownedsubmit; actual2392 rejects width4 because it requires8*4 commonentryCPUs. Existingfirstbindcontract violated. No arithmeticnegative.'),dict(kind='host_cgroup_OOM',evidence='GPUstep inherited --mem8192M despite8separateSDK/Torchprocesses; Slurm reports4cgroupoomkill events and OUT_OF_MEMORY. Not a CUDA VRAM failure; presentrankGPUmemory floors passed beforefailure. No permissingrank oom attribution invented.')],performance='UNKNOWN',Model=False,weights=False,new_jobs=0,new_compile=0,source_repair_candidate='Separateimmutable worker keeps originalcommonmask throughfirstownedsubmit, thenobserveSDK4CPUdisjointmask; GPUhostRAM32GiB,keeper4GiB and120/100/180 same; noC++recompile/replay untilpublishednewscope/fresh.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(passed=q['passed'],files=q['archive']['files'],metadata_ranks=len(q['metadata_rank_reports']),timed_frames=q['observed_timed_frames'],GPU=q['actual_GPU'],post=q['post']['actual_step'],output_SHA256=H(Path(a.output).read_bytes()))))

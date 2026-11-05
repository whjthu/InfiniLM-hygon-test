"""Independent stdlib consumer of the sole r470 device-v1 failure; never imports producer."""
import argparse,base64,collections,gzip,hashlib,json
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
SEAL='cf3a5d646c0fcf68bf402f3f5006b7bfc639fc1d73d39e69b271ff22e348f8db'
MF='f0ef73ba8c910f632bebeaeeb7563ac59173bc5556b24de457470f0d92b68226'
GPU_MF='b0ea733796870504db17123ccf63d201bfa0ed72607b61e27f4bfb850b00d3ad'
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014';LABEL='GLM-global-TP16-comm-r470-device-v1';D=R+'/incoming/GLM952-global-TP16-component-r470/device-v1/';O=R+'/results/'+LABEL+'/'
NODES=['a02r3n15','a02r3n16']
def audit(path):
 raw=Path(path).read_bytes();assert H(raw)==SEAL and len(raw)==53707
 a=json.loads(gzip.decompress(raw));files={}
 for f in a['files']:
  assert f['path']not in files;b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert len(b)==f['bytes']and H(b)==f['SHA256'];files[f['path']]=b
 assert len(files)==74 and sum(map(len,files.values()))==148886
 j=lambda p:json.loads(files[p]);sha=lambda p:H(files[p])
 assert sha(D+'SOURCE_MANIFEST.json')==MF
 for n,h in j(D+'SOURCE_MANIFEST.json')['files_SHA256'].items():assert sha(D+n)==h
 gpu_source=Path(__file__).resolve().parent.parent/'device-v1/GPU_MANIFEST.json';gpu_bytes=gpu_source.read_bytes();assert H(gpu_bytes)==GPU_MF
 m=json.loads(gpu_bytes);assert sha(D+'GPU_RESOURCE_CLAIM.json')==m['claim_SHA256'];assert sha(D+'CPU_BINDING.json')==m['CPU_binding_SHA256']
 binding=j(D+'CPU_BINDING.json');assert binding['actual_CPU']=='910407.1061'and binding['library_SHA256']=='a2ecf2ccfe787cdb814368d1e556eae2658829277a993cca7d5d2c8e89cc8983'
 w=j(R+'/results/'+LABEL+'-watch.json');assert w['completed']and not w['passed']and w['GPU']is None and w['CPU_keeper']=='910407.1065'and w['srun_rc']==143
 epoch=w['epoch'];assert epoch==1793891658.298647 and w['seconds']<300
 rows={v.split('|')[0]:v.split('|')for v in a['accounting']['stdout'].splitlines()};assert a['accounting']['rc']==0
 for step,state,seconds in [('910371.869','COMPLETED','0'),('910371.870','FAILED','151'),('910371.871','COMPLETED','1'),('910407.1065','FAILED','153')]:assert rows[step][2]==state and rows[step][4]==seconds and rows[step][3]==('0:0'if state=='COMPLETED'else'1:0')
 assert 'controller.py' in w['traceback']and 'row=account();assert row' in w['traceback']
 workers=[j(O+'worker-%d.json'%i)for i in range(2)];owners=[j(O+'owner-%02d.json'%i)for i in range(16)];uid=j(O+'UIDS.json');uids=[base64.b64decode(uid[k],validate=True)for k in ['world','tp']];assert uids[0]!=uids[1]and len(uids[0])==len(uids[1])
 assert uid['actual_GPU']=='910371.870'and uid['epoch']==epoch and uid['source']==MF
 detail=[];geometry_count=0
 expectedlibs={binding['library']:binding['library_SHA256'],**binding['verified_library_SHA256']}
 for node,worker in enumerate(workers):
  assert worker['actual_GPU']=='910371.870'and worker['gpu_epoch']==epoch and worker['node']==NODES[node]and len(worker['owners'])==8 and len(worker['common_CPUmask'])>=32
  for r0 in worker['owners']:
   i=r0['global_rank'];r=owners[i];assert i==node*8+r['device']and r['global_rank']==i and r['node']==NODES[node]and r['PID']==r['owner_TID']==r0['PID']and r['startticks']==r0['startticks']and r['UID']==154459
   assert r['entry_CPUmask']==worker['common_CPUmask']and r['actual_GPU']=='910371.870'and r['gpu_epoch']==epoch and r['source_manifest_SHA256']==MF
   assert not r['Model_READY']and not r['weights']and not r['cache_allocated']and r['HTTP']==0 and not r['passed']and not r['completed']
   assert r['before_create']['free']>=8*1024**3 and r['before_create']['total']==68702699520
   assert r['UID_SHA256']==list(map(H,uids))
   assert set(r['actual_provider_maps'])==set(expectedlibs)
   for p,h in expectedlibs.items():assert r['actual_provider_maps'][p]['SHA256']==h and r['actual_provider_maps'][p]['unique_loaded_inode_count']==1
   assert j(O+'created-%02d.json'%i)==dict(global_rank=i,epoch=epoch)
   assert O+'proved-%02d.json'%i not in files and O+'proved-%02d.json'%i in a['missing_exact_paths']
   assert len(files[O+'owner-%02d.log'%i])==0
   phases=[e['phase']for e in r['events']];assert phases[:2]==['library_load','create_global16']
   if i<8:
    assert phases==['library_load','create_global16','one_collective_proof']and r['phase']=='one_collective_proof';g=r['geometry'];assert g['component_created']and g['TP']==16 and g['PP']==1 and g['tp_rank']==g['world_rank']==i and g['capacity_per_row']==32768 and g['local_attention_heads']==4 and g['local_index_heads']==2 and g['layer_end']==78 and not g['Model_READY'];assert r['after_create_CPUmask']==r['entry_CPUmask']and r['after_create']['free']>=8*1024**3;geometry_count+=1
   else:assert phases==['library_load','create_global16']and r['phase']=='create_global16'and 'geometry'not in r
   detail.append(dict(rank=i,node=r['node'],local_device=r['device'],PID=r['PID'],startticks=r['startticks'],physical_PCI=r['physical_PCI'],last_persisted_phase=r['phase'],phase_relative_seconds=[dict(phase=e['phase'],seconds=e['epoch']-epoch)for e in r['events']],created_marker=True,proved_marker=False,normal_close=False,report_SHA256=sha(O+'owner-%02d.json'%i)))
 assert len({(r['node'],r['physical_PCI'])for r in owners})==16
 for n in NODES:assert len({r['physical_PCI']for r in owners if r['node']==n})==8
 assert workers[0]['completed']and not workers[0]['passed']and workers[0]['forced_TERM']and not workers[1]['completed']and not workers[1]['passed']
 log=files[R+'/logs/'+LABEL+'-910371.870.log'].decode();assert "TimeoutError: Original150worker deadline"in log
 post=w['failure_post'];assert post['raw']['rc']==0 and{v['node']for v in post['nodes']}==set(NODES)
 for p in post['nodes']:
  assert p['actual_step']=='910371.871'and p['KFD_available']and not p['kfd_pids']and not p['matching_component_processes']and not p['expected_owned_PIDs_still_live']and not p['expected_UIDsocket_listeners_still_live']and all(p['ports_bindable'].values())
 return dict(audit_passed=True,experiment_passed=False,decision='STOP_DEVICE_COLLECTIVE_ENCLOSURE_TIMEOUT_UNKNOWN',seal=dict(path=Path(path).name,SHA256=H(raw),files=len(files),decoded_bytes=sum(map(len,files.values()))),source_manifest_SHA256=MF,GPU_manifest_SHA256=GPU_MF,CPU_binding=binding,actual_GPU='910371.870',CPU_keeper='910407.1065',before='910371.869',post='910371.871',epoch=epoch,accounting=a['accounting'],watch_GPU_null_boundary='Producer assigns GPU only after success-accounting assertion; independent accounting proves actual failed870.',owners=detail,created_markers=16,persisted_geometry_reports=geometry_count,proved_markers=0,normal_owner_close=0,forced_TERM=True,worker1_finally_incomplete=True,independent_post_clean=True,causal_boundary='16 created markers occur only after actual create/geometry/memory/mask success. node0 last persisted inside combined proof; node1 last saved before create but has later created markers. No proof return or per-call internal phase. Cannot separate shared-file barrier visibility, mark persistence, allreduce, allgather, sync or D2H; no unique CCL/network diagnosis.',qualification=dict(global16_comm_create_return_markers=16,collective_numeric=False,normal16_destroy=False,Model=False,cache_fit=False,weights=False,performance=False),no_new_device_or_compile=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);b=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();Path(a.output).write_bytes(b);print(json.dumps(dict(audit_passed=q['audit_passed'],experiment_passed=False,files=q['seal']['files'],created=16,proved=0,post_clean=True,output_SHA256=H(b))))
if __name__=='__main__':main()

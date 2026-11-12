from pathlib import Path
import hashlib,json,subprocess
B=Path(r'C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7');D=B/'training-vggt-original-inputs-CPU-v1';out=json.loads((D/'transport-outcome.json').read_bytes())['value'];A=Path(r'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
code='''from pathlib import Path
import hashlib,json,os,signal,socket,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-original-inputs-CPU-v1');start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('independent original CPU artifact audit exhausted')));signal.alarm(55)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
out=json.loads((W/'outcome.json').read_bytes());assert out['claim']==CLAIM and out['request_id']==REQUEST and out['passed'] and out['returncode']==0
raw=(W/'CPU-result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==out['CPU_result_SHA256'];cpu=json.loads(raw);assert cpu['passed'] and cpu['valid_batches']==4 and cpu['candidate_batches']==7 and cpu['rejected_batches']==3 and cpu['historical_reference_identity']['weights_match'] and cpu['historical_reference_identity']['batches_match']
assert cpu['model_weight']['SHA256']=='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
artifact=Path(cpu['private_CPU_batch_artifact']['path']);assert artifact.parent==W and artifact.stat().st_size==cpu['private_CPU_batch_artifact']['bytes'] and sha(artifact)==cpu['private_CPU_batch_artifact']['SHA256']
assert sha(W/'CPU.private.log')==out['log_SHA256'] and sha(W/'DATA_FILES.private.json')==cpu['data_files_manifest_SHA256'];data=json.loads((W/'DATA_FILES.private.json').read_bytes());assert len(data['files'])==42 and len(data['attempts'])==7 and sum(not r['valid'] for r in data['attempts'])==3
import torch
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0;torch.set_num_threads(2)
def forbid(*_a,**_kw):raise RuntimeError('independent reader must remain CPU-only')
torch.cuda._lazy_init=forbid
source=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-initialization-CPU-v1/framework');sys.path.insert(0,str(source/'generation/src'))
from gencuda.hy_resume_checkpoint import tree_sha256
from gencuda.backends.vggt_training import _vggt_batch_content_sha256,valid_depth_batch
value=torch.load(artifact,map_location='cpu',weights_only=True);assert set(value)=={'batches','config_SHA256','selected_batch_content_SHA256'} and value['config_SHA256']==cpu['config_SHA256'] and len(value['batches'])==4
for batch,row in zip(value['batches'],cpu['batches']):
 assert tree_sha256(batch)==row['content_SHA256'] and valid_depth_batch(batch['point_masks'])
 for name,pin in row['fields'].items():
  tensor=batch[name];assert tensor.device.type=='cpu' and list(tensor.shape)==pin['shape'] and str(tensor.dtype)==pin['dtype'] and tree_sha256(tensor)==pin['SHA256'] and (not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()))
 assert batch['images'].shape==(1,2,3,224,224)
assert _vggt_batch_content_sha256(value['batches'])==value['selected_batch_content_SHA256']==cpu['selected_batch_content_SHA256'];assert not torch.cuda.is_initialized()
owner=out['owner'];alive=None
if socket.gethostname()==owner['host']:
 p=Path('/proc')/str(owner['PID'])/'stat';alive=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not alive
print(json.dumps(dict(passed=True,original_outcome=out,original_owner_alive=alive,typed_batches=4,typed_tensor_fields=sum(len(r['fields']) for r in cpu['batches']),all_typed_values_finite=True,all_field_hashes_match=True,historical_weights_and_inputs_match=True,original_outcome_SHA256=sha(W/'outcome.json'),reader_wall_seconds=time.monotonic()-start,GPU_initialized=torch.cuda.is_initialized(),raw_tensor_transfer_bytes=0,model_hash_repeated=False,data_pipeline_replayed=False)))
'''
payload=('CLAIM='+repr(out['claim'])+'\nREQUEST='+repr(out['request_id'])+'\n'+code).encode();(D/'independent-reader-intent.json').write_bytes((json.dumps(dict(original_claim=out['claim'],original_request=out['request_id'],source_SHA256=hashlib.sha256(payload).hexdigest(),CPU_only=True,raw_tensor_transfer_bytes=0,deadline_seconds=55),indent=2)+'\n').encode())
remote='CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/whj/nvidia-baseline/env/bin/python -B -u -';ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26',remote];r=subprocess.run(ssh,input=payload,capture_output=True,timeout=65);(D/'independent-reader-stdout.private.json').write_bytes(r.stdout);(D/'independent-reader-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0,r.stderr[-250:];q=json.loads(r.stdout);(D/'independent-original-inputs-terminal.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q[k] for k in ('passed','typed_batches','typed_tensor_fields','historical_weights_and_inputs_match','original_owner_alive','reader_wall_seconds','GPU_initialized','raw_tensor_transfer_bytes')}))

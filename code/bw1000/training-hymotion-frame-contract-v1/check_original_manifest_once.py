from pathlib import Path
import ast,datetime,hashlib,json,subprocess,uuid
B=Path(__file__).parent;D=B/'training-hymotion-original-metadata-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-hymotion-frame-contract-v1.json']));name='deltas/tools/hygon_training_recipe.py';raw=subprocess.check_output(git+['show',claim+':'+task['source']+'/'+name]);assert hashlib.sha256(raw).hexdigest()==task['source_SHA256'][name];node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='verify_motion_manifest');function=ast.get_source_segment(raw.decode(),node);request=uuid.uuid4().hex
REMOTE=r'''
from pathlib import Path
import hashlib,json,sys
assert 'torch' not in sys.modules
exec(compile(FUNCTION,'published_verify_motion_manifest','exec'))
plan=dict(case='T08.flow-matching-lora',config=dict(training_data_manifest=MANIFEST,training_data_manifest_sha256=EXPECTED))
rejected=False;reason=None
try:verify_motion_manifest(plan)
except ValueError as e:
 reason=str(e);rejected=reason.startswith('HY-Motion requires 128 frames; the pinned manifest declares 60.')
assert rejected and 'torch' not in sys.modules
p=Path(MANIFEST);raw=p.read_bytes();data=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==EXPECTED and data['frames']==60 and data['feature_dim']==201 and data['sample_count']==128
print(json.dumps(dict(request_id=REQUEST,rejected_as_expected=True,reason_code='TEMPORAL_LENGTH_MISMATCH',required_frames=128,actual_frames=data['frames'],sample_count=data['sample_count'],feature_dim=data['feature_dim'],manifest_SHA256=hashlib.sha256(raw).hexdigest(),published_guard_source_SHA256=hashlib.sha256(FUNCTION.encode()).hexdigest(),Torch_imported=False,device_backend_imported=False,model_weight_bytes_read=0,cache_tensor_body_bytes_read=0,training_executed=False,Model_GO=False)))
'''
payload=('FUNCTION='+repr(function)+'\nMANIFEST='+repr(task['original_manifest']['path'])+'\nEXPECTED='+repr(task['original_manifest']['SHA256'])+'\nREQUEST='+repr(request)+'\n').encode()+REMOTE.encode()
with (D/'published-guard-original-request.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),guard_function_source_SHA256=hashlib.sha256(function.encode()).hexdigest(),readonly=True,new_Slurm=0,new_GPU=0),f,indent=2)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
try:r=subprocess.run(ssh,input=payload,capture_output=True,timeout=30);rc=r.returncode;out=r.stdout;err=r.stderr
except subprocess.TimeoutExpired as e:rc=None;out=e.stdout or b'';err=e.stderr or b''
for n,raw in [('published-guard.stdout.private.json',out),('published-guard.stderr.private.txt',err)]:
 with (D/n).open('xb') as f:f.write(raw)
v=json.loads(out) if rc==0 else None;record=dict(returncode=rc,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=claim,request_id=request,value=v,stdout_SHA256=hashlib.sha256(out).hexdigest(),stderr_SHA256=hashlib.sha256(err).hexdigest())
with (D/'published-guard-original-manifest-result.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(dict(returncode=rc,rejected_as_expected=v.get('rejected_as_expected') if v else None,required_frames=128,actual_frames=v.get('actual_frames') if v else None,sample_count=v.get('sample_count') if v else None,Torch_imported=v.get('Torch_imported') if v else None,training_executed=False)))

"""Consume the actual full-read result and owned PID without another weight read."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_outcome']=='ORIGINAL_RETURNED' and original['value']['passed']
claim=original['claim'];request=original['request_id'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)];task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-vggt-target-full-SHA-CPU-v1.json']))
remote=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEXPECTED='+repr(original['value'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,socket,sys
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vggt-target-full-SHA-CPU-v1');artifacts={};values={}
for name in ('intent.json','outcome.json','progress.json'):
 p=W/name;assert p.is_file() and p.stat().st_size<=65536;raw=p.read_bytes();artifacts[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());values[name]=json.loads(raw)
intent=values['intent.json'];out=values['outcome.json'];progress=values['progress.json']
assert out==EXPECTED and out['passed'] and out['claim_commit']==intent['claim_commit']==CLAIM and out['request_id']==intent['request_id']==progress['request_id']==REQUEST
assert out['epoch']==intent['epoch']==progress['epoch'] and out['deadline']==intent['deadline']==progress['deadline']==out['epoch']+300 and out['work_deadline']==intent['work_deadline']==out['epoch']+270
assert 0<out['hash_seconds']<=out['wall_seconds']<270 and len(out['CPU_affinity'])==2 and out['address_space_limit_bytes']==512*1024**2 and out['peak_RSS_bytes']<=512*1024**2
assert out['file_bytes_read']==5026874952 and out['full_file_SHA256']=='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0' and out['before_stat']==out['after_stat'] and out['before_stat']['bytes']==out['file_bytes_read']
assert progress['planned_bytes']==5026874952 and 0<progress['completed_bytes']<=5026874952 and progress['failed'] is False
assert out['byte_stream_only'] and out['Torch_imported'] is False and out['pickle_imported'] is False and not out['checkpoint_values_decoded'] and not out['model_loaded'] and out['CO3D_input_bytes']==out['GPU_cards']==0
assert out['parent_state']=='RUNNING' and out['parent_nodes']==40
model=Path(out['model_path']);s=model.stat();current=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode);assert current==out['after_stat'] and not model.is_symlink() and sorted(p.name for p in model.parent.iterdir())==['model.pt']
owner=out['owner'];assert owner==intent['owner'] and owner['host']==socket.gethostname()
try:now=int(Path('/proc',str(owner['PID']),'stat').read_text().split(') ',1)[1].split()[19])
except (FileNotFoundError,ProcessLookupError):now=None
assert now!=owner['startticks']
old=model.parent.parent
for name,key in [('outcome.json','original_receiver_outcome_SHA256'),('intent.json','original_receiver_intent_SHA256')]:
 raw=(old/name).read_bytes();assert len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==out[key]
assert out['original_transfer_claim']=='7e69baa307250ceee7a9aed5b762212fc5a9cde3' and out['original_transfer_request']=='fdda8f5afc4c4e67a97ed634ea121a10'
assert 'torch' not in sys.modules and 'pickle' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,original_outcome=out,artifacts=artifacts,original_hash_owner_identity_present=False,current_target_stat=current,independent_full_target_read_verified=True,second_full_target_read_performed=False,weight_file_bytes_read_by_reader=0,Torch_imported=False,pickle_imported=False,GPU_GO=False,Model_GO=False)))
'''
with (D/'root-reader-v2-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,read_only=True,payload_SHA256=hashlib.sha256(remote).hexdigest()),f)
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for name,raw in [('root-reader-v2.stdout.private.json',r.stdout),('root-reader-v2.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,claim=claim,request_id=request,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'root-reader-v2-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,independent_terminal_reader_passed=v.get('passed'),original_hash_owner_identity_present=v.get('original_hash_owner_identity_present'),independent_full_target_read_verified=v.get('independent_full_target_read_verified'),weight_file_bytes_read_by_reader=0)))

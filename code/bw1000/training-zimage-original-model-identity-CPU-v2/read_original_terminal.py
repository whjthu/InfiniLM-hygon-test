"""Independent bounded original terminal consumer; never rehash weight bodies."""
from pathlib import Path
import datetime,hashlib,json,os,re,signal,socket,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-original-model-identity-CPU-v2')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded independent identity reader expired')))
signal.alarm(60);start=time.monotonic()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def stats(p):
 s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def process(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
  return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
artifacts={};artifacts_bytes=0;failure=None;identity=None;result=None;checked_headers=0;header_bytes_read=0
try:
 assert W.is_dir() and not W.is_symlink()
 for target,pin in EXPECTED_FILES.items():
  p=W/target;assert not p.is_symlink() and p.stat().st_size==pin['bytes']
  raw=p.read_bytes();assert sha(raw)==pin['SHA256']
 for n in ('intent.json','CPU-owner.json','CPU-child-owner.json','CPU-result.json','MODEL_IDENTITY.json','hash-progress.private.jsonl','hash-progress.json','outcome.json','CPU.private.log','failure-private.json','hash-failure.json'):
  p=W/n
  if p.exists():
   assert not p.is_symlink() and p.stat().st_size<=8*1024**2
   raw=p.read_bytes();artifacts_bytes+=len(raw);assert artifacts_bytes<=16*1024**2
   artifacts[n]=dict(bytes=len(raw),SHA256=sha(raw))
   if n.endswith('.json'):artifacts[n]['value']=json.loads(raw)
   elif n.endswith('.jsonl'):artifacts[n]['value']=[json.loads(line) for line in raw.splitlines() if line]
 intent=artifacts['intent.json']['value'];outcome=artifacts['outcome.json']['value'];result=artifacts['CPU-result.json']['value']
 assert intent['claim']==CLAIM and outcome['claim']==CLAIM and intent['request_id']==REQUEST==outcome['request_id']
 assert intent['epoch']==outcome['epoch']==result['epoch'] and intent['deadline']==outcome['deadline']==intent['epoch']+600
 assert result['deadline']==intent['epoch']+540
 assert outcome['returncode']==0 and outcome['passed'] and outcome['owned_group_reaped'] and outcome['stop_reason'] is None
 assert outcome['wall_seconds']<600 and len(outcome['CPU_affinity'])==2 and outcome['peak_group_RSS_bytes']<=1024**3
 assert result['passed'] and result==outcome['CPU_result'] and artifacts['CPU-result.json']['SHA256']==outcome['CPU_result_SHA256']
 assert result['framework_commit']=='7129df736edd2d4a2932b8409078c1ed4cfaccb7' and result['framework_subset_files']==1
 assert result['model_files']==77 and result['preset_files']==1 and result['full_file_bytes_read']==33071396964
 assert result['model_weight_files']==7 and result['preset_weight_files']==1
 assert result['transformer_tensors']==521 and result['preset_tensors']==480 and result['declared_preset_full_SHA_matched']
 assert not any(result[k] for k in ('decoded_tensor_values','weight_values_exported','model_loaded','Torch_imported','GPU_cards','GPU_GO','Model_GO','new_native_meta_runs'))
 for owner in (outcome['owner'],outcome['child'],result['owner']):
  current=process(owner['PID']);assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
 pgid=outcome['child']['PGID'];members=[]
 for n in os.listdir('/proc'):
  if n.isdigit():
   row=process(int(n))
   if row and row['PGID']==pgid and row['state']!='Z':members.append(row)
 assert not members
 identity=artifacts['MODEL_IDENTITY.json']['value']
 assert artifacts['MODEL_IDENTITY.json']['SHA256']==result['MODEL_IDENTITY_SHA256']
 assert artifacts['hash-progress.private.jsonl']['SHA256']==result['progress_SHA256']
 events=artifacts['hash-progress.private.jsonl']['value'];assert len(events)==result['progress_events']
 assert events[0]['planned_files']==78 and events[0]['planned_bytes']==33071396964
 assert len([e for e in events if e['stage']=='listed_complete'])==78
 original=json.loads((W/'src/ORIGINAL_FILE_STATS.json').read_bytes());assert original['host']==socket.gethostname()
 binding=json.loads((W/'src/ORIGINAL_BINDING.json').read_bytes())
 assert identity['original_inventory_SHA256']==sha((W/'src/ORIGINAL_FILE_STATS.json').read_bytes())
 assert identity['original_binding_SHA256']==sha((W/'src/ORIGINAL_BINDING.json').read_bytes())
 sys.path.insert(0,str(W/'framework'))
 import importlib.util
 spec=importlib.util.spec_from_file_location('published_identity_reader_helper',W/'framework/tools/hygon_indexed_model_identity.py')
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);header_layout=helper.header_layout
 for component,root,declared,count,total in (
  ('model',Path(original['model_root']),original['files'],77,32901268708),
  ('preset',Path(original['adapter']['path']).parent,{Path(original['adapter']['path']).name:original['adapter']},1,170128256)):
  data=identity[component];assert set(data['files'])==set(declared) and data['completed_files']==count and data['full_file_bytes_read']==total
  assert data['tensor_values_decoded'] is False and data['model_loaded'] is False and data['GPU_calls']==0
  assert sum(v['bytes'] for v in data['files'].values())==total
  canonical={}
  for n,pin in data['files'].items():
   p=root/n;want={k:declared[n][k] for k in ('bytes','mtime_ns','inode','device')}
   assert stats(p)==want=={k:pin[k] for k in want} and not p.is_symlink()
   assert re.fullmatch('[0-9a-f]{64}',pin['SHA256']);canonical[n]=dict(bytes=pin['bytes'],SHA256=pin['SHA256'])
   if n.endswith('.safetensors'):
    with p.open('rb') as stream:
     length_raw=stream.read(8);length=int.from_bytes(length_raw,'little');assert 0<length<=8*1024**2
     raw=stream.read(length);assert len(raw)==length
    header_bytes_read+=8+length;checked_headers+=1
    assert length==pin['header_bytes'] and sha(raw)==pin['header_SHA256']
    assert header_layout(raw,pin['bytes']-8-length)==pin['tensors'] and stats(p)==want
   assert stats(p)==want
  assert sha(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode())==data['canonical_file_manifest_SHA256']
 assert checked_headers==8
 adapter=identity['preset']['files'][Path(original['adapter']['path']).name]
 assert adapter['SHA256']==binding['declared_preset_file_SHA256']
 assert 'torch' not in sys.modules
 assert not any(identity[k] for k in ('historical_training_file_bytes_qualified','official_release_provenance_qualified','actual_model_loading_qualified','actual_preset_fusion_qualified','GPU_GO','Model_GO'))
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:])
answer=dict(passed=failure is None,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,
 failure=failure,artifacts=artifacts,expected_source_files=len(EXPECTED_FILES),header_only_files=checked_headers,header_only_bytes_read=header_bytes_read,
 reader_weight_body_bytes_read=0,new_model_or_CPU_launches=0,GPU_cards=0,GPU_GO=False,Model_GO=False,inside_seconds=time.monotonic()-start,
 qualification='Original executed fullfilehash receipts and source,exactstat,headerlayout,denominator and owner lifecycle checked independently. No second weightbodyhash;no loading,fusion,numerics,training,recovery,performance or effects qualification.')
print(json.dumps(answer))
if not answer['passed']:raise SystemExit(1)

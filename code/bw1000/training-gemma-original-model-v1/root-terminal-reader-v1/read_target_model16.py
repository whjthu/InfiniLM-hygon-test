from pathlib import Path
import hashlib,json,math,os,signal,socket,time
W=Path(PLAN['target_root']);model=W/'gemma-3-4b-it';started=time.monotonic();reader_epoch=time.time();reader_deadline=reader_epoch+600;signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Gemma original targetSHA reader deadline')));signal.setitimer(signal.ITIMER_REAL,595);available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(available[:2]));stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split();reader_owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(stat[19]),host=socket.gethostname());rintent=dict(reader_request_id=READER_REQUEST,original_claim=CLAIM,original_request_id=REQUEST,reader_epoch=reader_epoch,reader_deadline=reader_deadline,owner=reader_owner,expected_files=16,expected_bytes=8639630561,GPU_cards=0)
with (W/'root-model16-reader-intent-v1.json').open('x') as f:json.dump(rintent,f,indent=2)
print(json.dumps(dict(stage='READER_BEGIN',request_id=READER_REQUEST,owner=reader_owner)),flush=True)
def sha_small(p):return hashlib.sha256(p.read_bytes()).hexdigest()
i=json.loads((W/'intent.json').read_bytes());o=json.loads((W/'outcome.json').read_bytes());assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST and i['epoch']==o['epoch']==EPOCH and i['deadline']==o['deadline']==DEADLINE==EPOCH+43200;assert o==EXPECTED_RECEIVER and o['passed'] and o['files']==PLAN['files'] and o['weight_bytes']==8600277880 and o['total_file_bytes']==8639630561 and o['metadata_locally_reused_bytes']==39352681 and o['model_directory']==str(model) and o['GPU_cards']==0 and not o['model_loaded'] and not o['Model_GO'];assert sorted(p.name for p in model.iterdir())==sorted(PLAN['files']);actual={};headers={};tensor_map={};payload_total=0;param_total=0
progress=(W/'root-model16-reader-progress-v1.jsonl').open('x')
for name,pin in sorted(PLAN['files'].items()):
 p=model/name;assert p.is_file() and not p.is_symlink();before=p.stat();assert before.st_size==pin['bytes'];digest=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   assert time.time()<reader_deadline;raw=f.read(4*1024**2)
   if not raw:break
   digest.update(raw)
 after=p.stat();assert (before.st_size,before.st_ino,before.st_mtime_ns,before.st_dev)==(after.st_size,after.st_ino,after.st_mtime_ns,after.st_dev) and digest.hexdigest()==pin['SHA256'];actual[name]=dict(bytes=before.st_size,SHA256=digest.hexdigest(),stable_during_read=True);progress.write(json.dumps(dict(reader_request_id=READER_REQUEST,file=name,identity=actual[name],stat=dict(bytes=before.st_size,inode=before.st_ino,mtime_ns=before.st_mtime_ns,device=before.st_dev),completed_at_epoch=time.time()))+'\n');progress.flush();os.fsync(progress.fileno())
 if name in PLAN['weights']:
  with p.open('rb') as f:hlen=int.from_bytes(f.read(8),'little');assert 0<hlen<2*1024**2;header=f.read(hlen)
  binding=SOURCE_HEADERS[name];assert hlen==binding['header_bytes'] and hashlib.sha256(header).hexdigest()==binding['header_SHA256'];entries=json.loads(header);entries.pop('__metadata__',None);assert len(entries)==binding['tensor_count'];ranges=[];numel=0
  for key,value in entries.items():
   assert value['dtype']=='BF16' and key not in tensor_map and all(type(d) is int and d>0 for d in value['shape']);lo,hi=value['data_offsets'];count=math.prod(value['shape']);assert type(lo) is int and type(hi) is int and hi-lo==count*2 and lo>=0;tensor_map[key]=name;ranges.append((lo,hi));numel+=count
  ranges.sort();assert ranges[0][0]==0 and all(left[1]==right[0] for left,right in zip(ranges,ranges[1:])) and ranges[-1][1]==before.st_size-8-hlen;payload_total+=ranges[-1][1];param_total+=numel;headers[name]=dict(header_bytes=hlen,header_SHA256=hashlib.sha256(header).hexdigest(),tensor_count=len(entries),dtype='BF16',parameter_count=numel,weight_payload_bytes=ranges[-1][1])
progress.close()
index=json.loads((model/'model.safetensors.index.json').read_bytes());assert index['weight_map']==tensor_map and len(tensor_map)==883 and index['metadata']['total_size']==payload_total and payload_total==param_total*2;assert len(actual)==16 and sum(v['bytes'] for v in actual.values())==8639630561;cfg=json.loads((model/'config.json').read_bytes());assert cfg['model_type']=='gemma3';alive=None
if o['owner']['host']==socket.gethostname():
 try:s=Path('/proc',str(o['owner']['PID']),'stat').read_text().split(') ',1)[1].split();alive=int(s[19])==o['owner']['startticks']
 except FileNotFoundError:alive=False
 assert alive is False
answer=dict(target_independent_files_verified=True,reader_request_id=READER_REQUEST,original_claim=CLAIM,original_request_id=REQUEST,original_outcome=o,reader_owner=reader_owner,original_owner_alive=alive,reader_epoch=reader_epoch,reader_deadline=reader_deadline,reader_wall_seconds=time.monotonic()-started,verified_files=actual,weight_headers=headers,total_tensor_count=883,total_parameter_count=param_total,weight_payload_bytes=payload_total,index_exactly_covers_headers=True,total_model_file_bytes=8639630561,actual_weight_bytes_rehashed=8600277880,actual_total_bytes_rehashed=8639630561,model_directory=str(model),model_type=cfg['model_type'],architectures=cfg.get('architectures'),original_artifacts={n:dict(bytes=(W/n).stat().st_size,SHA256=sha_small(W/n)) for n in ['intent.json','outcome.json']},GPU_cards=0,Torch_imported=False,model_loaded=False,model_files_READY=True,Model_GO=False)
with (W/'root-model16-independent-file-sha-v1.json').open('x') as f:json.dump(answer,f,indent=2)
print(json.dumps(answer))

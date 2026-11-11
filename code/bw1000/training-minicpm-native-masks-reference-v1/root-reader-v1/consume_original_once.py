"""CPU-only independent consumers under one shared finite epoch."""
import base64,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1');helpers=ROOT/'root-mask-reader-v1';helpers.mkdir(mode=0o700)
original=(ROOT/'outcome.json').read_bytes();outcome=json.loads(original)
assert hashlib.sha256(original).hexdigest()=='0ea9b880aaa617be9daef8c1383ac817813bcc4c8702d5c6e9f2011e306a30ad'
assert outcome['native_returncode']==0 and outcome['keeper_passed'] and outcome['mask_rank_records']==8
installed=json.loads((ROOT/'installed.json').read_bytes());oldreader=ROOT/'src/read_reference.py';assert hashlib.sha256(oldreader.read_bytes()).hexdigest()==installed['files']['read_reference.py']
for name,item in READERS.items():
 new=base64.b64decode(item['b64'],validate=True);assert hashlib.sha256(new).hexdigest()==item['SHA256']
 assert name in ('read_native_masks.py','read_native_post.py')
 with (helpers/name).open('xb') as f:f.write(new)
epoch=time.time();origin=time.monotonic();deadline=epoch+720
with (helpers/'intent.json').open('x') as f:json.dump({'epoch':epoch,'deadline':deadline,'outcome_SHA256':hashlib.sha256(original).hexdigest(),'new_readers_SHA256':{n:x['SHA256'] for n,x in READERS.items()},'original_reader_SHA256':installed['files']['read_reference.py'],'GPU_allocation':0},f)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');rows=[]
for name,path in [('updates',oldreader),('masks',helpers/'read_native_masks.py'),('post',helpers/'read_native_post.py')]:
 forced=False
 with (helpers/(name+'.log')).open('xb') as log:
  p=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  values=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split();owner={'PID':p.pid,'PGID':int(values[2]),'startticks':int(values[19])}
  with (helpers/(name+'-owner.json')).open('x') as f:json.dump(owner,f)
  budget=min(deadline-time.time(),720-(time.monotonic()-origin))-15;assert budget>0
  try:p.wait(timeout=min(610,budget))
  except subprocess.TimeoutExpired:
   forced=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
 data=(helpers/(name+'.log')).read_bytes();rows.append({'name':name,'returncode':p.returncode,'forced_cleanup':forced,'log_SHA256':hashlib.sha256(data).hexdigest(),'tail':data.decode(errors='replace').splitlines()[-8:] if p.returncode else None,'owner':owner})
 if p.returncode:break
value={'epoch':epoch,'deadline':deadline,'wall_seconds':time.monotonic()-origin,'readers':rows,'GPU_allocation':0,'all_passed':len(rows)==3 and all(x['returncode']==0 and not x['forced_cleanup'] for x in rows)}
for name in ('independent-reference.json','independent-native-mask-reference.json','independent-native-checkpoints-and-post.json'):
 p=ROOT/name
 if p.exists():
  data=p.read_bytes();v=json.loads(data);value[name]={'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest(),'summary':{k:x for k,x in v.items() if k not in ('ranks','tensor_artifacts','checkpoints','tensor_files')}}
with (helpers/'outcome.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps({k:v for k,v in value.items() if k not in ('independent-reference.json','independent-native-checkpoints-and-post.json')}))

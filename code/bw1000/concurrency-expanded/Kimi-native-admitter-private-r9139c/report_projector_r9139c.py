import pathlib,json,os,time,datetime,socket,hashlib,sys
from selection_r9139 import summarize
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');inc=pathlib.Path(__file__).resolve().parent;cfg=json.loads((inc/'config-r9139.json').read_text());eos=set(json.loads((base/cfg['eos_relative']).read_text())['eos_token_ids']);host=socket.gethostname();node=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'].index(host);assert os.environ['SLURM_JOB_ID']=='910380';step='910380.'+os.environ['SLURM_STEP_ID'];src=base/cfg['reports'][node];out=base/'results'/('Kimi-admitter-projection-r9139c-node%d.json'%node);assert not out.exists();cache={};started=time.monotonic();counter=0
while time.monotonic()-started<1800:
 tick=time.monotonic();counter+=1
 try:
  with src.open('rb',buffering=0)as f:
   os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_RANDOM);parts=[]
   while True:
    v=f.read(65536)
    if not v:break
    parts.append(v)
  raw=b''.join(parts);doc=json.loads(raw);assert doc['node']==node;cases=[]
  for case in doc.get('batch_cases',[]):
   if node==2:
    label=case['label']
    if label not in cache:cache[label]=summarize(case,eos)
    cases.append({'label':label,'admitter_summary':cache[label]})
   else:cases.append({k:case[k]for k in ['label','state_stats_after64','I192_counts_delta8x2']})
  q={k:doc.get(k)for k in ['node','phase','active_batch_progress','error','shutdown_error','command_rejections']};q['batch_cases']=cases;q['_projection']={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'CPU_step':step,'PID':os.getpid(),'counter':counter,'source_SHA256':hashlib.sha256(raw).hexdigest(),'source_mtime':src.stat().st_mtime,'source_size':len(raw),'read_project_s':time.monotonic()-tick,'noGPUruntime':True};tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,ensure_ascii=False));tmp.replace(out)
 except (OSError,ValueError,KeyError,AssertionError,TypeError)as e:print(repr(e),flush=True)
 time.sleep(max(.2,5-(time.monotonic()-tick)))

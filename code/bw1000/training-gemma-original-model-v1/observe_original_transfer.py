from pathlib import Path
import concurrent.futures,ctypes,ctypes.wintypes,hashlib,json,runpy,subprocess,time
B=Path(__file__).parent;D=B/'training-gemma-original-model-v1';assert not (D/'first-observation.json').exists();intent=json.loads((D/'transfer-intent.json').read_bytes());owned=json.loads((D/'owned-local-transports.json').read_bytes());helper=runpy.run_path(str(D/'package/transfer_once.py'),run_name='owner_observation')['winowner']
def live(value):
 try:
  actual=helper(value['PID']);same=actual['startcreationticks']==value['startcreationticks'];k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.wintypes.DWORD,ctypes.wintypes.BOOL,ctypes.wintypes.DWORD];k.OpenProcess.restype=ctypes.wintypes.HANDLE;handle=k.OpenProcess(0x1000,False,value['PID']);exitcode=ctypes.wintypes.DWORD();k.GetExitCodeProcess.argtypes=[ctypes.wintypes.HANDLE,ctypes.POINTER(ctypes.wintypes.DWORD)];k.CloseHandle.argtypes=[ctypes.wintypes.HANDLE]
  try:assert k.GetExitCodeProcess(handle,ctypes.byref(exitcode))
  finally:k.CloseHandle(handle)
  return dict(**value,owned_start_matches=same,live=same and exitcode.value==259,exitcode=exitcode.value)
 except OSError:return dict(**value,live=False,owned_start_matches=None)
local={n:live(owned[n]) for n in ('driver','sender','receiver')};assert all(v['live'] for v in local.values()),'Original worker not live; reconcile exact outcome,never replay'
remote='''from pathlib import Path
import datetime,hashlib,json,os,socket
W=Path(ROOT);out=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),root_exists=W.exists())
for name in ['intent.json','progress.json','outcome.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=100000;out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
intent=out.get('intent.json',{}).get('value',{});assert intent.get('request_id')==REQUEST and intent.get('claim')==CLAIM and intent.get('epoch')==EPOCH and intent.get('deadline')==DEADLINE
owner=intent['owner'];out['owner_HOST_matches']=owner['host']==socket.gethostname();out['owned_PID_live']=None
if out['owner_HOST_matches']:
 p=Path('/proc',str(owner['PID']),'stat')
 if p.exists():
  fields=p.read_text().split(') ',1)[1].split();out['owned_PID_live']=int(fields[19])==owner['startticks'] and fields[0]!='Z'
 else:out['owned_PID_live']=False
if PLATFORM=='target':
 model=W/'gemma-3-4b-it';out['weight_file_sizes']={}
 for name in NAMES:
  for suffix in ['', '.partial']:
   p=model/(name+suffix)
   if p.exists():out['weight_file_sizes'][name+suffix]=p.stat().st_size
print(json.dumps(out))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns);A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');a=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];names=list(json.loads((D/'package/MODEL_FILES.json').read_bytes())['weights'])
def read(label,argv,root):
 code='ROOT='+repr(root)+'\nPLATFORM='+repr(label)+'\nNAMES='+repr(names)+'\nREQUEST='+repr(intent['request_id'])+'\nCLAIM='+repr(intent['claim'])+'\nEPOCH='+repr(intent['epoch'])+'\nDEADLINE='+repr(intent['deadline'])+'\n'+remote
 r=subprocess.run(argv,input=code.encode(),capture_output=True,timeout=25);(D/(label+'-first-observe-stdout.private.json')).write_bytes(r.stdout);(D/(label+'-first-observe-stderr.private.txt')).write_bytes(r.stderr);assert r.returncode==0,label;return label,json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(read,'source',a,'/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-original-model-v1'),pool.submit(read,'target',ns['ssh'],'/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-original-model-v1')];remotes=dict(f.result() for f in fs)
result=dict(local_owned_processes=local,remotes=remotes,claim=intent['claim'],request_id=intent['request_id'],epoch=intent['epoch'],deadline=intent['deadline'],transfer_terminal=False,GPU_cards=0,Model_GO=False);(D/'first-observation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(local_all3_owned_live=True,source_owner=remotes['source']['intent.json']['value']['owner'],target_owner=remotes['target']['intent.json']['value']['owner'],source_PID_live=remotes['source']['owned_PID_live'],target_PID_live=remotes['target']['owned_PID_live'],target_HOST_matches=remotes['target']['owner_HOST_matches'],target_weight_sizes=remotes['target'].get('weight_file_sizes'),source_progress=remotes['source'].get('progress.json',{}).get('value',{}).get('total_weight_bytes_read'),target_progress=remotes['target'].get('progress.json',{}).get('value',{}).get('total_weight_bytes_written'),epoch=intent['epoch'],deadline=intent['deadline'])))

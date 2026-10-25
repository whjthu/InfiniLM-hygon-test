"""Called by detached CPU coordinator; holds no Slurm CPU/GPU allocation."""
import pathlib,json,time,subprocess,hashlib
TARGET_HANDLE='910375.678'
TARGET_FILE='kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json'
def wait_ready(root,persist):
 source=root/'results'/TARGET_FILE;deadline=time.monotonic()+50*60;reads=0;last_error=None
 while time.monotonic()<deadline:
  ready=False
  try:
   raw=source.read_bytes();packet=json.loads(raw);A=[x for x in packet.get('crossover_arms',[])if x['arm']=='A'];calls=[x for x in packet.get('crossover_calls',[])if x['arm']=='A']
   if A and len(A[0]['anchor'])+len(A[0]['actual_ids'])==33 and len(calls)==33 and packet['node']==2:
    owners=calls[-1]['owners'];covered={l:set()for l in [19,37,66,78,90]}
    for node in [0,2]:
     trace=next(x for x in owners if x['node']==node)['trace_after'];assert trace['arm']=='A'and trace['finite'];layers={l:set()for l in [19,37,66,78,90]}
     for rec in trace['records']:
      if rec['point']!='decoder' or rec['layer']not in layers:continue
      assert rec['shape']==[1,len(rec['positions']),7168]and rec['bytes']==len(rec['positions'])*7168*2
      remote=pathlib.Path(rec['remote_owned_buffer']);assert str(remote).startswith('/refactor/results/Kimi-crossover-private-r9026-910375.678/'),str(remote)
      path=root/pathlib.Path(str(remote)[len('/refactor/'):]);assert path.is_file()and path.stat().st_size==rec['bytes'];layers[rec['layer']].update(rec['positions'])
     for layer,positions in layers.items():covered[layer].update(positions)
    ready=all(set(range(536))<=covered[l]for l in [19,37,66,78,90])
    if ready:return {'ready':True,'target_handle':TARGET_HANDLE,'target_report':str(source),'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),'reads':reads,'no_Slurm_allocation_while_waiting':True}
  except(FileNotFoundError,json.JSONDecodeError,KeyError,StopIteration,AssertionError)as e:last_error=repr(e)
  try:
   rows=[x.split('|')for x in subprocess.check_output(['sacct','-j',TARGET_HANDLE,'-n','-P','-o','JobID,State,ExitCode'],text=True,timeout=20).splitlines()if x.startswith(TARGET_HANDLE+'|')];assert len(rows)<=1
   if rows and rows[0][1].split()[0].rstrip('+')in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}:
    return {'ready':False,'target_terminal_without_A_ready':rows[0],'last_read_error':last_error,'no_GPU_submitted':True}
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError)as e:last_error=repr(e)
  reads+=1;persist({'state':'waiting_actual_A','target_handle':TARGET_HANDLE,'reads':reads,'last_read_error':last_error,'no_Slurm_allocation_while_waiting':True});time.sleep(15)
 return {'ready':False,'wait_limit_seconds':3000,'last_read_error':last_error,'no_GPU_submitted':True}

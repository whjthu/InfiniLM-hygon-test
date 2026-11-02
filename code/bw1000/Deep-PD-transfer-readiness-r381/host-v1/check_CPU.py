"""Actual loopback execute, failed staging and original deadline; no cluster/Native/GPU."""
from pathlib import Path
import socket,tempfile,threading,time,json,hashlib,subprocess
import worker,CPU_main

def pair(root,run,mf,nonce,deadline,wrong=False):
 probe=socket.socket();probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close();errors=[];results=[]
 def call(role):
  try:results.append(worker.execute(role,run,root,['127.0.0.1','127.0.0.1'],port,deadline,nonce,'LOCAL_RECORDING_ONLY',mf,inject_wrongpattern=wrong))
  except BaseException as e:errors.append((role,type(e).__name__))
 threads=[threading.Thread(target=call,args=(r,))for r in ['D','P']]
 for t in threads:t.start()
 for t in threads:t.join(10);assert not t.is_alive()
 return results,errors

def main():
 d=Path(__file__).resolve().parent;mf=CPU_main.sha(d/'SOURCE_MANIFEST.json');cases=[]
 with tempfile.TemporaryDirectory(prefix='deep381-local-')as temp:
  root=Path(temp);(root/'results').mkdir();(root/'control').mkdir()
  calls=[]
  def recording(args,**kw):
   calls.append(args)
   if args[0]=='scontrol':return subprocess.CompletedProcess(args,0,'JobState=RUNNING UserId=aclsylqidf(154459) NodeList=a02r4n[12-15]','')
   assert args[0]=='srun'and '--gres=none'in args and '-N2'in args and '-n2'in args and '--mem=4096M'in args
   env=kw['env'];values,errors=pair(root,env['DEEP_R381_CPU_RUN'],mf,env['DEEP_R381_CPU_NONCE'],float(env['DEEP_R381_CPU_DEADLINE']))
   assert len(values)==2 and not errors
   return subprocess.CompletedProcess(args,0,'local recording executes actual TCP consumer; no actual Slurm','')
  q=CPU_main.execute(root,d,mf,'a'*40,'cpu-local-success',recording)
  assert q['passed']and q['actual_CPU']=='LOCAL_RECORDING_ONLY'and len(calls)==2
  P,D=[next(v['data']for v in q['reports']if v['data']['role']==r)for r in ['P','D']]
  assert D['staged_shards']==8 and D['whole_ticket_host_publish_count']==1 and D['all_host_staging_released']and not D['Native_verify_called']and P['all8shards_publish_ACK_received']
  assert P['transfer']['block_SHA256']==D['transfer']['block_SHA256']
  cases.append('actual CPUentry sourceverify/scontrol/srunargv/env -> actual loopback full8shardstage/checksum/hostverify/publish/ACK -> bothcleanup')
  values,errors=pair(root,'cpu-local-corrupt',mf,'n'*32,time.time()+5,wrong=True)
  assert not values and len(errors)==2
  fail=json.loads((root/'results/Deep381-hostTCP-cpu-local-corrupt-D.json').read_bytes());src=json.loads((root/'results/Deep381-hostTCP-cpu-local-corrupt-P.json').read_bytes())
  assert fail['sink_aborted']and fail['all_host_staging_released']and fail['publish_count_on_exit']==0 and not src.get('all8shards_publish_ACK_received',False)
  cases.append('corrupt sourcepattern -> complete footer cannot bypass hostverify -> no publish/ACK and bothrolescleanup')
  try:worker.execute('P','cpu-local-expired',root,['127.0.0.1']*2,port=49181,deadline_unix=time.time()-1,nonce='n'*32,actual_step='LOCAL_RECORDING_ONLY',source_SHA=mf)
  except AssertionError:pass
  else:raise AssertionError('Expired original deadline accepted')
  fail=json.loads((root/'results/Deep381-hostTCP-cpu-local-expired-P.json').read_bytes());assert fail['error_phase']=='before_socket'and fail['normal_socket_close']
  cases.append('original deadline rejection before any socket')
  try:CPU_main.execute(root,d,mf,'a'*40,'cpu-local-success',recording)
  except AssertionError:pass
  else:raise AssertionError('Duplicate oneuseintent accepted')
  cases.append('duplicate intent rejects before second srun')
 print(json.dumps(dict(passed=True,cases=cases,source_manifest_SHA256=mf,GPU=0,actual_Slurm=False,actual_Native=False,real_loopback_TCP=True)))
if __name__=='__main__':main()

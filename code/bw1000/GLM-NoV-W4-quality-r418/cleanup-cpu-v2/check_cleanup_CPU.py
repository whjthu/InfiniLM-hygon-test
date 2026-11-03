"""Changed failure cleanup only: actual POSIX Popen groups, recorded Slurm leaf."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from unittest.mock import patch

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();assert os.name=='posix'
 D=Path(__file__).resolve().parent.parent/'execution-v2';spec=importlib.util.spec_from_file_location('actual418controllerCleanup',D/'controller.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
 rows=[];calls=[]
 out=Path(a.output);assert not out.exists()
 def save_partial(q):
  out.write_text(json.dumps(dict(completed=False,passed=False,case_records=rows,current_case=q,Native=0,GPU=0,Slurm=0),indent=2)+'\n')
 def cancel(args,**kwargs):
  assert args[0]=='scancel'and args[1]in('--signal=TERM','--signal=KILL')and args[2]=='910376.999999';calls.append(args);return subprocess.CompletedProcess(args,0,'recording Slurm leaf; no real scancel','')
 for name,ignore,unknown in [('TERM_normal',False,False),('transient_recorded_handle_KILL',True,False),('unknown_terminal_stays_failed',True,True)]:
  code='import signal,time,sys\n'+('signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'if ignore else'')+'print("ready",flush=True)\ntime.sleep(60)\n'
  proc=subprocess.Popen([sys.executable,'-I','-B','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,text=True)
  assert proc.stdout.readline().strip()=='ready';before=len(calls);observations=0
  def exact(label):
   nonlocal observations
   observations+=1
   if unknown or observations==1:raise subprocess.TimeoutExpired('recorded transient sacct',.01)
   return ['910376.999999',label,'CANCELLED'if proc.poll()is not None else'RUNNING','0:9','a02r4n12']
  try:
   with patch.object(C.subprocess,'run',side_effect=cancel):q=C.close_owned(name,proc,'910376.999999',exact,time.monotonic()+5,term_s=1.2,kill_s=2)
   save_partial(dict(name=name,evidence=q,actual_POSIX_Popen=True,Slurm_leaf_recorded=True));proc.wait(timeout=2);assert not q['PGID_still_live']and proc.poll()is not None
   seen=calls[before:];assert seen[0][1]=='--signal=TERM';assert bool(any(v[1]=='--signal=KILL'for v in seen))==ignore
   assert q['closed']==(not unknown)and q['terminal_unknown_or_unclosed']==unknown
   assert q['recorded_step']==q['actual_step']=='910376.999999'and any('observation_error'in x for x in q['observations'])
   rows.append(dict(name=name,passed=True,evidence=q,actual_POSIX_Popen=True,Slurm_leaf_recorded=True))
  finally:
   if proc.poll()is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 # Root/owner identity stays separate; only the execution controller/CLI changes.
 mf=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert mf['loader_manifest_SHA256']=='439cdee25fa8a0a3070b1aec8ef06601f176476a97028c56a46e3b8a8a47eddd'
 text=(D/'launch_once.py').read_text();assert "str(D/'controller.py')"in text and"if a.preflight_only:"in text
 result=dict(completed=True,passed=True,controller_SHA256=hashlib.sha256((D/'controller.py').read_bytes()).hexdigest(),cases=rows,actual_POSIX_Popen_groups=3,Native=0,GPU=0,Slurm=0,weights=0,old_encoding_and_HTTP_CPU_repeated=False)
 out=Path(a.output)
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(completed=True,passed=True,receipt_path=str(out),SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),Native=0,GPU=0,Slurm=0)))
if __name__=='__main__':main()

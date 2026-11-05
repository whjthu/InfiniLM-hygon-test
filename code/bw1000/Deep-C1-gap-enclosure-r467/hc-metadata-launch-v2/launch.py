"""Same GPUcontroller, one50s absolute HOSTcheck budget with persistentphases."""
from pathlib import Path
import argparse,hashlib,json,os,re,socket,subprocess,sys,time
D=Path(__file__).resolve().parent;R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');P=D.parent/'hc-metadata-GPU-v1';LABEL='Deep-HC-metadata-r467-device-v1';H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def persist(path,q):
 tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(q,indent=2)+'\n');os.replace(tmp,path)
def checked(argv,state,path,deadline):
 left=deadline-time.monotonic();assert left>0,'HOST50s exhausted';state['phase']='before_command';state['current_argv']=argv;state['remaining_s']=left;persist(path,state)
 try:
  z=subprocess.run(argv,capture_output=True,text=True,timeout=left);state['phase']='after_command';state['commands'].append(dict(argv=argv,rc=z.returncode,stdout=z.stdout[:32768],stderr=z.stderr[:4096]));persist(path,state);assert z.returncode==0,(argv,z.returncode,z.stderr[-1000:]);return z.stdout
 except subprocess.TimeoutExpired as e:state['phase']='command_timeout';state['timeout_argv']=argv;persist(path,state);raise

def main(argv=None):
 start=time.monotonic();deadline=start+50;pa=argparse.ArgumentParser();pa.add_argument('--manifest-sha',required=True);a=pa.parse_args(argv);assert H(D/'SOURCE_MANIFEST.json')==a.manifest_sha;m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items());binding=json.loads((D/'PARENT_BINDING.json').read_bytes());assert all(H(P/n)==h for n,h in binding['files_SHA256'].items());assert H(P/'GPU_MANIFEST.json')==binding['manifest_SHA256'];sys.path.insert(0,str(P));import support
 support.verify();phase=R/'results'/(LABEL+'-HOST-v2-phases.json');assert not phase.exists(),'Existingphase reconcile only';state=dict(HOST=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=Path('/proc/self/stat').read_text().split()[21],phase='verified_source_CPU',commands=[],budget_seconds=50,started_unix=time.time(),completed=False,GPU_submitted=False);persist(phase,state)
 try:
  for parent,node in [('910374','a02r4n04'),('910407','a14r2n20')]:
   text=checked(['scontrol','show','job',parent,'-o'],state,phase,deadline);assert'JobState=RUNNING'in text and'UserId=aclsylqidf('in text;spec=re.search(r'(?:^| )NodeList=([^ ]+)',text).group(1);nodes=checked(['scontrol','show','hostnames',spec],state,phase,deadline).splitlines();assert node in nodes
  rows=checked(['sacct','-S','2026-11-05','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State'],state,phase,deadline);assert not any(LABEL in x for x in rows.splitlines());assert not(R/'results'/LABEL).exists();assert not list((R/'logs').glob(LABEL+'*'));existing=[p.name for p in(R/'results').glob(LABEL+'*')if p!=phase];assert not existing,existing;assert deadline-time.monotonic()>5,'Insufficient originalHOSTbudget to submit safely'
  args=['srun','--jobid=910407','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c2','--mem=4096M','--gres=none','-w','a14r2n20','-t','00:03:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'python3','-B','-u',str(P/'controller.py')];intent=R/'results'/(LABEL+'-intent.json');state['phase']='before_exclusive_intent';state['keeper_argv']=args;persist(phase,state)
  with intent.open('x')as f:json.dump(dict(origin=state,parent_binding=binding,argv=args,GPU_epoch=False),f,indent=2);f.flush();os.fsync(f.fileno())
  with(R/'logs'/(LABEL+'-keeper-srun.log')).open('xb')as log:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch=dict(owned_keeper_PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],intent=str(intent),GPU=None);persist(R/'results'/(LABEL+'-launch.json'),launch);state.update(phase='keeper_submitted',GPU_submitted=False,keeper=launch);print(json.dumps(launch))
 except BaseException as e:state['error']=repr(e);raise
 finally:state.update(completed=True,elapsed_s=time.monotonic()-start);persist(phase,state)
if __name__=='__main__':main()

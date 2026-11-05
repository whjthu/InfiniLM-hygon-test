"""Source/CPU-bound single keeper intent; GPU only afterkeeper fresh."""
from pathlib import Path
import json,os,re,socket,subprocess,time
from support import D,R,LABEL,NODE,OUT,save,verify

def main():
 m,q=verify()
 if '--source-only'in __import__('sys').argv:print(json.dumps(dict(source_verified=True,CPU=q['actual_CPU'],Runtime=False,GPU=False,Model=False)));return
 for p in ['inside.sh','outer.sh']:subprocess.run(['bash','-n',str(D/p)],check=True,timeout=5)
 for parent,node in [('910374',NODE),('910407','a14r2n20')]:
  text=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=15);assert'JobState=RUNNING'in text and'UserId=aclsylqidf('in text;members=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',text).group(1)],text=True,timeout=10).splitlines();assert node in members
 z=subprocess.check_output(['sacct','-S','2026-11-05','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State'],text=True,timeout=20);assert not any(LABEL in x for x in z.splitlines());assert not OUT.exists()and not list((R/'results').glob(LABEL+'*'))and not list((R/'logs').glob(LABEL+'*'))
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c2','--mem=4096M','--gres=none','-w','a14r2n20','-t','00:03:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'python3','-B','-u',str(D/'controller.py')];intent=R/'results'/(LABEL+'-intent.json');origin=dict(HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],created=time.time(),source=m,argv=args,GPU_epoch=False)
 with intent.open('x')as f:json.dump(origin,f,indent=2);f.flush();os.fsync(f.fileno())
 with(R/'logs'/(LABEL+'-keeper-srun.log')).open('xb')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(owned_keeper_PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],intent=str(intent),GPU=None);save(R/'results'/(LABEL+'-launch.json'),q);print(json.dumps(q))
if __name__=='__main__':main()

"""Exclusive source/CPU-bound intent; no GPU epoch until keeper fresh passes."""
from pathlib import Path
import argparse,json,os,re,socket,subprocess,time
from support import D,R,RUN,LABEL,NODE,OUTPUT,clean_environment,publish,verify

def keeper_command(digest,commit):
 return ['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:50:00','--job-name='+LABEL+'-'+RUN+'-keeper','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-keeper-%J.log')),'python3','-B','-u',str(D/'controller.py'),'--manifest-sha',digest,'--publication-commit',commit]

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args(argv);assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha,require_ready=not a.source_only)
 if a.source_only:print(json.dumps(dict(source_verified=True,execution_READY=m['execution_READY'],CPU_pending=m['entry_CPU']is None,Slurm=0,GPU=0,Native_dlopen=False)));return 0
 for n in ['inside.sh','resource_wave_host_alias_outer.sh']:subprocess.run(['bash','-n',str(D/n)],check=True,timeout=10)
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-11-04','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()if'|'in s];assert not any(r[1].startswith(LABEL+'-'+RUN)for r in rows),'Original label exists; reconcile only'
 for parent,node in [('910374',NODE),('910407','a14r2n20')]:
  text=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25);assert 'JobState=RUNNING'in text and re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',text);membership=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',text).group(1)],text=True,timeout=20).splitlines();assert node in membership
 prefix=R/'results'/(LABEL+'-'+RUN);assert not OUTPUT.exists()and not list((R/'results').glob(prefix.name+'*'))and not list((R/'logs').glob(prefix.name+'*')),'Existing original intent/log/report forbids replay'
 intent=prefix.with_name(prefix.name+'-intent.json');argv=keeper_command(a.manifest_sha,a.publication_commit)
 origin=dict(HOST=socket.gethostname(),UID=os.getuid(),PID=os.getpid(),PGID=os.getpgrp(),startticks=Path('/proc/self/stat').read_text().split()[21])
 publish(intent,dict(origin=origin,publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,entry_source_manifest_SHA256=m['entry_source_manifest_SHA256'],entry_CPU=m['entry_CPU'],created_unix=time.time(),keeper_argv=argv,GPU_epoch_created=False,planned_cards=8,requests=5,observer_schedule=[False,False,False,True,False],samples=[8,87],HTTP=0,retry=False,reload=False,one_attempt=True))
 with(R/'logs'/(LABEL+'-'+RUN+'-keeper-srun.log')).open('x')as log:proc=subprocess.Popen(argv,env=clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,owned_startticks=Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21],origin=origin,intent_path=str(intent),GPU_epoch_created=False,actual_GPU=None,CPU_keeper=None)
 publish(prefix.with_name(prefix.name+'-launch.json'),q);print(json.dumps(q));return 0
if __name__=='__main__':raise SystemExit(main())

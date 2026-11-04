"""One bounded independent fixed slot: 64HTTP, no retry or redistribution."""
from pathlib import Path
import argparse,json,os,re,subprocess,time
import controller as C
from controller import R,D,PACKAGE,LABEL,verify,verify_gpu,clean_environment,publish

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--slot',type=int,required=True);p.add_argument('--gpu-manifest-sha');a=p.parse_args();slot=C.configure_slot(a.slot);RUN=C.RUN;NODES=C.NODES;PARENT=C.PARENT
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha)
 # Actual final public CLI/config/assets, no serving imports or GPU work.
 child=subprocess.Popen(['python3','-B',str(D/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--preflight-only','--slot',str(a.slot)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:stdout,stderr=child.communicate(timeout=90)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate(timeout=10)
  print(json.dumps(dict(phase='actualHOST_controller_wholeCLI',child_PID=child.pid,timeout_s=90,stdout=stdout,stderr=stderr,intent_created=False,Slurm_submitted=0)),flush=True)
  raise
 assert child.returncode==0,dict(phase='actualHOST_controller_wholeCLI',child_PID=child.pid,stdout=stdout,stderr=stderr)
 proof=json.loads(stdout);assert proof['passed']and proof['GPU']==proof['Slurm']==0
 assert verify_gpu(a.gpu_manifest_sha,m)
 for name in ['inside.sh','resource_wave_host_alias_outer.sh']:subprocess.run(['bash','-n',str(D/name)],check=True,timeout=10)
 rows=[r.split('|')for r in subprocess.check_output(['sacct','-S','2026-10-31','-j',str(PARENT)+',910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()if'|'in r]
 assert not any(r[1].startswith(LABEL+'-'+RUN)for r in rows),'Existing label must follow original handle'
 for parent,nodes in [(str(PARENT),NODES),('910407',['a14r2n20'])]:
  raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25)
  assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw
  members=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)],text=True,timeout=20).splitlines();assert set(nodes)<=set(members)
 intent=R/'results'/(LABEL+'-'+RUN+'-intent.json');watch=R/'results'/(LABEL+'-'+RUN+'-watch.json')
 assert not intent.exists()and not watch.exists()and not(R/'results/GLM-W4-fewshot-quality-r444'/RUN).exists()
 assert not list((R/'logs').glob(LABEL+'-'+RUN+'*')),'Original logs prohibit replay'
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','09:00:00','--job-name='+LABEL+'-'+RUN+'-keeper',
  '--output='+str(R/'logs'/(LABEL+'-'+RUN+'-keeper-%J.log')),'python3','-B','-u',str(D/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--slot',str(a.slot),'--gpu-manifest-sha',a.gpu_manifest_sha or '']
 q=dict(publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,preflight_passed=True,actual_public_CLI=proof,
  keeper_argv=argv,intent_created=False,GPU_epoch_created=False,Slurm_submitted=0,planned_cards=16,HTTP=64,global_HTTP=512,slot=a.slot,shards=8,resource_claim=slot,warms=0,retry=False,reload=False)
 if a.preflight_only:print(json.dumps(q));return
 publish(intent,dict(**q,created_unix=time.time(),one_attempt=True))
 with(R/'logs'/(LABEL+'-'+RUN+'-keeper-client.log')).open('x')as log:
  proc=subprocess.Popen(argv,env=clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,intent_path=str(intent),publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,no_epoch_until_freshidle=True)))
if __name__=='__main__':main()

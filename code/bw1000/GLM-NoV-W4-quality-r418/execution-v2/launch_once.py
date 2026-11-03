"""One explicit two-replica keeper; full preflight returns before any intent/Popen."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-NoV-W4-quality-r418';D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
from controller import LABEL,RUN,PAIRS
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 from verify_execution import verify
 from check_shell_layout import check
 m=verify(R,S,a.manifest_sha)
 shell=check(D,Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json'),R/'incoming/GLM-NoV-W4-normal-profile-r417',m['loader_manifest_SHA256'],m['profile_manifest_SHA256'])
 rows=[r.split('|')for r in subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()if'|'in r]
 assert not any(r[1].startswith(LABEL+'-'+RUN)for r in rows),'Existinglabel must followoriginalhandle;no retry'
 for parent,nodes in [('910376',sum(PAIRS,[])),('910407',['a14r2n20'])]:
  raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25);assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw
  members=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)],text=True,timeout=20).splitlines();assert set(nodes)<=set(members)
 intent=R/'results'/(LABEL+'-'+RUN+'-intent.json');watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');cancel=R/'control'/(LABEL+'-'+RUN+'-cancel.json')
 assert not any(p.exists()for p in[intent,watch,cancel])and not list((R/'logs').glob(LABEL+'-'+RUN+'*'))
 for i in range(2):
  run=RUN+'-r'+str(i)
  assert not list((R/'results').glob('*'+run+'*'))and not list((R/'control').glob('*'+run+'*')),'Existing run artifacts prohibitreplay'
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','04:10:00','--job-name='+LABEL+'-'+RUN+'-keeper','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-keeper-%J.log')),'python3','-B','-u',str(D/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit]
 result=dict(publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,preflight_passed=True,actual_four_role_Bash=shell,keeper_argv=argv,intent_created=False,GPU_epoch_created=False,Slurm_submitted=0,GPU=0,planned_cards=32,HTTP=128,warms=0)
 if a.preflight_only:print(json.dumps(result));return
 assert m['GPU_GO']and m['Model_GO']and m['resource']['active_claim'],'No actualModelGO inpreparationmanifest'
 with intent.open('x')as f:json.dump(dict(**result,created_unix=time.time(),one_attempt=True,no_retry_reload=True),f,indent=2)
 with(R/'logs'/(LABEL+'-'+RUN+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,srun_PID=proc.pid,intent_path=str(intent),publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,no_epoch_untilfreshidle=True)))
if __name__=='__main__':main()

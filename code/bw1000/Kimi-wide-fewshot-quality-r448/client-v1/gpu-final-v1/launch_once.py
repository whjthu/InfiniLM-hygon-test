"""Reuse the bounded r418 keeper launch; one explicit public256-request intent."""
from pathlib import Path
import argparse,json,os,re,subprocess,time
from controller import R,D,FINAL,PACKAGE,LABEL,RUN,NODES,verify,verify_execution,clean_environment,publish

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--execution-manifest-sha',required=True);a=p.parse_args()
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha);go=verify_execution(a.execution_manifest_sha,m)
 # Actual final public CLI/config/assets, no serving imports or GPU work.
 z=subprocess.run(['python3','-B',str(FINAL/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--execution-manifest-sha',a.execution_manifest_sha,'--preflight-only'],capture_output=True,text=True,timeout=120)
 assert z.returncode==0,(z.stdout,z.stderr);proof=json.loads(z.stdout);assert proof['passed']and proof['GPU']==proof['Slurm']==0
 for name in ['inside.sh','resource_wave_host_alias_outer.sh']:subprocess.run(['bash','-n',str(D/name)],check=True,timeout=10)
 rows=[r.split('|')for r in subprocess.check_output(['sacct','-S','2026-10-31','-j','910380,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()if'|'in r]
 assert not any(r[1].startswith(LABEL+'-'+RUN)for r in rows),'Existing label must follow original handle'
 for parent,nodes in [('910380',NODES),('910407',['a14r2n20'])]:
  raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25)
  assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw
  members=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)],text=True,timeout=20).splitlines();assert set(nodes)<=set(members)
 intent=R/'results'/(LABEL+'-'+RUN+'-intent.json');watch=R/'results'/(LABEL+'-'+RUN+'-watch.json')
 assert not intent.exists()and not watch.exists()and not list((R/'results/public-Kimi-text-provider-r434').glob('*'+RUN+'*'))
 assert not list((R/'logs').glob(LABEL+'-'+RUN+'*')),'Original logs prohibit replay'
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','09:00:00','--job-name='+LABEL+'-'+RUN+'-keeper',
  '--output='+str(R/'logs'/(LABEL+'-'+RUN+'-keeper-%J.log')),'python3','-B','-u',str(FINAL/'controller.py'),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--execution-manifest-sha',a.execution_manifest_sha]
 q=dict(publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,preflight_passed=True,execution_manifest_SHA256=a.execution_manifest_sha,actual_public_CLI=proof,
  keeper_argv=argv,intent_created=False,GPU_epoch_created=False,Slurm_submitted=0,planned_cards=32,HTTP=256,warms=0,retry=False,reload=False)
 if a.preflight_only:print(json.dumps(q));return
 assert go['GPU_GO']and go['Model_GO'];publish(intent,dict(**q,created_unix=time.time(),one_attempt=True))
 with(R/'logs'/(LABEL+'-'+RUN+'-keeper-client.log')).open('x')as log:
  proc=subprocess.Popen(argv,env=clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,intent_path=str(intent),publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,no_epoch_until_freshidle=True)))
if __name__=='__main__':main()

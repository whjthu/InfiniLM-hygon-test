"""One immutable shard keeper; only originalj0 firstpair gates shards1..7."""
from pathlib import Path
import argparse,json,os,re,subprocess,time
import controller as c

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);shard=c.configure(a.shard);m=c.verify(a.manifest_sha)
 cmd=['python3','-B',str(c.D/'controller.py'),'--shard',str(a.shard),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit,'--preflight-only']
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=120);assert z.returncode==0,(z.stdout,z.stderr);proof=json.loads(z.stdout);assert proof['passed']and proof['GPU']==proof['Slurm']==0
 for name in ['inside.sh','resource_wave_host_alias_outer.sh']:subprocess.run(['bash','-n',str(c.D/name)],check=True,timeout=10)
 q=dict(shard=a.shard,publication_commit=a.publication_commit,deployment_manifest_SHA256=a.manifest_sha,preflight_passed=True,actual_public_CLI=proof,intent_created=False,GPU_epoch_created=False,Slurm_submitted=0,planned_cards=8,HTTP=64,warms=0,retry=False,reload=False)
 if a.preflight_only:print(json.dumps(q));return 0
 assert m['GPU_GO']and m['Model_GO']
 if a.shard:
  from first_pair_gate import check
  gate=check(c.R,c.D,m);q['j0_firstpair_gate']=gate
  if not gate['passed']:print(json.dumps(dict(q,gate_pending=True,keeper_submitted=False)));return 0
 rows=[r.split('|')for r in subprocess.check_output(['sacct','-S','2026-10-31','-j',c.GPU_PARENT+',910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()if'|'in r]
 assert not any(r[1].startswith(c.LABEL+'-'+c.RUN)for r in rows),'Original label must follow its handle, never replay'
 for parent,nodes in [(c.GPU_PARENT,c.NODES),('910407',['a14r2n20'])]:
  raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25);assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw
  members=subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)],text=True,timeout=20).splitlines();assert set(nodes)<=set(members)
 intent=c.R/'results'/(c.LABEL+'-'+c.RUN+'-intent.json');watch=c.R/'results'/(c.LABEL+'-'+c.RUN+'-watch.json')
 assert not intent.exists()and not watch.exists()and not(c.R/'results/Deep-broad-fewshot-fullpre-quality-r450'/c.RUN).exists()
 assert not list((c.R/'logs').glob(c.LABEL+'-'+c.RUN+'*'))
 command=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','09:00:00','--job-name='+c.LABEL+'-'+c.RUN+'-keeper','--output='+str(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-%J.log')),'python3','-B','-u',str(c.D/'controller.py'),'--shard',str(a.shard),'--manifest-sha',a.manifest_sha,'--publication-commit',a.publication_commit]
 q['keeper_argv']=command;c.publish(intent,dict(q,created_unix=time.time(),one_attempt=True))
 with(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(command,env=c.clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(q,keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,intent_path=str(intent),no_epoch_until_freshidle=True)));return 0
if __name__=='__main__':raise SystemExit(main())

"""Only this new constructor intent. Preflight returns before intent/log/Popen."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys,time
import controller as c

def keeper_command(m,commit):
 return ['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:11:00',
 '--job-name='+c.LABEL+'-'+c.RUN+'-keeper','--output='+str(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-%J.log')),
 'python3','-B','-u',str(c.D/'controller.py'),'--manifest-sha',m,'--publication-commit',commit]

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=c.verify(a.manifest_sha);proof=c.whole_cli();membership=c.membership()
 rows=c.rows();assert not any(v[1].startswith(c.LABEL+'-'+c.RUN)for v in rows),'Follow original, never replay'
 for row in rows:
  if re.fullmatch(r'910380[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in c.TERMINAL:
   assert not set(c.hosts(row[4]))&set(c.NODES),'Live overlapping GPU step'
 folder=c.host(c.config()['runtime_directory']);assert not folder.exists()
 intent=c.R/'results'/(c.LABEL+'-'+c.RUN+'-intent.json');watch=c.R/'results'/(c.LABEL+'-'+c.RUN+'-watch.json')
 assert not intent.exists()and not watch.exists()and not list((c.R/'logs').glob(c.LABEL+'-'+c.RUN+'*'))
 args=keeper_command(a.manifest_sha,a.publication_commit)
 q=dict(passed=True,actual_whole_cli=proof,membership=membership,keeper_argv=args,planned_cards=32,embedding_lookups_per_owner=4,weight_keys=1,fullModel_forward=0,HTTP=0,
   source_manifest_SHA256=a.manifest_sha,publication_commit=a.publication_commit,Slurm_submitted=0,intent_created=False,GPU_epoch_created=False)
 if a.preflight_only:print(json.dumps(q));return
 assert m['GPU_GO']and not m['Model_GO'];c.publish(intent,dict(q,one_attempt=True,created_unix=time.time()))
 with(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-client.log')).open('x')as log:
  proc=subprocess.Popen(args,env=c.clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,intent_created=True,intent_path=str(intent),source_manifest_SHA256=a.manifest_sha)))
if __name__=='__main__':main()

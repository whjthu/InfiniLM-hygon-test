"""Only this new constructor intent. Preflight returns before intent/log/Popen."""
from pathlib import Path
import argparse,json,os,re,signal,subprocess,sys,time
import controller as c

def host_whole_cli(manifest,commit):
 """One actual HOST child;80s metadata work+10s owned-group cleanup, file stdio."""
 ledger=c.R/'results'/(c.LABEL+'-'+c.RUN+'-HOST.json')
 prefix=c.R/'logs'/(c.LABEL+'-'+c.RUN+'-HOST')
 q=dict(source_manifest_SHA256=manifest,publication_commit=commit,phase='before-Popen',intent=False,GPU=0,Slurm=0,passed=False)
 c.publish(ledger,q);started=time.monotonic();limit=started+90;proc=None
 def save():
  q['elapsed_s']=time.monotonic()-started;t=ledger.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(ledger)
 def live():
  proc.poll()
  try:os.killpg(proc.pid,0);return True
  except ProcessLookupError:return False
 def send(kind):
  try:os.killpg(proc.pid,kind);q.setdefault('signals',[]).append(int(kind))
  except ProcessLookupError:pass
 stdout=Path(str(prefix)+'.stdout');stderr=Path(str(prefix)+'.stderr')
 try:
  with stdout.open('xb')as out,stderr.open('xb')as err:
   proc=subprocess.Popen([sys.executable,'-B','-u',str(c.D/'controller.py'),'--manifest-sha',manifest,'--publication-commit',commit,'--preflight-only'],env=c.clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   q.update(phase='HOST_METADATA_RUNNING',PID=proc.pid,PGID=proc.pid,UID=os.getuid());save()
   while proc.poll()is None and time.monotonic()<started+80:time.sleep(.1)
   q['timeout']=proc.poll()is None
   if live():
    send(signal.SIGTERM);term=min(limit,time.monotonic()+5)
    while live()and time.monotonic()<term:time.sleep(.1)
    if live():send(signal.SIGKILL)
    while live()and time.monotonic()<limit:time.sleep(.1)
  q.update(parent_rc=proc.poll(),group_gone=not live(),stdout_path=str(stdout),stderr_path=str(stderr),stdout_SHA256=c.sha(stdout),stderr_SHA256=c.sha(stderr),phase='HOST_METADATA_TERMINAL')
  q['passed']=q['parent_rc']==0 and q['group_gone']and not q['timeout']
  save();assert q['passed'],'Actual HOST metadata failed; no intent/Slurm/GPU, never replay'
  payload=json.loads(stdout.read_bytes());assert payload['passed']and payload['GPU']==payload['Slurm']==0
  return q
 finally:
  if proc is not None:
   if live():
    send(signal.SIGTERM);term=min(limit,time.monotonic()+5)
    while live()and time.monotonic()<term:time.sleep(.1)
    if live():send(signal.SIGKILL)
    while live()and time.monotonic()<limit:time.sleep(.1)
   q.update(parent_rc=proc.poll(),group_gone=not live())
   if not q['group_gone']:q['passed']=False
  q['terminal_UNKNOWN']=not q.get('group_gone',False);save()

def keeper_command(m,commit):
 return ['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','09:00:00',
 '--job-name='+c.LABEL+'-'+c.RUN+'-keeper','--output='+str(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-%J.log')),
 'python3','-B','-u',str(c.D/'controller.py'),'--manifest-sha',m,'--publication-commit',commit]

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=c.verify(a.manifest_sha);proof=host_whole_cli(a.manifest_sha,a.publication_commit);membership=c.membership()
 rows=c.rows();assert not any(v[1].startswith(c.LABEL+'-'+c.RUN)for v in rows),'Follow original, never replay'
 for row in rows:
  if re.fullmatch(r'910380[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in c.TERMINAL:
   assert not set(c.hosts(row[4]))&set(c.NODES),'Live overlapping GPU step'
 folder=c.host(c.config()['runtime_directory']);assert not folder.exists()
 intent=c.R/'results'/(c.LABEL+'-'+c.RUN+'-intent.json');watch=c.R/'results'/(c.LABEL+'-'+c.RUN+'-watch.json')
 allowed_HOST_logs={c.LABEL+'-'+c.RUN+'-HOST.stdout',c.LABEL+'-'+c.RUN+'-HOST.stderr'}
 assert not intent.exists()and not watch.exists()and {p.name for p in(c.R/'logs').glob(c.LABEL+'-'+c.RUN+'*')}==allowed_HOST_logs
 args=keeper_command(a.manifest_sha,a.publication_commit)
 q=dict(passed=True,actual_whole_cli=proof,membership=membership,keeper_argv=args,planned_cards=32,planned_questions=256,complete_checkpoint_loader=True,HTTP=0,
   source_manifest_SHA256=a.manifest_sha,publication_commit=a.publication_commit,Slurm_submitted=0,intent_created=False,GPU_epoch_created=False)
 if a.preflight_only:print(json.dumps(q));return
 assert m['GPU_GO']and m['Model_execution_authorized'];c.publish(intent,dict(q,one_attempt=True,created_unix=time.time()))
 with(c.R/'logs'/(c.LABEL+'-'+c.RUN+'-keeper-client.log')).open('x')as log:
  proc=subprocess.Popen(args,env=c.clean_environment(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,intent_created=True,intent_path=str(intent),source_manifest_SHA256=a.manifest_sha)))
if __name__=='__main__':main()

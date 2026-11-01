"""Only the one published HC363 CPU keeper; GPU submission belongs to controller."""
import pathlib,json,hashlib,subprocess,argparse,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-HC-shared-owner-r363/GPU-v1';LABEL='Deep-HC-shared-owner-r363-v1'
p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
assert hashlib.sha256((S/'GPU_MANIFEST.json').read_bytes()).hexdigest()==a.manifest_sha
m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert all(hashlib.sha256((S/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
intent=R/'results'/(LABEL+'-intent.json');watch=R/'results'/(LABEL+'-watch.json');assert not intent.exists()and not watch.exists()
q=dict(publication_commit=a.publication_commit,manifest_SHA256=a.manifest_sha,CPU_keeper_only=True,one_attempt=True,GPU_max_s=600,CPU_keeper_max_s=900,Model_loaded=False)
with intent.open('x')as f:json.dump(q,f,indent=2)
cmd=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','00:15:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'python3',str(S/'controller.py'),'--publication-commit',a.publication_commit,'--manifest-sha',a.manifest_sha]
with(R/'logs'/(LABEL+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(q,CPU_keeper_client_PID=proc.pid,actual_CPU='PENDING_NOT_YET_OBSERVED',actual_GPU='NOT_SUBMITTED_BY_LAUNCHER')))

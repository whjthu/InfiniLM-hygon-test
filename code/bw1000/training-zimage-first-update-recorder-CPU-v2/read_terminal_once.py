"""Read the one original request and its exact terminal artifacts once per tag."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
intent=json.loads((D/'launch-intent.json').read_bytes());claim=intent['claim'];request=intent['request_id']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-zimage-first-update-recorder-CPU-v2.json']))
sources={}
for n,pin in task['source_SHA256'].items():
 raw=subprocess.check_output(git+['show',claim+':'+task['source']+'/'+n]);assert hashlib.sha256(raw).hexdigest()==pin;sources[n]=raw
files={target:dict(bytes=len(sources[n]),SHA256=hashlib.sha256(sources[n]).hexdigest()) for target,n in task['remote_files'].items()}
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEXPECTED_FILES='+repr(files)+'\n').encode()+sources['read_original_terminal.py']
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
start=time.monotonic();r=subprocess.run(ssh,input=payload,capture_output=True,timeout=85)
(D/(tag+'-stdout.private.json')).write_bytes(r.stdout);(D/(tag+'-stderr.private.txt')).write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout.strip() else None
receipt=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,request_id=request,claim=claim,
 stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),value=v,new_execution_attempted=False)
with (D/(tag+'-reader-result.json')).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed') if v else None,failure=v.get('failure') if v else None,
 artifacts=len(v.get('artifacts',{})) if v else 0,header_only_files=v.get('header_only_files') if v else None,
 reader_weight_body_bytes_read=v.get('reader_weight_body_bytes_read') if v else None,new_execution_attempted=False)))

"""Run only the published independent reader against this original CPU terminal."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
intent=json.loads((D/'launch-intent.json').read_bytes());out=json.loads((D/'transport-outcome.json').read_bytes())
assert out['remote_outcome']=='ORIGINAL_RETURNED' and out['transport_rc'] in (0,1) and 'value' in out
assert out['value']['returncode'] is not None and out['value']['owned_group_reaped']
claim=intent['claim'];request=intent['request_id']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
source=subprocess.check_output(git+['show',claim+':code/bw1000/training-zimage-native-meta-CPU-v1/read_original_terminal.py'])
code=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\n').encode()+source
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
start=time.monotonic();r=subprocess.run(ssh,input=code,capture_output=True,timeout=75)
for n,b in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
    with (D/(tag+'-root-'+n)).open('xb') as f:f.write(b)
result=dict(claim=claim,request_id=request,transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,
    stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_CPU_invocations=0)
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/(tag+'-root-result.json')).open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,original_terminal_consumed=v.get('original_terminal_consumed'),
    native_meta_qualified=v.get('native_meta_qualified'),actual_base_metadata_rows=v.get('actual_base_metadata_rows'),
    actual_trainable_rows=v.get('actual_trainable_rows'),actual_trainable_elements=v.get('actual_trainable_elements'),
    error_type=v.get('failure',{}).get('error_type'),phase=v.get('failure',{}).get('phase'),new_CPU_invocations=0)))

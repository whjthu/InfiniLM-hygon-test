"""Consume the completed original reference once, with CPU masks and provider audit."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
code=r'''
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-a100-first-updates-r3')
assert not (root/'independent-reader-intent.json').exists(),'Reconcile the saved original reader before any repeat'
raw=(root/'outcome.json').read_bytes();outcome=json.loads(raw)
assert outcome['native_returncode']==0 and outcome['keeper_passed'] and outcome['capture_rank_records']==8 and outcome['native25_rank_records']==8
installed=json.loads((root/'installed.json').read_bytes())
p=root/'src/read_reference.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==installed['files']['read_reference.py']
runtime=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1')
plans={name:json.loads((root/'src'/filename).read_bytes())['files'] for name,filename in [('','FAST_FILES.json'),('triton-v1','TRITON_FILES.json')]}
provider_rows=[];seen={}
for rank in range(8):
    p=root/'capture'/('rank-%02d'%rank)/'runtime-providers.json';data=p.read_bytes();providers=json.loads(data)
    assert set(providers)=={'triton','triton._C.libtriton','causal_conv1d_cuda','fla.ops.gated_delta_rule'}
    for name,item in providers.items():
        path=Path(item['path']);sub='triton-v1' if name.startswith('triton') else '';rel=path.relative_to(runtime/sub).as_posix();bound=plans[sub][rel]
        assert item['SHA256']==bound['SHA256']
        if item['path'] not in seen:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==item['SHA256'];seen[item['path']]=h.hexdigest()
    provider_rows.append({'rank':rank,'receipt_SHA256':hashlib.sha256(data).hexdigest(),'providers':providers})
with (root/'independent-reader-intent.json').open('x') as f:json.dump({'epoch':time.time(),'outcome_SHA256':hashlib.sha256(raw).hexdigest(),'provider_paths_independently_rehashed':seen},f)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',PYTHONPATH=str(root/'src'))
epoch=time.time()
with (root/'independent-reader.stdout').open('xb') as stdout,(root/'independent-reader.stderr').open('xb') as stderr:
    proc=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(root/'src/read_reference.py')],env=env,stdout=stdout,stderr=stderr)
    stat=Path('/proc')/str(proc.pid)/'stat';ticks=int(stat.read_text().split(') ',1)[1].split()[19])
    with (root/'independent-reader-owner.json').open('x') as f:json.dump({'PID':proc.pid,'startticks':ticks,'epoch':epoch},f)
    try:rc=proc.wait(timeout=630)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:rc=proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();rc=proc.wait(timeout=5)
result={'reader_rc':rc,'wall_seconds':time.time()-epoch,'provider_rows':provider_rows,'GPU_execution':False,'stdout_tail':(root/'independent-reader.stdout').read_text()[-1500:],'stderr_tail':(root/'independent-reader.stderr').read_text()[-1500:]}
if (root/'independent-reference.json').is_file():
    data=(root/'independent-reference.json').read_bytes();result['independent_reference_SHA256']=hashlib.sha256(data).hexdigest();result['value']=json.loads(data)
with (root/'independent-reader-wrapper.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
with (BASE/'reader-launch-intent.json').open('x') as f:json.dump({'reader':'original published read_reference.py','claim_commit':'e8898fdd6d006a60e0960dd52eba17da6a28d287'},f)
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26','python3 -B -u -'],input=code.encode(),capture_output=True,timeout=680)
with (BASE/'independent-reader.transport.stdout').open('xb') as f:f.write(r.stdout)
with (BASE/'independent-reader.transport.stderr').open('xb') as f:f.write(r.stderr)
assert r.returncode==0,r.stderr.decode(errors='replace')[-1000:]
v=json.loads(r.stdout)
with (BASE/'independent-reader-original-result.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k not in ('value','provider_rows')}))

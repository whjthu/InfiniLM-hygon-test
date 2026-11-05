from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,os,socket,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/GLM952-prefix-continuation/entry-env-repair-v3'
label='GLM952-prefix-continuation-env-CPU-v3'
publication=PUBLICATION
claim=json.loads(files['CPU_RESOURCE_CLAIM.json'])
pin=sha256(files['CPU_RESOURCE_CLAIM.json']).hexdigest()
assert pin=='a20cb1373a3de26c571a55dbb6f246614a0b60179a139fdce870b69a2cf7a8f4'
assert not D.exists() and not (R/'results'/label).exists() and not list((R/'logs').glob(label+'-*'))
assert set(files)==set(claim['files_SHA256'])|{'CPU_RESOURCE_CLAIM.json'}
assert all(sha256(files[n]).hexdigest()==v for n,v in claim['files_SHA256'].items())
rows=subprocess.check_output(['sacct','-S','2026-11-05','-j','910371','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode'],text=True,timeout=15)
assert not any(label in line for line in rows.splitlines())
assert any(line.startswith('910371.851|') and line.endswith('|COMPLETED|0:0') for line in rows.splitlines())
outer=R/'incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/CPU_outer.sh'
assert sha256(outer.read_bytes()).hexdigest()=='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d'
D.mkdir()
for name,data in files.items():
    assert Path(name).name==name
    with (D/name).open('xb') as stream:stream.write(data)
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_'))}
env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
    GLM_PREFIX_ENV_CPU_CLAIM_SHA=pin,WAVE_INSIDE_R2323='GLM952-prefix-continuation/entry-env-repair-v3/inside.sh')
args=['srun','--jobid=910371','--overlap','--exact','--immediate=20','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1',
    '-c2','--gres=none','--mem=2048M','-w','a02r3n15,a02r3n16','-t','00:01:00','--job-name='+label,
    '--output='+str(R/'logs'/(label+'-%J-%t.log')),'--export=ALL','bash',str(outer),'probe']
intent=R/'results'/(label+'-intent.json')
record=dict(HOST=socket.gethostname(),UTC=datetime.now(timezone.utc).isoformat(),publication=publication,claim_SHA256=pin,GPU=0)
with intent.open('x') as stream:json.dump(record,stream)
with (R/'logs'/(label+'-client.log')).open('x') as stream:
    p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record.update(srun_PID=p.pid,owned_PGID=p.pid,startticks=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19])
with (R/'results'/(label+'-launch.json')).open('x') as stream:json.dump(record,stream)
print(json.dumps(record))

"""Exercise the frozen-source bootstrap and a real container read-only bind."""
import hashlib,json,os,subprocess,socket,sys,time
from pathlib import Path
from frozen_bootstrap import bind_frozen_framework

root=Path(sys.argv[1])
src=root/'src'
intent=json.loads((root/'intent.json').read_bytes())
manifest=(src/'FRAMEWORK_FILES.json').read_bytes()
frozen=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-HIP-attention-preflight-CPU-v1/framework')
node=socket.gethostname().split('.')[0]
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
assert node=='f13r2n07' and step.startswith('1103606.')
reference=bind_frozen_framework(root/'witness',request_id=intent['request_id'],claim_commit=intent['claim'],
    source_manifest_SHA256=intent['source_files']['SOURCE_FILES.json'],framework_root=frozen,framework_manifest=manifest,
    framework_manifest_SHA256=intent['source_files']['FRAMEWORK_FILES.json'],epoch=intent['epoch'],deadline=intent['work_deadline'])
assert reference['files_verified']==382 and reference['framework_files_copied']==0
probe="""from pathlib import Path
import hashlib,json
line,=[x for x in Path('/proc/self/mountinfo').read_text().splitlines() if x.split()[4]=='/frozen-framework']
assert 'ro' in line.split()[5].split(',')
data=Path('/frozen-framework/generation/src/gencuda/zimage_hip_attention_probe.py').read_bytes()
print(json.dumps({'mount_read_only':True,'probe_SHA256':hashlib.sha256(data).hexdigest()}))
"""
env=os.environ.copy()
env.update(APPTAINERENV_HIP_VISIBLE_DEVICES='-1',APPTAINERENV_ROCR_VISIBLE_DEVICES='-1',APPTAINERENV_CUDA_VISIBLE_DEVICES='-1')
p=subprocess.run(['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/','--bind',
    reference['framework_root']+':/frozen-framework:ro',
    '/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1','python3','-B','-c',probe],
    env=env,capture_output=True,text=True,timeout=min(25,intent['work_deadline']-time.time()))
if p.returncode:
    print(p.stderr[-2000:],file=sys.stderr)
    raise SystemExit(p.returncode)
result=json.loads(p.stdout.splitlines()[-1])
assert result['mount_read_only'] and result['probe_SHA256']==json.loads(manifest)['files']['generation/src/gencuda/zimage_hip_attention_probe.py']['SHA256']
with (root/'evidence/CPU-result.json').open('x') as f:
    json.dump(dict(passed=True,owner={'node':node,'step':step},reference=reference,mount=result,GPU_cards=0),f)

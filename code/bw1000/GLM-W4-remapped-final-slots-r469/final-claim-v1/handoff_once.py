"""Install the separately published claim once. No launch/poll/Runtime here.

After this succeeds, use the unchanged deployment launch_once.py separately
for each exact slot. Its existing HOSTpreflight/fresh/dedup/keeper owns execution.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--claim-sha',required=True);p.add_argument('--publication-commit',required=True);a=p.parse_args()
    F=Path(__file__).resolve().parent;D=F.parent/'deployment-v1';R=D.parents[2]
    assert R in [Path('/refactor'),Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')]
    assert D==R/'incoming/GLM-W4-remapped-final-slots-r469/deployment-v1'
    H=lambda raw:hashlib.sha256(raw).hexdigest()
    raw=(F/'GPU_MANIFEST-remapped-two.json').read_bytes();assert H(raw)==a.claim_sha
    claim=json.loads(raw);resource=json.loads((F/'GPU_RESOURCE_CLAIM.json').read_bytes())
    assert resource['GPU_manifest_SHA256']==a.claim_sha and re.fullmatch('[0-9a-f]{40}',a.publication_commit)
    assert claim['GPU_GO']and claim['Model_GO']and claim['admitted_slots']==[4,5]
    assert claim['deployment_source_SHA256']=='b516618e2560771a6f7f2fa6288f543d3d42af180fe35761af87b5bd88484aeb'
    assert H((D/'SOURCE_MANIFEST.json').read_bytes())==claim['deployment_source_SHA256']
    sys.path.insert(0,str(D));import controller as C
    assert C.D.resolve()==D.resolve()
    for slot in [4,5]:C.configure_slot(slot);C.verify(claim['deployment_source_SHA256'])
    receipt=R/Path(claim['remaining_scope_CPU_receipt']['path']).relative_to('/refactor')
    cpu=receipt.read_bytes();assert H(cpu)==claim['remaining_scope_CPU_receipt']['SHA256']
    q=json.loads(cpu);assert q['completed']and q['passed']and q['actual_CPU']=='910407.988'and q['actual_public_nonserve_CLI_roles']==4
    account=subprocess.run(['sacct','-S','2026-11-04','-j','910371,910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100'],capture_output=True,text=True,timeout=30);assert account.returncode==0
    rows=[line.split('|')for line in account.stdout.splitlines()if'|'in line]
    cpu_rows=[row for row in rows if row[0]=='910407.988']
    assert len(cpu_rows)==1 and cpu_rows[0][1]=='GLM469-remapped-two-deployment-CPU-v1'and cpu_rows[0][2:4]==['COMPLETED','0:0']
    for slot in [4,5]:
        s=resource['slots'][slot-4];assert s['slot']==slot
        prefix=s['label']+'-'+s['run']
        assert not any(row[1].startswith(prefix)for row in rows),'Original actual label exists; follow it, never replace'
        for suffix in ['-intent.json','-launch.json','-watch.json']:
            assert not(R/'results'/(prefix+suffix)).exists(),'Original run state exists'
        assert not(R/'results/GLM-W4-fewshot-quality-r444'/s['run']).exists()
        assert not list((R/'logs').glob(prefix+'*'))
    target=D/resource['handoff']['target_filename'];old=target.read_bytes()
    assert H(old)==resource['handoff']['expected_old_false_SHA256']
    original=json.loads(old);assert original['GPU_GO']is False and original['Model_GO']is False
    intent=F/'handoff-intent.json';assert not intent.exists()and not(F/'handoff-result.json').exists()
    record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),publication_commit=a.publication_commit,claim_SHA256=a.claim_sha,old_false_SHA256=H(old),source_manifest_SHA256=claim['deployment_source_SHA256'],origin_HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],new_jobs=0,new_HTTP=0,phase='KNOWN_FALSE_TO_PUBLISHED_CLAIM_HANDOFF')
    with intent.open('x')as f:json.dump(record,f,indent=2)
    with(F/'original-deployment-false-proposal.json').open('xb')as f:f.write(old)
    temporary=D/('GPU_MANIFEST-remapped-two.'+a.claim_sha[:16]+'.new')
    with temporary.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert H(target.read_bytes())==H(old)
    os.replace(temporary,target);assert H(target.read_bytes())==a.claim_sha
    for slot in [4,5]:C.configure_slot(slot);m=C.verify(claim['deployment_source_SHA256']);assert C.verify_gpu(a.claim_sha,m)
    record.update(phase='HANDOFF_COMPLETED_NO_LAUNCH',target=str(target),source_and_8configs_unchanged=True,slots=[4,5],fresh_idle_not_yet_executed=True)
    with(F/'handoff-result.json').open('x')as f:json.dump(record,f,indent=2)
    print(json.dumps(record))


if __name__=='__main__':main()

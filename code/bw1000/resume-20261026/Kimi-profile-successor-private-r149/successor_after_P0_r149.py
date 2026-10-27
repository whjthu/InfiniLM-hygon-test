"""One CPU waiter -> one existing r136b invoker; no source mutation/retry."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
GPU, KEEPER = '910380.438', '910407.477'
RUN = 'kimi-agent-p0-r144d-a91b0c6e'
NODES = ['a14r2n08', 'a14r2n09', 'a14r2n10', 'a14r2n12']
PROFILE_MANIFEST = '99d845cc08d0b13569e7ccd0ea253c27945391709f9175ceae4e845c44b15a14'
PROFILE_PINS = 'afcc8fba9498850cf6a5ebadf37664e089b69da339940ae48a0564ae3b9eb373'
PROFILE_ENTRY = '6a7485910ea3626925420815d27bb43136e08cb51a2bdd8f211effea1aaa9486'
P0_MANIFEST = 'ebb654ec3f5f20f23b77fedbbf7329111cc4c360c4da844bf4839b3a7c67ff1a'
ACTIVE = {'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING'}
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_claim(go, claim):
    assert go['publication_branch'] == 'bw1000-experiments'
    assert re.fullmatch('[0-9a-f]{40}', go['publication_commit'])
    assert go['claim_export_is_current_branch_head'] is True
    assert go['GPU_successor_authorized'] is True and go['source_published'] is True
    assert claim['work_item'] == 'kimi-current-c8-verify-profile'
    assert 'DEFERRED' not in claim['state'] and 'PARK' not in claim['state']
    assert claim['new_source_manifest_SHA256'] == PROFILE_MANIFEST
    v=claim['resources']
    assert v['target_parent']=='910380' and v['target_nodes']==NODES and v['target_cards']==32
    assert (v['draft_parent'],v['draft_node'],v['draft_cards'])==('910371','a02r3n03',1)
    assert claim['handoff_gate']=={'predecessor_GPU':GPU,'predecessor_keeper':KEEPER,
        'successor':'Kimi-sparse-profile-r136b','order':'P0_TERMINAL_CLEANUP_THEN_PROFILE'}


def predecessor_cleanup(watch, node_reports, identity, rows):
    if not all(handle in rows and rows[handle][1] in TERMINAL for handle in [GPU,KEEPER]):
        return False
    if not watch.get('post_empty'):
        return False
    assert watch['actual_handle']==GPU and watch['CPU_actual_step']==KEEPER
    assert watch['terminal'][0]==GPU and watch['terminal'][2].split()[0].rstrip('+') in TERMINAL
    post=watch['post_KFD'];probe=post['rows']
    assert post['terminal'][2:4]==['COMPLETED','0:0']
    assert len(probe)==4 and {x['node']for x in probe}==set(NODES)
    assert all(x['passed'] and x['kfd_pids']==[] for x in probe)
    assert identity['actual_step']==GPU and identity['cards']==32
    assert identity['manifest_SHA256']==P0_MANIFEST
    assert len(node_reports)==4
    for index,report in enumerate(node_reports):
        assert report['node']==index and report['run']==RUN and report['manifest_SHA256']==P0_MANIFEST
        assert report['all8_owned_workers_shutdown_completed'] is True
        assert report['agent_HTTP_normal_shutdown'] is True
        assert not report.get('shutdown_error')
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--go-file',required=True);ap.add_argument('--source-manifest-sha',required=True);a=ap.parse_args()
    assert os.environ['SLURM_JOB_ID']=='910407' and socket.gethostname()=='a14r2n20'
    source=Path(__file__).resolve().parent;mp=source/'source-manifest-r149.json'
    assert sha(mp)==a.source_manifest_sha
    manifest=json.loads(mp.read_bytes())
    for rel,h in manifest['files_SHA256'].items():assert sha(source/rel)==h
    go_path=Path(a.go_file);go=json.loads(go_path.read_bytes());go_hash=sha(go_path)
    assert go['source_manifest_SHA256']==a.source_manifest_sha
    claim_path=Path(go['claim_path']);assert claim_path.resolve().is_relative_to(R/'incoming')
    assert sha(claim_path)==go['claim_SHA256'];validate_claim(go,json.loads(claim_path.read_bytes()))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_')) or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    result=R/'results/Kimi-profile-after-P0-r149.json';intent=R/'results/Kimi-profile-after-P0-r149-intent.json'
    assert not result.exists() and not intent.exists()
    record={'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'predecessor_GPU':GPU,'predecessor_keeper':KEEPER,
        'profile_invoked':False,'retries':0,'GPU_submitted_by_successor':False,'publication_commit':go['publication_commit'],'claim_SHA256':go['claim_SHA256']}
    def save():
        record['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(result)
    def account():
        z=subprocess.run(['sacct','-S','2026-10-27','-j',GPU+','+KEEPER,'-n','-P','-o','JobIDRaw,State%40,ExitCode'],env=env,text=True,capture_output=True,timeout=25);assert z.returncode==0
        rows={}
        for line in z.stdout.splitlines():
            v=line.split('|')
            if len(v)>=3 and v[0]in [GPU,KEEPER]:
                assert v[0]not in rows;v[1]=v[1].split()[0].rstrip('+');rows[v[0]]=v
        return rows
    # Reserve the unchanged downstream keeper's full 2h15 limit and 120s
    # dispatcher budget inside this chain's 4h reservation. Missing this
    # latest launch point needs root review, never an unbounded late launch.
    save();deadline=time.monotonic()+4*3600-(2*3600+15*60)-120
    try:
        while time.monotonic()<deadline:
            assert sha(go_path)==go_hash and sha(claim_path)==go['claim_SHA256'],'GO or current exported claim changed; fail closed'
            rows=account();record['predecessor_states']=rows
            watchpath=R/'results/Kimi-agent-service-r144d-watch.json'
            if all(x in rows and rows[x][1] in TERMINAL for x in [GPU,KEEPER]) and watchpath.exists():
                watch=json.loads(watchpath.read_bytes())
                if watch.get('post_empty'):
                    reports=[json.loads((R/'results'/('Kimi-agent-service-r144d-'+RUN+'-node%d.json'%i)).read_bytes())for i in range(4)]
                    identity=json.loads((R/'results'/('Kimi-agent-service-identity-r144d-'+RUN+'.json')).read_bytes())
                    if predecessor_cleanup(watch,reports,identity,rows):break
            save();time.sleep(30)
        else:raise TimeoutError('4h chain reservation cannot fit original profile keeper; no invocation')
        record['predecessor_cleanup_verified']=True;record['P0_passed']=watch.get('P0_passed');save()
        profile=R/'incoming/Kimi-sparse-profile-Model-r136b';entry=profile/'launcher/preflight_then_run.py';pins=profile/'launcher/model-source-pins.json'
        assert sha(entry)==PROFILE_ENTRY and sha(pins)==PROFILE_PINS
        pin=json.loads(pins.read_bytes());assert pin['manifest_SHA256']==PROFILE_MANIFEST
        for path,h in pin['files'].items():assert sha(path.replace('/refactor',str(R),1))==h
        profile_go=json.loads((profile/'launcher/GPU-GO-after-source-publication.json').read_bytes())
        assert profile_go['authorized_by']=='parent-root' and profile_go['manifest_SHA256']==PROFILE_MANIFEST
        assert re.fullmatch('[0-9a-f]{40}',profile_go['Git_publication_SHA'])
        steps=subprocess.check_output(['squeue','--steps','-j','910407','-h','-o','%i|%j|%N'],env=env,text=True,timeout=25).splitlines()
        assert len([x for x in steps if x.split('|')[0].split('.')[-1].isdigit()])<4
        assert not (R/'results/Kimi-sparse-profile-r136b-intent.json').exists()
        with intent.open('x') as stream:json.dump({'profile_entry_SHA256':PROFILE_ENTRY,'profile_pins_SHA256':PROFILE_PINS,'profile_manifest_SHA256':PROFILE_MANIFEST,'predecessor_GPU':GPU,'predecessor_keeper':KEEPER,'no_retry':True},stream)
        # Existing entry alone owns CPU/profile scheduling, fresh probes and pins.
        z=subprocess.run(['python3','-B',str(entry)],env=env,text=True,capture_output=True,timeout=120)
        record.update(profile_invoked=True,profile_invoker_returncode=z.returncode)
        launch=R/'results/Kimi-sparse-profile-r136b-launch.json'
        if launch.exists():record['profile_launch_receipt_SHA256']=sha(launch)
        save();assert z.returncode==0
    except BaseException as error:
        record['error_class']=type(error).__name__;record['needs_root_review']=True;save();raise


if __name__=='__main__':main()

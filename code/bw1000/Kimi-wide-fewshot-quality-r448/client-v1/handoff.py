"""Real r434 weights-ready/claim path; no invented MODEL_READY filename."""
from pathlib import Path
import hashlib,json
NATIVE='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def ready_handoff(runtime,run,step,source,config_sha,model_sha):
    runtime=Path(runtime);records=[]
    for node in range(4):
        claim=runtime/('claim-'+run+'-node%d.json'%node)
        report=runtime/('Kimi-public-r428-'+run+'-node%d.json'%node)
        try:a=json.loads(claim.read_bytes());b=json.loads(report.read_bytes())
        except(FileNotFoundError,json.JSONDecodeError):return None
        assert (a['run'],a['actual_step'],a['node'],a['source_manifest_SHA256'],a['config_SHA256'])==(run,step,node,source,config_sha)
        assert type(a['PID'])is int and a['PID']>0
        assert (b['run'],b['node'],b['public_source_manifest_SHA256'])==(run,node,source)
        if b.get('error'):raise RuntimeError('Actual owner failed before HTTP: '+repr(b['error']))
        if b.get('phase')!='weights_ready':return None
        assert b['library_sha256']==NATIVE and b['manifest_SHA256']==model_sha
        assert (b['TP'],b['PP'])==(16,2) and b['model_weight_keys_validated']
        assert b['actual_allstage_weight_keys']['all93_layer_ids_exactly_once_across_PPstages']
        records.append(dict(node=node,PID=a['PID'],claim_SHA256=sha(claim.read_bytes()),report_SHA256=sha(report.read_bytes())))
    return dict(actual_four_owner_reports_weights_ready=True,Native_internal_workers=32,records=records,
                public_listener_yet_verified=False,scope='Owner source publishes weights_ready before its all-rank barrier; client subsequently proves owner-PID HTTP LISTEN after public lifespan start.')

def stop_requested(runtime,run,source,config_sha,epoch):
    path=Path(runtime)/('STOP-'+run+'.json')
    if not path.exists():return False
    value=json.loads(path.read_bytes())
    assert value==dict(identity=dict(run=run,source_manifest_SHA256=source,config_SHA256=config_sha,gpu_start_epoch=epoch),action='stop')
    return True

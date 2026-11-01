import hashlib,json,re
from pathlib import Path
STAGES=((910371,'a02r3n03'),(910371,'a02r3n07'),(910374,'a02r4n04'),(910374,'a02r4n05'))
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
NATIVE='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
GPU_SECONDS,NATIVE_SECONDS,CPU_SECONDS=7200,6600,7800
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(g):
    assert re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9-]{7,60}',g['run'])
    assert g['execute_once']is True and g['Native_SHA256']==NATIVE
    assert g['stages']==[dict(stage=s,parent=p,node=n)for s,(p,n)in enumerate(STAGES)]
    assert(g['cards'],g['TP'],g['PP'],g['rows'],g['context'],g['pairs'],g['requests'],g['ordinary_budget'])==(32,8,4,2,32768,6,12,4096)
    assert(g['GPU_max_s'],g['Native_max_s'],g['CPU_max_s'])==(GPU_SECONDS,NATIVE_SECONDS,CPU_SECONDS)
    assert g['publication_branch']=='bw1000-experiments' and re.fullmatch('[a-f0-9]{40}',g['publication_commit'])
    assert all(re.fullmatch('[a-f0-9]{64}',g[k])for k in ['owner_manifest_SHA256','controller_manifest_SHA256','CPU_receipt_SHA256'])
    assert g['no_retry_reload']is True
    return g


def bundle(directory,expected):
    directory=Path(directory);assert sha(directory/'SOURCE_MANIFEST.json')==expected
    q=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
    for n,h in q['files_SHA256'].items():
        assert not Path(n).is_absolute()and'..'not in Path(n).parts
        assert sha(directory/n)==h,n
    return q

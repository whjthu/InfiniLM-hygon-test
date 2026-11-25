"""Stage and operate one published Pack16 request over the existing bound SSH route."""
import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
from uuid import uuid4

G=Path.cwd().resolve()
LOCAL=G/"local-only"
LABEL='training-O01-pack16-r41-n10-single-DCU-v1'
SPEC=G/"coordination/tasks"/f"{LABEL}.json"
PACKAGE=G/"code/bw1000"/LABEL
ROUTE=LOCAL/"run-minicpm-sdpa-target-asset-preflight-original.py"
SEND=G/"tools/run_hygon_bound_payload.py"


def hash_bytes(raw):
    return sha256(raw).hexdigest()


REMOTE=r"""
from base64 import b64decode
from datetime import datetime,timezone
from hashlib import sha256
import json,os,re,subprocess,sys,time
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
results=root/'results'
runtime=root/'incoming/O01-pack16-deps-json-c0243062d1f2-reviewed/runtime'
owner=CLAIM['request']
parent=results/CLAIM['label']
run=parent/owner
def need(ok,reason):
    if not ok:
        raise ValueError(reason)
def digest(raw):
    return sha256(raw).hexdigest()
def read(path,cap=65536):
    need(path.resolve(strict=True)==path and path.is_file() and 0<path.stat().st_size<=cap,'EXACT_BOUNDED_PROJECT_PATH')
    return path.read_bytes()
def new(path,raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
def doc(item):
    return (json.dumps(item,sort_keys=True,allow_nan=False)+'\n').encode()
def main():
    need(os.getuid()==154459 and CLAIM['node']=='f12r2n10' and CLAIM['DCU']==1 and CLAIM['budget_seconds']==240
         and CLAIM['ordinary_exclusion_until_independent_post'] is True and CLAIM['one_max_actual_launch']==1,'EXACT_NEW_N10_SCOPE')
    raw=sys.stdin.buffer.read(180001)
    need(0<len(raw)<=180000,'BOUND_COMPONENT_PACKET_EXTENT')
    payload=json.loads(raw)
    os.umask(0o077)
    if MODE=='stage':
        names=CLAIM['sources']
        blobs={name:b64decode(data,validate=True) for name,data in payload['files'].items()}
        need({n:digest(raw) for n,raw in blobs.items()}==names and set(names)==
             {'SOURCE_FILES.json', 'audit_result.py', 'gate.py', 'host.py', 'physical.py', 'precheck.py', 'read_result.py', 'step_evidence.py'},'EXACT_PUBLISHED_COMPONENT_SOURCE_CLOSURE')
        need(digest(read(runtime/'SCOPED_FILES.json'))==CLAIM['original_runtime_manifest_SHA256'],'ORIGINAL_PACK16_RUNTIME')
        need((root/'containers/dtk26-sandbox-r1').is_dir() and (results/'training-t06-sdxl-runtime-v1/site').is_dir(),
             'EXACT_COMPONENT_ORIGINAL_CONTAINER')
        parent.mkdir(exist_ok=True)
        need(parent.resolve(strict=True)==parent,'CANONICAL_COMPONENT_RESULTS')
        run.mkdir()
        for name in ('src','compute'):
            (run/name).mkdir()
        for name,value in blobs.items():
            new(run/'src'/name,value)
        receipt=dict(state='STAGED_SOURCE_ONLY',sources=names,owner=owner,publication=PUBLICATION,slurm_steps=0,requested_DCU=0)
        new(run/'STAGE.json',doc(receipt))
        return receipt
    need(run.resolve(strict=True)==run and all(digest(read(run/'src'/n))==s for n,s in CLAIM['sources'].items()),
         'EXACT_ALREADY_STAGED_COMPONENT')
    stage=json.loads(read(run/'STAGE.json'))
    need(stage['publication']==PUBLICATION and stage['owner']==owner,'EXACT_STAGE_PROVENANCE')
    if MODE=='audit':
        proof=read(run/'R41_INDEPENDENT_POST.json',16384)
        cmd=['/usr/local/python3.12/bin/python3','-I','-S','-B',str(run/'src/audit_result.py'),
             '--request',owner,'--runtime',str(runtime),'--reader',str(run/'src/read_result.py'),
             '--post',str(run/'R41_INDEPENDENT_POST.json'),'--post-sha256',digest(proof)]
        child=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,timeout=35)
        need(child.returncode==0 and 0<len(child.stdout)<=32768,'R41_EXACT_INDEPENDENT_MEASUREMENT_READER')
        return {'state':'R41_INDEPENDENT_MEASUREMENT_READ','result':json.loads(child.stdout),
                'post_SHA256':digest(proof),'zero_new_GPU':True}
    args=['/usr/local/python3.12/bin/python3','-I','-S','-B',str(run/'src/host.py'),MODE,
          '--request',owner,'--publication',PUBLICATION,'--source-manifest-sha256',CLAIM['sources']['SOURCE_FILES.json']]
    if MODE in ('launch','post'):
        physical=payload['physical']
        custody=payload['custody']
        need(custody['owner_request']==owner and custody['publication']==PUBLICATION and custody['node']=='f12r2n10'
             and custody['parent']=='1103606' and custody['exact_root_sole_dispatcher'] is True
             and custody['ordinary_exclusion_until_independent_post'] is True and custody['allows_one_DCU_once'] is True
             and physical['custody_SHA256']==payload['custody_SHA256'],'R41_ACTUAL_EXTERNAL_CUSTODY_GATE')
        if MODE=='launch':
            now=time.time()
            need(0<=now-custody['observed_epoch']<=60 and now+240<=custody['valid_until_epoch'],
                 'R41_CURRENT_EXTERNAL_CUSTODY_EXPIRY')
        token=physical['request']
        need(token==(CLAIM['pre_request'] if MODE=='launch' else CLAIM['post_request']),'R41_NEW_DISTINCT_PHYSICAL_REQUEST')
        need(re.fullmatch('[0-9a-f]{32}',token) and physical['node']=='f12r2n10' and physical['state']==
             'ORIGINAL_ZERO_GRES_PHYSICAL_SAMPLE_CAPTURED','EXACT_FRESH_N10_OBSERVER')
        target=results/('training80-o01-n10-physical-'+token)/'accounting-read.json'
        pre=read(target,16384)
        need(json.loads(pre)==physical,'ORIGINAL_OBSERVER_BYTES_MATCH')
        need(physical['physical_mode']==('pre' if MODE=='launch' else 'post') and physical['owner_request']==owner
             and physical['old_counter_SHA256']==CLAIM['prior_counter_SHA256'], 'EXACT_PRE_POST_OWNER')
        if MODE=='launch':
            release=dict(incumbent_accounting=physical['incumbent_accounting'],old_counter_SHA256=physical['old_counter_SHA256'],
                  owner_request=owner,publication=PUBLICATION,cgroup_SHA256=physical['incumbent_cgroup_SHA256'],
                  terminal_SHA256=physical['incumbent_terminal_SHA256'])
            proof=doc(release)
            [row]=physical['Slurm_accounting_observed']
            wall=datetime.strptime(row[8],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            need(0<=time.time()-wall<=45 and not (run/'GPU_INTENT.json').exists(),'IMMEDIATE_PRE_AND_NO_REPEAT')
            new(run/'PREDECESSOR.json',proof)
            authority=dict(schema='O01_R41_N10_NEW_DIRECT_OWNER_V1',request=owner,publication=PUBLICATION,node='f12r2n10',
              parent='1103606',previous_step=CLAIM['previous_actual_terminal'],historical_release_SHA256=digest(proof),
              historical_zero_GRES_SHA256=CLAIM['prior_counter_SHA256'],root_approved_one_DCU=True,
              sole_dispatcher_n10_until_independent_post=True,automatic_ordinary_refill_allowed=False,borrowed_ordinary_grant=False,
              observed_epoch=wall,valid_until_epoch=min(wall+300,custody['valid_until_epoch']),pre_request=token,pre_sha256=digest(pre))
            hard=doc(authority)
            target=run/'ROOT_DIRECT_AUTHORITY.json'
            new(target,hard)
            args+=['--runtime',str(runtime),'--authority',str(target),'--authority-pin',digest(hard)]
        else:
            args+=['--post-request',token,'--post-sha256',digest(pre)]
            need(not (run/'R41_INDEPENDENT_POST.json').exists(),'R41_EXISTING_POST_USE_INDEPENDENT_AUDIT_ONLY')
    limit=250 if MODE=='launch' else 22
    child=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,timeout=limit)
    need(0<len(child.stdout)<=32768,'BOUNDED_EXACT_HOST_REPORT')
    response=json.loads(child.stdout)
    if MODE=='post' and child.returncode==0 and response['refill_permitted'] is True:
        new(run/'R41_INDEPENDENT_POST.json',doc(response))
    return {'state':response.get('state'),'result':response,'host_rc':child.returncode,'host_stderr_SHA256':digest(child.stderr)}
try:
 report=main()
except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
 report={'state':'ADMISSION_BLOCKED_OR_RECONCILE','error':type(exc).__name__,
         'gate':str(exc) if type(exc) is ValueError and re.fullmatch('[A-Z0-9_]+',str(exc)) else None}
print(json.dumps({**report,'request':REQUEST,'scope_request':owner,'replay_allowed':False},sort_keys=True),flush=True)
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("stage","pre","launch","terminal","post-sample","post","audit"))
    parser.add_argument("publication")
    parser.add_argument("--physical-receipt",type=Path)
    parser.add_argument("--custody",type=Path)
    args=parser.parse_args()
    assert re.fullmatch("[0-9a-f]{40}",args.publication)
    claim=json.loads(SPEC.read_bytes())
    git=["git","--no-optional-locks","-c",f"safe.directory={G.as_posix()}"]
    subprocess.run([*git,"merge-base","--is-ancestor",args.publication,"refs/remotes/origin/bw1000-experiments"],check=True,capture_output=True,timeout=8)
    for p in (SPEC,PACKAGE/"carrier.py",*(PACKAGE/n for n in claim["sources"])):
        frozen=subprocess.check_output([*git,"cat-file","blob",args.publication+":"+p.relative_to(G).as_posix()],timeout=8)
        assert frozen==p.read_bytes() and (p==SPEC or p.name=="carrier.py" and hash_bytes(frozen)==claim["carrier_SHA256"] or p.name in claim["sources"] and hash_bytes(frozen)==claim["sources"][p.name])
    assert hash_bytes(ROUTE.read_bytes())=="530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
    assert hash_bytes(SEND.read_bytes())=="16563ae8141d79ef83990f8c0c16baf2ad4505a013dbbf4c8bfb9395d038d172"
    request=claim['pre_request'] if args.mode=='pre' else claim['post_request'] if args.mode=='post-sample' else uuid4().hex
    payload={}
    custody=None
    custody_sha=None
    if args.mode in ('pre','launch','post-sample','post'):
        assert args.custody is not None and args.custody.resolve(strict=True).is_relative_to(LOCAL)
        raw=args.custody.read_bytes()
        assert 0<len(raw)<=16384
        custody,custody_sha=json.loads(raw),hash_bytes(raw)
        assert custody['owner_request']==claim['request'] and custody['publication']==args.publication
        assert custody['node']=='f12r2n10' and custody['parent']=='1103606' and custody['exact_root_sole_dispatcher'] is True
        assert custody['ordinary_exclusion_until_independent_post'] is True and custody['allows_one_DCU_once'] is True
        payload.update(custody=custody,custody_SHA256=custody_sha)
    if args.mode=="stage":
        payload["files"]={n:b64encode((PACKAGE/n).read_bytes()).decode() for n in claim["sources"]}
    if args.mode in ("launch","post"):
        assert args.physical_receipt is not None and args.physical_receipt.resolve().is_relative_to(LOCAL)
        x=json.loads(args.physical_receipt.read_bytes())["transport"]
        assert x["remote_outcome"]=="RESPONSE_OBSERVED" and x["rc"]==0 and x["response"]["request"]==x["request"]
        payload["physical"]=x["response"]
    source="\n".join(f"{key}={value!r}" for key,value in dict(REQUEST=request,MODE=args.mode,CLAIM=claim,PUBLICATION=args.publication).items())+"\n"+REMOTE
    if args.mode in ("pre","post-sample"):
        probe_path=G/"tools/run_training80_n02_guarded_refill_once.py"
        assert hash_bytes(probe_path.read_bytes())=="49f637945d2850b81ed666cb6976f25ae57b233babbae021fc806ec3cbcf734c"
        probe_spec=G/"coordination/tasks/training80-f13r1n02-group13-zero-gres-clearance-v1.json"
        assert hash_bytes(probe_spec.read_bytes())=="33431059cf33a06aab2b335429911d216c2a065673ed5646f0f38053adcc90a6"
        node_probe=runpy.run_path(str(probe_path))["NODE_PROBE"]
        pins=json.loads(probe_spec.read_bytes())["staged_probe_SHA256"]
        source="\n".join(f"{k}={v!r}" for k,v in dict(REQUEST=request,PROBE=node_probe,PINS=pins,OLD=claim["incumbent"],
                     OWNER=claim["request"],ORIGINALS=claim["previous_rows"],CUSTODY=custody,CUSTODY_SHA=custody_sha,MODE="pre" if args.mode=="pre" else "post").items())+"\n"+(PACKAGE/"physical.py").read_text(encoding="utf-8")
    compile(source,"<bound Pack16 host carrier>","exec")
    group=LOCAL/LABEL
    group.mkdir(exist_ok=True)
    stem=args.mode if args.mode in ("stage","launch") else f"{args.mode}-{request}"
    data=(json.dumps(payload,sort_keys=True)+"\n").encode()
    intent={"request":request,"owner":claim["request"],"program_SHA256":hash_bytes(source.encode()),"data_SHA256":hash_bytes(data)}
    with (group/f"{stem}-intent.json").open("xb") as h:
        h.write((json.dumps(intent,sort_keys=True)+"\n").encode())
    route=runpy.run_path(str(ROUTE))["ssh"](G.parent)
    proof=runpy.run_path(str(SEND))["send"](route,request=request,program=source,data=data,send_token=group/f"{stem}-send.json",
                                            outcome_seconds=265 if args.mode=="launch" else 50)
    blob=(json.dumps(dict(intent=intent,transport=proof),sort_keys=True)+"\n").encode()
    receipt=group/f"{stem}-reply.json"
    with receipt.open("xb") as stream:stream.write(blob)
    print(json.dumps(dict(receipt=receipt.name,SHA256=hash_bytes(blob),route=proof["remote_outcome"],body=proof["response"])))


if __name__=="__main__":
    main()

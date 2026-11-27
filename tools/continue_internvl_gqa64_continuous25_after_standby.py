"""Continue the original staged GQA25 request at its first controller invocation."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from uuid import uuid4

from launch_internvl_gqa64_first2_once import check_local_holds
from render_internvl_gqa64_first2 import REPO
from run_hygon_bound_payload import digest, document, send


REQUEST = "72fb511d59b14872a2289ddf152343d0"
ORIGINAL_PUBLICATION = "7b1df167d693846f4313ed4becccdbcc487325d1"
HOST = r'''
import fcntl,json,os,re,subprocess,sys,time
from hashlib import sha256
from pathlib import Path

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
original='e23c26259c4344bc837a693ae9c2f11a'
request='72fb511d59b14872a2289ddf152343d0'
label='training-internvl-gqa64-continuous25-hip-v1'
stage=R/(label+'-source-'+original)
assert os.getuid()==154459 and stage.resolve(strict=True)==stage
os.environ.update(TZ='UTC',LC_ALL='C')
packet=json.loads(sys.stdin.buffer.read(4096))
assert (packet['request'],packet['original_publication'])==(request,'7b1df167d693846f4313ed4becccdbcc487325d1')

def read(path,limit):
    assert path.is_file() and path.resolve(strict=True)==path
    with path.open('rb') as stream:raw=stream.read(limit+1)
    assert 0<len(raw)<=limit
    return raw

def digest(raw):
    return sha256(raw).hexdigest()

def put(path,payload):
    with path.open('x') as out:
        json.dump(payload,out,sort_keys=True);out.flush();os.fsync(out.fileno())

def accounting():
    options=['sacct','-nP','-j','1103606','-o',
             'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End']
    lines=subprocess.check_output(options,text=True,timeout=8).splitlines()
    out=[line.split('|') for line in lines]
    assert all(len(row)==9 for row in out)
    return out

with (R/'.training80-dcu-standby.lock').open('ab') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mf=read(stage/'SOURCE_FILES.json',16384)
    assert digest(mf)=='52444c213cac9529bb150e8b5d10aa107c8cc9c29a162a6b490060223ad92ed7'
    manifest=json.loads(mf)
    assert manifest['request']==original and len(manifest['files'])==16
    for name,expected in manifest['files'].items():
        assert Path(name).name==name
        raw=read(stage/name,expected['bytes'])
        assert len(raw)==expected['bytes'] and digest(raw)==expected['SHA256']
    signal=json.loads(read(stage/'standby-signal-result.json',8192))
    signal_intent=json.loads(read(stage/'standby-signal-intent.json',8192))
    assert signal['rc']==0 and signal_intent['request']==original and signal_intent['target']=='1103606.4382'
    assert not (stage/'controller-start-intent.json').exists()
    assert not (R/label).exists() and not (R/(label+'-controller')).exists()
    proof=read(stage/'independent-standby-post.json',16384)
    assert digest(proof)==packet['post_SHA256']
    post=json.loads(proof);worker=post['worker']
    assert post['request']==original and post['worker_rc']==0
    assert post['standby_terminal_UTC'][0]=='1103606.4382' and post['standby_terminal_UTC'][2:4]==['COMPLETED','0:0']
    a=post['accounting_UTC']
    assert a[0]==worker['step'] and a[1]==post['probe_label'] and a[2:4]==['COMPLETED','0:0']
    assert a[5:7]==['f13r1n14','cpu=2,mem=2G,node=1']
    assert worker['pci_count']==8 and worker['same_pci_and_boot_as_4374'] is worker['full_fd_inventory_performed'] is True
    assert worker['kfd_holders']==worker['prior_standby_cgroup_processes']==0 and worker['training_port_bindable']
    assert 0<=time.time()-worker['observed_epoch']<=1800
    current=accounting()
    matching=[row for row in current if row[0] in {a[0],'1103606.4382'}]
    assert len(matching)==2 and post['standby_terminal_UTC'] in matching and a in matching
    assert not any(row[1].startswith(label+'-') for row in current)
    sys.path.insert(0,str(stage))
    from control_once import placement
    from native import binding
    from node import actual_model_plan
    placement()
    claim_raw=read(stage/'CLAIM.json',16000)
    claim=json.loads(claim_raw)
    assembler,plan,view=actual_model_plan(claim['exports_SHA256'])
    assembler.verify_originals(plan)
    b=binding()
    assert len(b['domain_source_rows'])==64 and len(view)==20 and len(b['trainable_manifest'])==504
    put(stage/'controller-start-intent.json',{'request':original,'continuation_request':request,
        'independent_post_step':a[0],'independent_post_SHA256':packet['post_SHA256'],'epoch':time.time()})
    args=[sys.executable,'-B',str(stage/'control_once.py'),'--publication',packet['original_publication'],
          '--claim-sha256',digest(claim_raw),'--manifest-sha256',digest(mf),'--exports-sha256',claim['exports_SHA256']]
    child=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,timeout=45)
    for name,data in [('continuation.stdout',child.stdout),('continuation.stderr',child.stderr)]:
        with (stage/name).open('xb') as out:out.write(data)
    outcome={'request':request,'original_request':original,'rc':child.returncode,
             'out_SHA256':digest(child.stdout),'err_SHA256':digest(child.stderr),'post_step':a[0]}
    if child.returncode==0:
        controller=json.loads(child.stdout)
        assert controller['request']==original and controller['state']=='CONTROLLER_STARTED'
        outcome['controller']=controller
    put(stage/'continuation-result.json',outcome)
    print(json.dumps(outcome,sort_keys=True),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--publication")
    parser.add_argument("--post-sha256")
    args=parser.parse_args()
    compile(HOST,"<GQA25 after clean standby>", "exec")
    if not re.fullmatch("[0-9a-f]{32}", REQUEST):
        raise ValueError("Invalid continuation transport identity")
    if args.check:
        print(json.dumps({"request":REQUEST,"host_SHA256":digest(HOST.encode()),"remote_calls":0}))
        return 0
    if not re.fullmatch("[0-9a-f]{40}",args.publication) or not re.fullmatch("[0-9a-f]{64}",args.post_sha256 or ""):
        raise ValueError("Published continuation and actual post receipt digest required")
    repo=REPO
    for p in (Path(__file__).resolve(),repo/"tools/run_hygon_bound_payload.py"):
        key=p.relative_to(repo).as_posix()
        found=subprocess.check_output(["git","-c",f"safe.directory={repo.as_posix()}","show",
                                       args.publication+":"+key],cwd=repo)
        if found!=p.read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("Published continuation source differs")
    check_local_holds()
    local=repo/"local-only"
    helper=local/"run-minicpm-sdpa-target-asset-preflight-original.py"
    raw=helper.read_bytes()
    if digest(raw)!="530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing project transport differs")
    transport=ModuleType("existing_ssh")
    exec(compile(raw,str(helper),"exec"),vars(transport))
    data=document({"request":REQUEST,"original_publication":ORIGINAL_PUBLICATION,"post_SHA256":args.post_sha256})
    result=send(transport.ssh(repo.parent),request=REQUEST,program=HOST,data=data,
                send_token=local/f"internvl-gqa25-continuation-{REQUEST}-send.json",outcome_seconds=90)
    path=local/f"internvl-gqa25-continuation-{REQUEST}-{uuid4().hex}.json"
    with path.open("xb") as out:
        out.write(document(result))
    print(json.dumps({"receipt":path.name,"receipt_SHA256":digest(path.read_bytes()),**result},sort_keys=True))
    return int(result["remote_outcome"]!="RESPONSE_OBSERVED" or result["response"].get("rc")!=0)


if __name__=="__main__":
    raise SystemExit(main())

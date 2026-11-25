"""Execute one published CPU-only native sampler witness or independently read its terminal."""
import argparse
from base64 import b64decode, b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "local-only"
PACKAGE = Path(__file__).resolve().parent
LABEL = "training-gemma-usgs-r28-sampler-native-CPU-v1"
CLAIM = ROOT / "coordination/tasks" / f"{LABEL}.json"
R27 = ROOT / "code/bw1000/training-gemma-usgs-r26-native-image-CPU-v1/host_launch_once.py"
R27_SHA = "9c11255b163faf83aa9c2fc2b528312f2939900a81b07f122a0f234dd2ea7db4"
BUNDLE = ("probe.py", "SOURCE_FILES.json", "command.py")
HELPER = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
TRANSPORT = ROOT / "tools/run_hygon_bound_payload.py"


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Pinned source changed: {path.name}")
    return raw


def render(mode, transport_id, publication, claim, claim_sha, r27_sha):
    values = dict(MODE=mode, ID=transport_id, PUBLICATION=publication, CLAIM=claim, CLAIM_SHA=claim_sha, R27_SHA=r27_sha)
    return "\n".join(f"{key}={value!r}" for key, value in values.items()) + "\n" + HOST


HOST = r"""
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

def need(condition, reason):
    if not condition:
        raise ValueError(reason)

def digest(raw):
    return sha256(raw).hexdigest()

def original_packet():
    raw = sys.stdin.buffer.read(120001)
    need(0 < len(raw) <= 120000, "BOUNDED_SOURCE_ONLY_PACKET")
    packet = json.loads(raw)
    need(set(packet) == {'claim','r27','sources'}, "EXACT_SOURCE_ONLY_PACKET")
    claim_raw = b64decode(packet['claim'],validate=True)
    previous_host = b64decode(packet['r27'],validate=True)
    sources = {key:b64decode(value,validate=True) for key,value in packet['sources'].items()}
    need(digest(claim_raw)==CLAIM_SHA and json.loads(claim_raw)==CLAIM and digest(previous_host)==R27_SHA,
         "EXACT_PRIOR_HOST_OR_CURRENT_CLAIM_SOURCE")
    need(set(sources)=={'probe.py','SOURCE_FILES.json','command.py'}
         and {key:digest(raw) for key,raw in sources.items()}==CLAIM['sources'], "EXACT_RELEASED_SOURCE_BYTES")
    manifest = json.loads(sources['SOURCE_FILES.json'])
    need(manifest['files']=={'probe.py': {'bytes': len(sources['probe.py']), 'SHA256': digest(sources['probe.py'])}},
         "EXACT_COMPUTE_MANIFEST_MEMBERS")
    host = {'__name__':'r27_admission_reused'}
    exec(compile(previous_host,'<R27 published HOST functions>','exec'),host)
    argv = {'__name__':'r28_pinned_command', '__file__':'/source-only/command.py'}
    exec(compile(sources['command.py'],'<R28 published CPU command>','exec'),argv)
    return host,argv,sources

def select_account(query, argv, job, parent, node):
    all_rows=[line.split('|') for line in query(*argv['accounting_argv'](parent)[3:])]
    need(all(len(row)==9 for row in all_rows),"EXACT_ACCOUNTING_COLUMNS")
    rows=[row for row in all_rows if row[1]==job]
    need(len(rows)<=1 and all(re.fullmatch(parent+r'[.][0-9]+',row[0]) and row[5]==node for row in rows),
         "EXACT_ORIGINAL_ACCOUNTING_IDENTITY")
    return rows

def project(result):
    visible=('state','phase','reason','passed','step','actual_native_rank_projections','actual_distinct_physical_slots',
             'actual_swift_physical_rows','actual_gpu_steps','actual_optimizer_updates','actual_live_training_ranks',
             'training_ready','private_projection')
    return {key:result[key] for key in visible if key in result}

def main():
    scope,parent,node,label = (CLAIM[key] for key in ('request','parent','node','label'))
    need(os.getuid()==154459 and (parent,node)==('1103606','f13r1n12')
         and label=='training-gemma-usgs-r28-sampler-native-CPU-v1'
         and CLAIM['target_GPU']==0 and CLAIM['CPU']==4 and CLAIM['memory_MiB']==8192
         and CLAIM['work_seconds']==120 and CLAIM['max_target_steps']==1 and CLAIM['replay_allowed'] is False,
         "EXACT_PUBLISHED_R28_ZERO_GRES")
    host,argv,sources=original_packet()
    results=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
    run=results/label/scope
    job=label+'-'+scope[:12]
    host.update(JOB=job,REQUEST=scope,LABEL=label,RUN=run)
    env=host['host_environment']()
    os.umask(0o077)
    start=time.monotonic()
    def query(*args):
        need(args[0] in ('sacct','scontrol','squeue'),"READ_ONLY_SCHEDULER")
        limit=min(8,30-(time.monotonic()-start))
        need(limit>0,"BOUNDED_SCHEDULER_READ")
        call=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=limit,env=env)
        need(len(call.stdout)<=1048576,"BOUNDED_SCHEDULER_METADATA")
        return call.stdout.decode('ascii').splitlines()
    if MODE=='terminal':
        accounting=select_account(query,argv,job,parent,node)
        result={'state':'ORIGINAL_ACCOUNTING_NOT_VISIBLE','accounting':accounting,'replay_allowed':False}
        if os.path.lexists(run/'out/RESULT.json'):
            read=host['hash_read']
            native_raw=read(run/'out/RESULT.json')
            native=json.loads(native_raw)
            need(native['request']==scope and native['node']==node,"SAME_NATIVE_REQUEST_IDENTITY")
            result.update(state='ORIGINAL_NATIVE_RESULT_READ',worker=project(native),worker_SHA256=digest(native_raw))
            if accounting:
                row,=accounting
                need(native['step']==row[0],"SAME_NATIVE_STEP_IDENTITY")
                if row[2:4]==['COMPLETED','0:0']:
                    initial=json.loads(read(run/'HOST_INTENT.json'))
                    private=read(run/'out/PRIVATE_CPU_NATIVE_ORDER.json')
                    result['independent']=argv['independent'](row,native,private,parent,node,scope,initial['epoch'],CLAIM['sources']['SOURCE_FILES.json'])
                elif row[2].split()[0] in host['TERMINAL']:
                    result['state']='ORIGINAL_EXACT_NUMERIC_CPU_TERMINAL_FAIL'
        return result
    need(MODE=='launch' and not os.path.lexists(run) and not select_account(query,argv,job,parent,node),
         "ORIGINAL_ALREADY_PRESENT_RECONCILE_ONLY")
    placement=host['fresh_slots'](query,time.time())
    need(not select_account(query,argv,job,parent,node),"ORIGINAL_LABEL_VISIBLE_AFTER_PLACEMENT")
    run.parent.mkdir(exist_ok=True)
    need(run.parent.resolve(strict=True)==run.parent,"EXACT_CPU_RUN_PARENT")
    run.mkdir()
    (run/'src').mkdir()
    (run/'out').mkdir()
    for name in ('probe.py','SOURCE_FILES.json'):
        with (run/'src'/name).open('xb') as file:
            file.write(sources[name])
            file.flush()
            os.fsync(file.fileno())
    now=time.time()
    command=argv['runner'](parent,node,scope,now,CLAIM['sources']['SOURCE_FILES.json'])
    need(command[0]=='srun' and '--gres=none' in command and '--overlap' not in command
         and '--cpus-per-task=4' in command and '--mem=8192M' in command and now+120<=placement['parent_end_epoch'],
         "EXACT_ZERO_GRES_ARGV_AND_EPOCH")
    binds=[Path(command[i+1].split(':',1)[0]) for i,arg in enumerate(command) if arg=='--bind']
    need(len(binds)==11 and all(p.resolve(strict=True)==p and (p.is_dir() or p.is_file()) for p in binds)
         and Path(command[-6]).is_dir() and Path(command[command.index('--time=00:03:00')+1]).is_file(),
         "EXACT_REAL_SAMPLER_MOUNTS")
    intent={'scope_request':scope,'publication':PUBLICATION,'epoch':now,'deadline':now+120,'parent':parent,'node':node,
            'argv_SHA256':digest(host['doc'](command)), 'claim_SHA256':CLAIM_SHA,'placement':placement,'replay_allowed':False}
    host['save'](run,'HOST_INTENT.json',intent)
    attempt=False
    try:
        with (run/'HOST.stdout').open('xb') as out,(run/'HOST.stderr').open('xb') as err:
            remaining=now+120-time.time()
            need(remaining>0,"ORIGINAL_CPU_EPOCH_EXPIRED")
            attempt=True
            child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=remaining,env=env,check=False)
            observation=dict(state='CPU_SRUN_RETURNED_REQUIRES_INDEPENDENT_TERMINAL',child_rc=child.returncode)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        observation=dict(state='CPU_SEND_OUTCOME_REQUIRES_RECONCILIATION',error_class=type(error).__name__)
    observation.update(attempted=attempt,placement=placement,replay_allowed=False)
    host['save'](run,'HOST_RETURN.json',observation)
    return observation

try:
    report=main()
except (OSError,ValueError,KeyError,TypeError,AttributeError,subprocess.SubprocessError) as fault:
    report={'state':'CPU_REQUEST_BLOCKED_RECONCILE_IF_INTENT_EXISTS','error_class':type(fault).__name__,
            'reason':str(fault) if type(fault) is ValueError and re.fullmatch(r'[A-Z0-9_]+',str(fault)) else None,
            'replay_allowed':False}
print(json.dumps({**report,'request':ID,'scope_request':CLAIM['request']},sort_keys=True,allow_nan=False),flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "launch", "terminal"))
    parser.add_argument("--publication")
    args = parser.parse_args()
    claim_raw = CLAIM.read_bytes()
    claim = json.loads(claim_raw)
    pinned(Path(__file__), claim['host_sampler_SHA256'])
    prior = pinned(R27, R27_SHA)
    source = {name: pinned(PACKAGE / name, claim["sources"][name]) for name in BUNDLE}
    request = uuid4().hex
    program = render(args.mode, request, args.publication, claim, digest(claim_raw), R27_SHA)
    compile(program, "<published sampler CPU admission>", "exec")
    if args.mode=="check":
        command=runpy.run_path(str(PACKAGE/"command.py"))
        example=command["runner"](claim["parent"],claim["node"],claim["request"],1795590000,claim["sources"]["SOURCE_FILES.json"])
        assert "--gres=none" in example and "--overlap" not in example and len([x for x in example if x=="--bind"])==11
        print(json.dumps(dict(source_members=3,planned_CPU=4,GRES=0,source_manifest=claim["sources"]["SOURCE_FILES.json"])))
        return
    if not args.publication or not re.fullmatch(r"[0-9a-f]{40}", args.publication):
        raise ValueError("Ordinary published scope required")
    git = ["git", "--no-optional-locks", "-c", f"safe.directory={ROOT.as_posix()}"]
    subprocess.run([*git,"merge-base","--is-ancestor",args.publication,"refs/remotes/origin/bw1000-experiments"],cwd=ROOT,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=8)
    for path in (CLAIM,R27,Path(__file__),*(PACKAGE/name for name in BUNDLE)):
        current=path.read_bytes()
        uploaded=subprocess.check_output([*git,"cat-file","blob",args.publication+":"+path.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=8)
        if uploaded!=current:
            raise ValueError("Published source or claim differs")
    transport=runpy.run_path(str(TRANSPORT))["send"] if pinned(TRANSPORT,"16563ae8141d79ef83990f8c0c16baf2ad4505a013dbbf4c8bfb9395d038d172") else None
    pinned(HELPER,"530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd")
    route=runpy.run_path(str(HELPER))["ssh"](ROOT.parent)
    inbox=LOCAL / LABEL
    inbox.mkdir(exist_ok=True)
    stem="single-CPU-launch" if args.mode=="launch" else f"terminal-{request}"
    package={"claim":b64encode(claim_raw).decode(),"r27":b64encode(prior).decode(),
             "sources":{name:b64encode(raw).decode() for name,raw in source.items()}}
    data=(json.dumps(package,sort_keys=True)+"\n").encode()
    intent={"transport_request":request,"scope_request":claim["request"],"mode":args.mode,"publication":args.publication,
            "program_SHA256":digest(program.encode()),"payload_SHA256":digest(data),"replay_allowed":False}
    with (inbox/f"{stem}-intent.json").open("xb") as file:
        file.write((json.dumps(intent,sort_keys=True)+"\n").encode())
    result=transport(route,request=request,program=program,data=data,send_token=inbox/f"{stem}-send.json",
                     outcome_seconds=154 if args.mode=="launch" else 42)
    receipt=(json.dumps(dict(intent=intent,transport=result),sort_keys=True)+"\n").encode()
    path=inbox/f"{stem}-receipt.json"
    with path.open("xb") as file:
        file.write(receipt)
    print(json.dumps(dict(receipt=path.name,SHA256=digest(receipt),route_state=result["remote_outcome"],body=result["response"])))


if __name__=="__main__":
    main()

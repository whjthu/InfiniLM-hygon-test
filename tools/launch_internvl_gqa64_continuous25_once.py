"""Take over the exact owned standby and run one published InternVL GQA25 job."""
import argparse
from base64 import b64encode
import gzip
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from uuid import uuid4

import launch_internvl_gqa64_first2_once as previous
import render_internvl_gqa64_continuous25 as source
from run_hygon_bound_payload import digest, document, send


FENCE = "d6dafa1c160720a0fcb4555e6dee15b684b0fcc6085642ee35fbb22ca3773596"
STANDBY_TO_TRAINING = r'''
parent,node='1103606','f13r1n14'
standby_id='1103606.4382'
standby_request='e6c3cef544e94af58e9d718d38f0861d'
standby_label='training80-dcu-standby-'+standby_request[:12]+'-'+node
standby=R/('training80-dcu-standby-'+standby_request)/node
cols='JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'

def query(*args):
    return subprocess.check_output(args, text=True, timeout=8).splitlines()

def fields(line):
    return dict(term.split('=',1) for term in line.split() if '=' in term)

def save(path,value):
    with path.open('x') as out:
        json.dump(value,out,sort_keys=True)
        out.flush()
        os.fsync(out.fileno())

def pinned(path,expected=None):
    assert path.is_file() and not path.is_symlink()
    with path.open('rb') as f:raw=f.read(12001)
    assert 0<len(raw)<=12000 and (expected is None or sha(raw)==expected)
    return json.loads(raw)

def accounts(handle):
    lines=[line.split('|') for line in query('sacct','-nP','-j',handle,'-o',cols)]
    assert all(len(line)==9 for line in lines)
    return lines

def standby_state():
    [job]=map(fields,query('scontrol','show','job','-o',parent))
    assert job['JobId']==parent and job['JobState']=='RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
    assert 'gres/dcu=320' in job['AllocTRES'].split(',')
    hosts={}
    def members(spec):
        assert spec and spec not in {'(null)','Unknown','N/A','None'}
        if spec not in hosts:
            hosts[spec]=[spec] if re.fullmatch(r'f\d+r\d+n\d+',spec) else query('scontrol','show','hostnames',spec)
        return hosts[spec]
    assert node in members(job['NodeList'])
    active={}
    for step in map(fields,query('scontrol','show','step',parent,'--oneliner')):
        identity=step['StepId']
        if re.fullmatch(parent+r'\.\d+',identity) and node in members(step['NodeList']):
            assert members(step['NodeList'])==[node] and identity not in active
            active[identity]=(step['Name'],step['State'])
    queued=set()
    for line in query('squeue','--steps','-h','-j',parent,'-o','%i|%N'):
        identity,hosts_text=line.split('|')
        if re.fullmatch(parent+r'\.\d+',identity) and node in members(hosts_text):
            queued.add(identity)
    assert set(active)<={standby_id} and queued<={standby_id},'another work step owns the target'
    [row]=[r for r in accounts(standby_id) if r[0]==standby_id]
    assert row[1]==standby_label and row[5:7]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1']
    return row,active,queued

def replace_exact_standby():
    [post]=[r for r in accounts('1103606.4374') if r[0]=='1103606.4374']
    [scratch]=[r for r in accounts('1103606.4380') if r[0]=='1103606.4380']
    assert post[2:4]==scratch[2:4]==['COMPLETED','0:0'] and post[5]==scratch[5]==node
    assert 'gres' not in post[6] and 'gres' not in scratch[6]
    row,active,queued=standby_state()
    expected=pinned(standby/'expected.json','815008ee9328db11c61ec3de4ff1215e0590913a3bb9fa9c907952af6bc5a8db')
    worker=pinned(standby/'worker.json','eb56ec109e99cf03c34832a1b1da167dddb5cff61b146f2e0c661e7174f074cc')
    assert expected['label']==standby_label and expected['node']==worker['node']==node
    assert expected['request']==standby_request and expected['parent']==parent
    assert worker['step']==standby_id and worker['synthetic'] is True and worker['worker_pid']==2744358
    assert len(set(worker['pci']))==8 and set(worker['pci'])==set(expected['pci'])
    if row[2]=='RUNNING':
        assert active=={standby_id:(standby_label,'RUNNING')} and not (standby/'terminal.json').exists()
        for index,pci in enumerate(worker['pci']):
            card=pinned(standby/f'card-{index}.json')
            assert card['pci']==pci and card['synthetic'] is card['active'] is card['ready'] is True
            assert card['calls']>0 and 0<=time.time()-card['epoch']<25
        save(stage/'standby-signal-intent.json',{'target':standby_id,'accounting_UTC':row,'request':REQUEST})
        result=subprocess.run(['scancel','--signal=TERM',standby_id],stdin=subprocess.DEVNULL,
                              capture_output=True,timeout=8)
        save(stage/'standby-signal-result.json',{'rc':result.returncode,'err_SHA256':sha(result.stderr)})
        assert result.returncode==0
    deadline=time.monotonic()+80
    while True:
        row,active,queued=standby_state()
        if not active and not queued and row[2] in {'COMPLETED','CANCELLED'} and row[3]=='0:0' and row[8]!='Unknown':
            proof=pinned(standby/'terminal.json')
            assert (proof['step'],proof['state'],proof['node'])==(standby_id,'WORKER_RETURNED',node)
            save(stage/'standby-graceful-terminal.json',{'request':REQUEST,'accounting_UTC':row,'worker':proof})
            return row
        assert time.monotonic()<deadline,'exact standby did not return normally'
        time.sleep(1.5)
'''


def host():
    program = previous.HOST
    updates = [
        (previous.source.REQUEST, source.REQUEST), (previous.source.LABEL, source.LABEL),
        ("len(files)==16", "len(files)==17"),
        ("claim['planned_updates']==2 and claim['planned_nonzero_updates']==1 and claim['planned_microbatches']==64",
         "claim['planned_updates']==25 and claim['planned_nonzero_updates']==24 and claim['planned_microbatches']==800"),
        ("stage=R/(LABEL+'-source-'+REQUEST)", "stage=R/(LABEL+'-source-'+REQUEST)\n" + STANDBY_TO_TRAINING),
        ("    placement()\n    assert all(accounting", "    assert all(accounting"),
        ("    with (stage/'controller-start-intent.json').open('x') as out:",
         "    prior=replace_exact_standby()\n    placement()\n    with (stage/'controller-start-intent.json').open('x') as out:"),
        ("result={'request':REQUEST,'rc':child.returncode,", "result={'request':REQUEST,'rc':child.returncode,'predecessor_terminal':prior,"),
    ]
    for old,new in updates:
        program=source.first.replace(program,old,new)
    compile(program,"<InternVL GQA25 host controller>","exec")
    return program


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--publication")
    args=parser.parse_args()
    payload_host=host()
    files=source.render()
    if args.check:
        print(json.dumps({"request":source.REQUEST,"host_SHA256":digest(payload_host.encode()),
                          "manifest_SHA256":digest(files["SOURCE_FILES.json"]),"remote_calls":0}))
        return 0
    if not re.fullmatch("[0-9a-f]{40}",args.publication):
        raise ValueError("Exact published source revision required")
    repo=source.first.REPO
    paths=(Path(__file__),Path(source.__file__),source.ARTIFACT_READER,
           Path(previous.__file__),Path(source.first.__file__),source.first.ROSTER,
           repo/"tools/run_hygon_bound_payload.py")
    for path in paths:
        key=path.resolve().relative_to(repo).as_posix()
        found=subprocess.check_output(["git","-c",f"safe.directory={repo.as_posix()}","show",
                                       args.publication+":"+key],cwd=repo)
        if found!=path.read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("GQA25 publication and current bytes differ")
    previous.check_local_holds()
    local=repo/"local-only"
    helper=local/"run-minicpm-sdpa-target-asset-preflight-original.py"
    raw=helper.read_bytes()
    if digest(raw)!="530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing project transport differs")
    transport=ModuleType("existing_ssh")
    exec(compile(raw,str(helper),"exec"),vars(transport))
    data={"publication":args.publication,"fence_SHA256":FENCE,
          "files":{key:b64encode(contents).decode() for key,contents in files.items()}}
    result=send(transport.ssh(repo.parent),request=source.REQUEST,program=payload_host,
                data=b64encode(gzip.compress(document(data),mtime=0)),
                send_token=local/f"internvl-gqa25-{source.REQUEST}-send.json",outcome_seconds=160)
    path=local/f"internvl-gqa25-{source.REQUEST}-receipt-{uuid4().hex}.json"
    with path.open("xb") as out:
        out.write(document(result))
    print(json.dumps({"receipt":path.name,"receipt_SHA256":digest(path.read_bytes()),**result},sort_keys=True))
    return int(result["remote_outcome"]!="RESPONSE_OBSERVED" or result["response"].get("rc")!=0)


if __name__=="__main__":
    raise SystemExit(main())

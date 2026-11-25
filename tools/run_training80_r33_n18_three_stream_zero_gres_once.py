"""Collect separate stderr channels for one new zero-GRES diagnostic on f12r3n18."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import FunctionType


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r33-n18-three-stream-zero-gres-v1'
WRAPPER=ROOT/'tools/run_training80_r33_n18_three_stream_zero_gres_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
DIAG=ROOT/'tools/run_training80_r29_warning_zero_gres_once.py'
DIAG_SHA='f294f01800f1859b16a3da72702afd7f7034c14c141b53bd52816d897cd54829'
R31=ROOT/'tools/run_training80_r29_other11_warning_guarded_refill_once.py'
R31_SHA='c3162682176376fcd84099c09372ec3e3010de54c106e04278ed3752e132159b'
JOURNAL=LOCAL/'training80-r31-other11-guarded-first-controller-d047aecf163e4b6a8a769a0b624f1edf.jsonl'
JOURNAL_SHA='19d3633df2ac2aef4970ee3e5cd04e4ddf8dc05ad1188e06e1bb6a0cf03578dd'
NODE,PARENT,PREVIOUS='f12r3n18','1194900','1194900.1277'
OLD='4d4e0f3f8ced5f41af7542c22e023e8a'
OLD_SCOPE=f'training80-r29-other11-warning-guarded-bf16-v1-{NODE}-{OLD}'
REQUEST='d1783c8023ad4708ac6ae56b97a79a9d'
OLD_REPLY_SHA='0d44fab3d56986003528fb6bf305090bc3b6bcd294b4c2e687db45cc9184568e'
OLD_PROGRAM_SHA='f47143d657945e59a8c77e79f5eb7dd7dcf22d45283dc7f49ba5ff633ab63a6e'


CHILD=r'''
import os,sys
from pathlib import Path

binary,slot,program,*probe_args=sys.argv[1:]
log=Path(slot)
assert log.is_absolute() and log.parent.resolve(strict=True)==log.parent
assert log.parent.stat().st_uid==os.getuid() and log.name=='child-probe.stderr.private.raw'
fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.dup2(fd,2,inheritable=True)
os.close(fd)
os.execv(binary,[binary,'-I','-S','-B','-c',program,*probe_args])
'''


STREAMS=r'''
def classify_slot(path):
    try:
        info=path.lstat()
    except FileNotFoundError:
        return dict(available=False,kind='UNOBSERVED',exact_SHA256=None,exact_bytes=None,matches_prior_rc0_stderr=False)
    if path.resolve(strict=True)!=path or not path.is_file() or info.st_uid!=os.getuid() or info.st_size>65536:
        return dict(available=False,kind='UNQUALIFIED_OR_OVERSIZE',exact_SHA256=None,exact_bytes=info.st_size,matches_prior_rc0_stderr=False)
    with path.open('rb') as handle:
        raw=handle.read(65537)
    require(len(raw)==info.st_size and len(raw)<=65536,'stderr file changed or exceeded bound')
    classified=classify_stderr(raw)
    # Only bounded structural category and exact fingerprint are returned, never text.
    return dict(available=True,kind=classified['standard_type'],exact_SHA256=classified['exact_SHA256'],
                exact_bytes=classified['exact_bytes'], matches_prior_rc0_stderr=classified['matches_prior_rc0_stderr'])


def route_observation(login,slurm_task,child):
    observations={'login_srun':classify_slot(login),'slurm_task':classify_slot(slurm_task),'child_probe':classify_slot(child)}
    matching=[name for name,record in observations.items() if record['matches_prior_rc0_stderr']]
    if len(matching)==1:
        state='EXACT_OLD_FINGERPRINT_IN_ONE_CHANNEL'
    elif matching:
        state='EXACT_OLD_FINGERPRINT_IN_MULTIPLE_CHANNELS'
    else:
        state='OLD_FINGERPRINT_NOT_REPRODUCED_OR_UNOBSERVED'
    return dict(channels=observations,matching_channels=matching,route_status=state,process_origin_proven=False,
                warning_gate_changed=False,GPU_admission_eligible=False)
'''


def digest(raw):
    return sha256(raw).hexdigest()


def need(ok,message):
    if not ok:
        raise ValueError(message)


def pin(path,wanted=None):
    need(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'exact local proof path differs')
    data=path.read_bytes()
    need(0<len(data)<=262144 and (wanted is None or digest(data)==wanted),'pinned local evidence differs')
    return data


def evidence():
    pin(DIAG,DIAG_SHA); pin(R31,R31_SHA)
    diag=runpy.run_path(str(DIAG))
    r31=runpy.run_path(str(R31)); old=r31['bound_api'](); ctx=old['load'](); row=ctx[-2][NODE]
    matches=[json.loads(line) for line in pin(JOURNAL,JOURNAL_SHA).splitlines()]
    [line]=[entry for entry in matches if entry['node']==NODE]
    require_path=LOCAL/f'{OLD_SCOPE}-reply.json'
    receipt=line['dispatch']['out']; previous=line['predecessor']['out']
    need(receipt['receipt']==require_path.name and receipt['SHA256']==OLD_REPLY_SHA and line['request']==row['new_request']==OLD,
         'exact R31 journal/guard identity differs')
    reply=json.loads(pin(require_path,OLD_REPLY_SHA))
    wire=reply['transport']; recorded=wire['response']
    predecessor=json.loads(pin(LOCAL/previous['path'],previous['SHA256']))
    account,terminal=old['terminal_identity'](row,line['nonce'],predecessor)
    original=old['render'](ctx,row,account,terminal,reply['publication'])['launch']
    need(reply['program_SHA256']==digest(original.encode())==OLD_PROGRAM_SHA and reply['terminal_read_id']==line['nonce']
         and reply['terminal_read_SHA256']==previous['SHA256'] and previous['old_terminal_proven'] is True
         and wire['authority_verified'] is wire['remote_received'] is True and wire['rc']==0 and wire['remote_outcome']=='RESPONSE_OBSERVED'
         and wire['request']==recorded['request']==OLD and recorded['node']==NODE and recorded['parent']==PARENT
         and recorded['state']=='GPU_NOT_SENT' and recorded['probe_rc']==0 and recorded['zero_GRES_call_attempted'] is True
         and recorded['GPU_srun_attempted'] is False and recorded['probe_stderr_SHA256']==diag['OLD_STDERR']
         and recorded['detail_SHA256']==diag['OLD_DETAIL'] and 'probe_step' not in recorded
         and all(key.endswith('SHA256') for key in recorded if 'stderr' in key or 'stdout' in key),
         'original R31 authenticated rejection differs')
    marker='require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,'
    need(original.count(marker)==1 and 'probe_stderr_SHA256=sha256(observation.stderr).hexdigest()' in original
         and 'stderr.private.raw' not in original and 'observation.stderr)' in original,
         'cannot prove that original strict guard retained a fingerprint only')
    # Owner/placement/PCI come from the published ordinary guard, never from stderr.
    _,base_api,base_ctx,_,diag_claim=diag['scope']()
    need(row['parent']==PARENT and row['node']==NODE and row['previous']==PREVIOUS and len(row['pci'])==8,
         'target prior ordinary scope differs')
    return diag,r31,base_api,base_ctx,row,recorded,account


def claim_for(diag):
    return dict(scope=STEM,state='PREPARED_ONLY_THREE_STDERR_CHANNEL_DIAGNOSTIC',launcher=WRAPPER.relative_to(ROOT).as_posix(),
                previous=PREVIOUS,node=NODE,parent=PARENT,request=REQUEST,prior_R31_request=OLD,prior_R31_reply_SHA256=OLD_REPLY_SHA,
                prior_R31_program_SHA256=OLD_PROGRAM_SHA,prior_R31_journal_SHA256=JOURNAL_SHA,prior_stderr_SHA256=diag['OLD_STDERR'],
                CPU=1,memory_MiB=256,Slurm_step_seconds=60,max_invocations=1,GPU=0,GPU_admission_eligible=False,replay_allowed=False,
                remote_only_raw_streams=['login_srun','slurm_task','child_probe'],prior_original_raw_content_available=False)


def prepared(args):
    diag,r31,api,ctx,row,recorded,account=evidence()
    raw=pin(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    need(json.loads(raw)==claim_for(diag),'new single-node zero-GRES claim differs')
    # The old routine is reused only to produce source and never to send its original scope.
    legacy_material=({NODE:(OLD_REPLY_SHA,recorded)},api,ctx,row,digest(raw))
    base=diag['render']
    adapted=FunctionType(base.__code__,dict(base.__globals__,NODE=NODE,PARENT=PARENT,PREVIOUS=PREVIOUS,REQUEST=REQUEST,STEM=STEM))
    host=adapted(legacy_material,args.publication)
    host=api['unique'](host,f'OLD_SCOPE={"training80-r25-last40-ordinary-guarded-bf16-v1-"+NODE+"-"+OLD!r}',f'OLD_SCOPE={OLD_SCOPE!r}')
    old_label='training80-r29-warning-'+REQUEST[:12]
    host=api['unique'](host,f'LABEL={old_label!r}',f'LABEL={"training80-r33-stderr-"+REQUEST[:12]!r}')
    host=api['unique'](host,'\ndef main():',STREAMS+'\n'+f'CHILD={CHILD!r}\n'+'\ndef main():')
    host=api['unique'](host,"            probe=run(command,18)\n            write_raw(root/'stderr.private.raw',probe.stderr)",
        "            slurm_stream=root/'slurm-task.stderr.private.raw'\n"
        "            child_stream=root/'child-probe.stderr.private.raw'\n"
        "            login_stream=root/'login-srun.stderr.private.raw'\n"
        "            require(not any(os.path.lexists(p) for p in (slurm_stream,child_stream,login_stream)),'new channel namespace already used')\n"
        "            python='/usr/local/python3.12/bin/python3'\n"
        "            ix=command.index(python)\n"
        "            command=command[:ix]+['--error='+str(slurm_stream)]+[python,'-I','-S','-B','-c',CHILD,python,str(child_stream),NODE_PROBE]+command[ix+6:]\n"
        "            probe=run(command,20)\n"
        "            write_raw(login_stream,probe.stderr)")
    host=api['unique'](host,'probe_rc=probe.returncode,stderr_projection=classify_stderr(probe.stderr),',
                                    'probe_rc=probe.returncode,stderr_route=route_observation(login_stream,slurm_stream,child_stream),')
    need(host.count("'--gres=none'")==1 and not re.search(r'subprocess\.Popen|controller\s*\[|--gres=dcu:',host),
         'new diagnosis unexpectedly permits GPU')
    compile(host,'<exact R33 three-channel solver>', 'exec')
    return diag,r31,legacy_material,host


def fixtures(host):
    from contextlib import redirect_stdout
    from io import StringIO
    from unittest.mock import patch
    import os

    # Exercise the actual child envelope locally with a harmless substitute probe.
    with TemporaryDirectory() as temp:
        file=Path(temp).resolve()/'child-probe.stderr.private.raw'
        mock_text=b'import sys; print("structured"); print("child only",file=sys.stderr)'
        mock_probe=f"exec(bytes.fromhex('{mock_text.hex()}'))"
        py=Path(sys.executable).resolve()
        compatible=CHILD.replace('log.parent.stat().st_uid==os.getuid()','True').replace('|os.O_NOFOLLOW','')
        call=subprocess.run([str(py),'-I','-S','-B','-c',compatible,str(py),str(file),mock_probe],capture_output=True,timeout=6)
        need(call.returncode==0 and call.stdout.splitlines()==[b'structured'] and not call.stderr
             and file.read_bytes().splitlines()==[b'child only'],'child vs parent stderr channel fixture differs')
        funcs={'classify_slot','route_observation','classify_stderr','diagnose_stderr'}
        fitted=[obj for obj in ast.parse(host).body if isinstance(obj,ast.FunctionDef) and obj.name in funcs]
        scope=dict(sha256=sha256,re=re,os=os,OLD_STDERR=digest(file.read_bytes()),require=need)
        exec(compile(ast.Module(body=fitted,type_ignores=[]),'<actual bounded route projection>','exec'),scope)
        login=file.with_name('login');login.write_bytes(b'')
        slurm=file.with_name('missing')
        with patch.object(os,'getuid',lambda:file.stat().st_uid,create=True):
            got=scope['route_observation'](login,slurm,file)
        need(got['matching_channels']==['child_probe'] and got['channels']['slurm_task']['available'] is False
             and got['channels']['login_srun']['exact_bytes']==0 and got['process_origin_proven'] is False
             and 'child only' not in json.dumps(got),'bounded three-stream evidence fixture differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--publication')
    args=parser.parse_args()
    diag,r31,material,host=prepared(args)
    fixtures(host)
    if args.check:
        result=dict(request=REQUEST,node=NODE,parent=PARENT,previous=PREVIOUS,host_SHA256=digest(host.encode()),claim_SHA256=material[-1],
                    no_original_raw_content_retained_by_guard=True,offline_channel_fixtures=2,GPU=0,Git=0,SCNet=0,Slurm=0)
    else:
        need(Path(__file__).resolve()==WRAPPER and re.fullmatch(r'[a-f0-9]{40}',args.publication),'publish exact new source first')
        _,publisher=material[2][3][-2]['modules']()
        for dependency in (DIAG,diag['CLAIM'],R31,r31['CLAIM']):
            need(publisher['git']('cat-file','blob',args.publication+':'+dependency.relative_to(ROOT).as_posix())==pin(dependency),
                 'published exact diagnostic ancestor differs')
        sender=diag['run_once']
        bound=FunctionType(sender.__code__,dict(sender.__globals__,WRAPPER=WRAPPER,CLAIM=CLAIM,STEM=STEM,NODE=NODE,REQUEST=REQUEST))
        result=bound(material,args.publication,host)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

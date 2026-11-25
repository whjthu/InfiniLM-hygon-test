"""Give f12r1n16 one distinct standby attempt after the .1375 diagnostic."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r29-n16-warning-diagnostic-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r29_n16_warning_diagnostic_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
SINGLE = ROOT / 'tools/run_training80_r25_last40_ordinary_guarded_refill_once.py'
SINGLE_SHA = '575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e'
DIAG = ROOT / 'tools/run_training80_r29_warning_zero_gres_once.py'
DIAG_SHA = 'f294f01800f1859b16a3da72702afd7f7034c14c141b53bd52816d897cd54829'
NODE, PARENT, PREVIOUS, FIRST_CPU, NEW_CPU = 'f12r1n16','1194900','1194900.1317','1194900.1366','1194900.1375'
OLD_REQUEST = '4c9703063158573d90245cc4d1d3a4ef'
DIAG_REQUEST = '1a836ae5987c47389862adf0b3703417'
REQUEST = '6e5de0cf03fd4e68853626e8f111fbd8'
DIAG_RECEIPTS = {'intent':'2bcc9315a44b016cea1703d968d947891546120667040c07100f7f22f1fcb27d',
                 'send':'c659534485c9d28f929945c151768e6c1ba8acf4b8dda10f3d5eeef1a6ebec8d',
                 'reply':'11d115d6ab7d4b95fa19a0f0c8ba13f0dfc712e1013726cd4578c7cf7c28f03f'}
DIAG_PUB = '50754bc3d3679299e8365c8c8749304ba46f4590'
DIAG_PROGRAM = '0f2bc3a889c601074e553976608c2b9a15b31c343611e34407613776e635b84b'
SOURCE_CHECK = r'''
def verify_warning_diagnostic():
    def bounded(path):
        require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'earlier diagnostic paths changed')
        with path.open('rb') as stream:
            prior_bytes=stream.read(8193)
        require(len(prior_bytes)<=8192,'prior diagnostic file exceeded bound')
        return prior_bytes

    old_path=RESULTS/OLD_AUDIT/'outcome.json'
    diag=RESULTS/DIAG_AUDIT
    require(json.loads(bounded(old_path))==OLD_BODY and json.loads(bounded(diag/'outcome.json'))==DIAG_BODY,
            'prior no-GPU-send or authenticated diagnostic outcome changed')
    require(not os.path.lexists(RESULTS/('training80-dcu-standby-'+OLD_REQUEST)), 'earlier GPU namespace exists')
    require(bounded(diag/'stderr.private.raw')==b'' and sha256(bounded(diag/'stdout.private.raw')).hexdigest()==DIAG_BODY['stdout_SHA256'],
            'warning-free diagnostics receipt bytes changed')
    require(account(FIRST_CPU)==DIAG_BODY['original_zero_GRES_accounting'] and account(DIAG_CPU)==DIAG_BODY['new_zero_GRES_accounting'],
            'earlier two exact CPU terminal rows changed')
'''


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sources():
    for path,sha in ((SINGLE,SINGLE_SHA),(DIAG,DIAG_SHA)):
        require(path.resolve(strict=True)==path and digest(path.read_bytes())==sha,'published guarded dependency changed')
    a=runpy.run_path(str(SINGLE)); original=a['bound_api']()
    diag=runpy.run_path(str(DIAG)); prepared=diag['scope']()
    return original,diag,prepared


def proven_diagnostic(original,diag,prepared):
    prefix=LOCAL/f"{diag['STEM']}-{DIAG_REQUEST}"
    objects={kind:json.loads(original['pin'](prefix.with_name(prefix.name+'-'+kind+'.json'),sha))
             for kind,sha in DIAG_RECEIPTS.items()}
    intent,sent,reply=(objects[k] for k in ('intent','send','reply'))
    transport=reply['transport']
    body=transport['response']
    [_,_,_,row,claim_sha]=prepared
    require(intent==dict(request=DIAG_REQUEST,publication=DIAG_PUB,claim_SHA256=claim_sha,program_SHA256=DIAG_PROGRAM,GPU=0,replay_allowed=False)
            and {name:reply[name] for name in intent}==intent and digest(diag['render'](prepared,DIAG_PUB).encode())==DIAG_PROGRAM
            and transport['authority_verified'] is transport['remote_received'] is transport['payload_send_attempted'] is True
            and sent['request']==transport['request']==body['request']==DIAG_REQUEST and transport['payload_sha256']==sent['payload_sha256']
            and transport['rc']==0 and transport['remote_outcome']=='RESPONSE_OBSERVED' and body['state']=='DIAGNOSTIC_C0_AND_PHYSICAL_OBSERVED'
            and body['node']==NODE and body['parent']==PARENT and body['GPU_srun_attempted'] is body['GPU_admission_eligible'] is False
            and body['stderr_projection']['exact_SHA256']==digest(b'') and body['stderr_projection']['exact_bytes']==0
            and body['stderr_projection']['matches_prior_rc0_stderr'] is False and body['new_zero_GRES_step']==NEW_CPU,
            'actual diagnostic identity or no-warning/no-GPU result differs')
    physical=body['physical_evidence']
    empty=dict(PCIs=row['pci'],KFD_holders=0,project_UID_device_FD_holders=0,occupied_ports=[])
    require(physical['samples']==[empty,empty] and physical['project_UID']==154459
            and physical['global_KFD_observed'] is physical['own_probe_cgroup'] is True
            and physical['other_UID_device_FDs']=='UNOBSERVED'
            and body['original_zero_GRES_accounting'][:6]==[FIRST_CPU,'training80-r25-fresh-'+OLD_REQUEST[:12],
                                                            'COMPLETED','0:0',NODE,'cpu=1,mem=256M,node=1']
            and body['new_zero_GRES_accounting'][:6]==[NEW_CPU,'training80-r29-warning-'+DIAG_REQUEST[:12],
                                                       'COMPLETED','0:0',NODE,'cpu=1,mem=256M,node=1'],
            'actual diagnostics CPU/PCI/KFD/UID evidence differs')
    return body


def expected_claim(diag):
    return dict(scope=STEM,state='PREPARED_ONLY_NEW_PHYSICAL_AND_TERMINAL_ADMISSION_REQUIRED',request=REQUEST,
                node=NODE,parent=PARENT,previous=PREVIOUS,rejected_GPU_request=OLD_REQUEST,original_zero_GRES=FIRST_CPU,
                diagnosis_zero_GRES=NEW_CPU,diagnosis_request=DIAG_REQUEST,diagnosis_reply_SHA256=DIAG_RECEIPTS['reply'],
                original_footer_SHA256=diag['FOOTER_SHA'],original_stderr_SHA256=diag['OLD_STDERR'],warning_policy='ALL_NONEMPTY_STDERR_BLOCKS_NEW_GPU',
                max_invocations=1,ledger_mutations=0,GPU=8,Slurm_cap_seconds=11100,replay_allowed=False,
                launcher=WRAPPER.relative_to(ROOT).as_posix())


def bound_api():
    original,diag,prepared=sources()
    verified=proven_diagnostic(original,diag,prepared)
    former=prepared[0][NODE][1]
    env=dict(original,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC={NODE:REQUEST})
    for name,function in original.items():
        if isinstance(function,FunctionType) and function.__globals__ is original:
            target=FunctionType(function.__code__,env,name,function.__defaults__,function.__closure__)
            target.__kwdefaults__=function.__kwdefaults__
            env[name]=target
    proven_render=env['render']

    def load(draft=False):
        base,initial,ordinary,plan,read,rows,_=original['load']()
        row=rows[NODE]
        raw=original['pin'](DRAFT if draft and not CLAIM.exists() else CLAIM)
        require((row['node'],row['parent'],row['previous'],row['new_request'])==(NODE,PARENT,PREVIOUS,OLD_REQUEST)
                and json.loads(raw)==expected_claim(diag),'fixed single-node diagnostic successor differs')
        return base,initial,ordinary,plan,read,{NODE:dict(row,new_request=REQUEST)},digest(raw)

    def render(context,row,account,terminal_pin,publication):
        program=proven_render(context,row,account,terminal_pin,publication)
        launch=program['launch']
        prefix=dict(OLD_AUDIT=f'training80-r25-last40-ordinary-guarded-bf16-v1-{NODE}-{OLD_REQUEST}',
                    DIAG_AUDIT=f'{diag["STEM"]}-{DIAG_REQUEST}', OLD_REQUEST=OLD_REQUEST,FIRST_CPU=FIRST_CPU,DIAG_CPU=NEW_CPU,
                    DIAG_BODY=verified,OLD_BODY=former)
        launch=''.join(f'{name}={value!r}\n' for name,value in prefix.items())+launch
        launch=original['unique'](launch,'\ndef run_guarded():',SOURCE_CHECK+'\ndef run_guarded():')
        launch=original['unique'](launch,'([old_label], [gpu_label])',f"([old_label], [gpu_label], ['training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}'])")
        launch=original['unique'](launch,'            check_previous(account(PREVIOUS))',
                                   '            check_previous(account(PREVIOUS))\n            verify_warning_diagnostic()')
        require(launch.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<launch.index('            verify_warning_diagnostic()')
                <launch.index('outcome["zero_GRES_call_attempted"] = True')
                <launch.index('require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,'),
                'fresh warning-free route was weakened')
        compile(launch,'<n16 strict post-diagnostic standby continuation>','exec')
        program['launch']=launch
        return program

    def publication(context,revision):
        require(Path(__file__).resolve()==WRAPPER,'exact installed successor required')
        original['publication'](context,revision)
        _,publisher=context[3][-2]['modules']()
        for path in (WRAPPER,CLAIM,DIAG,diag['CLAIM']):
            ref=revision+':'+path.relative_to(ROOT).as_posix()
            require(publisher['git']('cat-file','blob',ref)==original['pin'](path),'published n16 diagnostic successor differs')

    env.update(load=load,render=render,publication=publication)
    return env


if __name__=='__main__':
    bound_api()['main']()

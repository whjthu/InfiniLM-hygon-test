"""One strict f12r3n18 standby continuation after the .1386 diagnosis."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from types import FunctionType, SimpleNamespace


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r33-n18-postdiagnostic-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r33_n18_postdiagnostic_guarded_refill_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
N16=ROOT/'tools/run_training80_r29_n16_warning_diagnostic_guarded_refill_once.py'
N16_SHA='88d8ece33936d23e48a3c5ff1bd16b0ef504dd236f5ae5060eeacf5731befab5'
DIAG=ROOT/'tools/run_training80_r33_n18_three_stream_zero_gres_once.py'
DIAG_SHA='e45b6cfd513489bfb519e3cad42da804f2bdf23b9d1684fd1520cad8b831f7cf'
LATE=LOCAL/'training-r33-exact-gemma-journal-and-n18-stream-late-5f661c5a8ba74fb891cb4b97a9a06aa1.json'
LATE_SHA='a60fc8f14b9a6e5606171f626caa3c36218c7cc1bc7e0cf84695f07632098894'
NODE,PARENT,PREVIOUS,FIRST_CPU,NEW_CPU='f12r3n18','1194900','1194900.1277','1194900.1382','1194900.1386'
OLD_REQUEST='4d4e0f3f8ced5f41af7542c22e023e8a'
DIAG_REQUEST='d1783c8023ad4708ac6ae56b97a79a9d'
REQUEST='ec1fdd96c5bc4a0e943a1e0eba1f2c99'
DIAG_PUB='b015e20096aad7001beedd021b1970134e48a71c'
DIAG_PROGRAM='317dc79b148106d99c62bf417b0495cab7d83efc9db5c06b04f98c216017c4a5'
DIAG_RECEIPTS={'intent':'6c4bbe70ae9fdb72f1d8933b08c6aa4dcd1422a72b8d3b6839836610a2b79894',
               'send':'07f18e00771eb868b4b240220f4d3a0f63e65ab72d4291f15df6720743be26f6',
               'reply':'c6e00616048fe1fb3db2431ae2d73ab158f45f0b69866c8f5d6f24830a79f6c3'}
FILE_LABELS=('login-srun.stderr.private.raw','slurm-task.stderr.private.raw','child-probe.stderr.private.raw')


def digest(raw):
    return sha256(raw).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def pinned_source(path,sha):
    require(path.resolve(strict=True)==path and digest(path.read_bytes())==sha,'published diagnostic successor source differs')
    return runpy.run_path(str(path))


def sources():
    diag=pinned_source(DIAG,DIAG_SHA)
    same_old,_,api,ctx,row,rejected,account=diag['evidence']()
    require((row['node'],row['parent'],row['previous'],row['new_request'],account[0],account[2])
            ==(NODE,PARENT,PREVIOUS,OLD_REQUEST,PREVIOUS,'COMPLETED') and rejected['GPU_srun_attempted'] is False,
            'original n18 natural return or rejected GPU request differs')
    old_loader=api['load']

    def load():
        base,config,ordinary,plan,read,roster,claim=old_loader()
        current=roster[NODE]
        require((current['previous'],current['worker_SHA256'],current['expected_SHA256'])
                ==(PREVIOUS,row['worker_SHA256'],row['expected_SHA256']),'ordinary n18 worker identity differs')
        return base,config,ordinary,plan,read,dict(roster,**{NODE:dict(current,new_request=OLD_REQUEST)}),claim

    api['load']=load
    return api,diag,({NODE:(diag['OLD_REPLY_SHA'],rejected)},api,ctx,row,same_old)


def proven_diagnostic(api,diag,material):
    _,_,exact,host=diag['prepared'](SimpleNamespace(check=False,publication=DIAG_PUB))
    names={kind:LOCAL/f'{diag["STEM"]}-{DIAG_REQUEST}-{kind}.json' for kind in DIAG_RECEIPTS}
    intent,sent,reply=(json.loads(api['pin'](names[k],DIAG_RECEIPTS[k])) for k in ('intent','send','reply'))
    transport=reply['transport']; body=transport['response']
    require(intent==dict(request=DIAG_REQUEST,publication=DIAG_PUB,claim_SHA256=exact[-1],program_SHA256=DIAG_PROGRAM,GPU=0,replay_allowed=False)
            and {k:reply[k] for k in intent}==intent and digest(host.encode())==DIAG_PROGRAM
            and sent['request']==transport['request']==body['request']==DIAG_REQUEST and sent['payload_sha256']==transport['payload_sha256']
            and transport['rc']==0 and transport['remote_outcome']=='RESPONSE_OBSERVED'
            and transport['authority_verified'] is transport['remote_received'] is transport['payload_send_attempted'] is True
            and body['state']=='DIAGNOSTIC_C0_AND_PHYSICAL_OBSERVED' and body['parent']==PARENT and body['node']==NODE
            and body['GPU_srun_attempted'] is body['GPU_admission_eligible'] is False and body['probe_rc']==0,
            'authentic diagnostic result or compiled payload differs')
    check=body['physical_evidence']
    empty=dict(PCIs=material[3]['pci'],KFD_holders=0,project_UID_device_FD_holders=0,occupied_ports=[])
    require(check['samples']==[empty,empty] and check['project_UID']==154459
            and check['global_KFD_observed'] is check['own_probe_cgroup'] is True
            and check['other_UID_device_FDs']=='UNOBSERVED'
            and body['original_zero_GRES_accounting'][:6]==[FIRST_CPU,'training80-r25-fresh-'+OLD_REQUEST[:12],
                    'COMPLETED','0:0',NODE,'cpu=1,mem=256M,node=1']
            and body['new_zero_GRES_accounting'][:6]==[NEW_CPU,'training80-r33-stderr-'+DIAG_REQUEST[:12],
                    'COMPLETED','0:0',NODE,'cpu=1,mem=256M,node=1']
            and body['new_zero_GRES_step']==NEW_CPU and body['stderr_route']['channels']['child_probe']['kind']=='UNOBSERVED'
            and not any(channel['matches_prior_rc0_stderr'] for channel in body['stderr_route']['channels'].values()),
            'actual .1382/.1386 and two PCI samples differ')
    late=json.loads(api['pin'](LATE,LATE_SHA)); delivery=late['observed']
    physical_zero=dict(SHA256=digest(b''),bytes=0,present=True)
    require(late['rc']==0 and delivery['scheduler_mutations']==0 and delivery['raw_exported'] is False
            and delivery['n18_channel_late_read']==dict.fromkeys(('child_probe','login_srun','slurm_task'),physical_zero)
            and body['stderr_route']['matching_channels']==[] and body['stderr_route']['process_origin_proven'] is False,
            'late original three empty stderr files or warning inference differs')
    return body


def expected_claim(diag):
    return dict(scope=STEM,state='PREPARED_ONLY_NEW_TERMINAL_AND_PHYSICAL_ADMISSION_REQUIRED',request=REQUEST,launcher=WRAPPER.relative_to(ROOT).as_posix(),
                node=NODE,parent=PARENT,previous=PREVIOUS,rejected_R31_request=OLD_REQUEST,original_zero_GRES=FIRST_CPU,
                diagnostic_zero_GRES=NEW_CPU,diagnostic_request=DIAG_REQUEST,diagnostic_reply_SHA256=DIAG_RECEIPTS['reply'],
                late_channels_SHA256=LATE_SHA,previous_warning_reproduced=False,warning_source_proven=False,
                fresh_warning_policy='ALL_NONEMPTY_STDERR_BLOCKS_GPU',GPU=8,Slurm_cap_seconds=11100,max_invocations=1,ledger_mutations=0,replay_allowed=False)


def bound_api():
    n16=pinned_source(N16,N16_SHA)
    function=n16['bound_api']
    needle="bounded(diag/'stderr.private.raw')==b''"
    require(n16['SOURCE_CHECK'].count(needle)==1,'qualified diagnostic locked stream boundary differs')
    check=n16['SOURCE_CHECK'].replace(needle,f"all(bounded(diag/name)==b'' for name in {FILE_LABELS!r})",1)
    env=dict(function.__globals__,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,DIAG=DIAG,
             NODE=NODE,PARENT=PARENT,PREVIOUS=PREVIOUS,FIRST_CPU=FIRST_CPU,NEW_CPU=NEW_CPU,OLD_REQUEST=OLD_REQUEST,
             DIAG_REQUEST=DIAG_REQUEST,REQUEST=REQUEST,SOURCE_CHECK=check,sources=sources,proven_diagnostic=proven_diagnostic,expected_claim=expected_claim)
    api=FunctionType(function.__code__,env)()
    scoped_render=api['render']; scoped_publication=api['publication']

    def render(context,row,account,terminal_pin,publication):
        wrapped=scoped_render(context,row,account,terminal_pin,publication)
        wrong=f'training80-r25-last40-ordinary-guarded-bf16-v1-{NODE}-{OLD_REQUEST}'
        actual=f'training80-r29-other11-warning-guarded-bf16-v1-{NODE}-{OLD_REQUEST}'
        require(wrapped['launch'].count(wrong)==1,'R31 rejection scope rewrite differs')
        wrapped['launch']=wrapped['launch'].replace(wrong,actual,1)
        compile(wrapped['launch'],'<n18 three-stream successor with fresh empty stderr>', 'exec')
        return wrapped

    def publication(context,revision):
        scoped_publication(context,revision)
        _,publisher=context[3][-2]['modules']()
        ref=revision+':'+N16.relative_to(ROOT).as_posix()
        require(digest(publisher['git']('cat-file','blob',ref))==N16_SHA,'published n16 reuse dependency differs')

    api.update(render=render,publication=publication)
    return api


if __name__=='__main__':
    bound_api()['main']()

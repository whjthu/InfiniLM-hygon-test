"""One distinct strict standby attempt for each of eleven proven R29 no-sends."""

from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r29-other11-warning-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r29_other11_warning_guarded_refill_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
SINGLE=ROOT/'tools/run_training80_r25_last40_ordinary_guarded_refill_once.py'
SINGLE_SHA='575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e'
DIAG=ROOT/'tools/run_training80_r29_warning_zero_gres_once.py'
DIAG_SHA='f294f01800f1859b16a3da72702afd7f7034c14c141b53bd52816d897cd54829'
PROOFS={
    '1103606':('596bf177ea2b43b7b91e4c62b7ee7195','62637d954c4b45915222af9e8f23a9107c1410860c6ee7d99bbf3467f95be5d5'),
    '1194900':('66c0d16dad11418895e112c60c962662','e86ee6f4cb9ff293b0d58c24c293935fff205ae1da15105b6a78d05105e168ce'),
}
NAMESPACE=UUID('cbef7316-3b6d-4ace-9322-080ab2aceaec')
EXCLUDED_NODE='f12r1n16'
EXCLUDED_REQUEST='6e5de0cf03fd4e68853626e8f111fbd8'
GPU_TRES='cpu=8,gres/dcu=8,mem=16G,node=1'
CPU_TRES='cpu=1,mem=256M,node=1'
HOST_CHECK=r'''
def verify_prior_warning_no_send():
    original=RESULTS/OLD_AUDIT/'outcome.json'
    require(original.resolve(strict=True)==original and original.stat().st_uid==os.getuid(), 'original no-send receipt path or UID differs')
    with original.open('rb') as handle:
        sealed_no_send_bytes=handle.read(8193)
    require(0<len(sealed_no_send_bytes)<=8192 and sha256(sealed_no_send_bytes).hexdigest()==OLD_GUARD_SHA
            and json.loads(sealed_no_send_bytes)==OLD_BODY, 'durable original no-GPU-send receipt changed')
    require(not os.path.lexists(RESULTS/('training80-dcu-standby-'+OLD_REQUEST)), 'old warning GPU namespace exists')
    require(account(OLD_CPU[0])==OLD_CPU,'previous exact zero-GRES C0 and End changed')
'''


def digest(data):
    return sha256(data).hexdigest()


def require(value,why):
    if not value:
        raise ValueError(why)


def load_inputs():
    for path,sha in ((SINGLE,SINGLE_SHA),(DIAG,DIAG_SHA)):
        require(path.resolve(strict=True)==path and digest(path.read_bytes())==sha,'published source dependency changed')
    single=runpy.run_path(str(SINGLE)); api=single['bound_api']()
    diag=runpy.run_path(str(DIAG)); old=diag['classify_originals']()
    targets={}
    for parent,(nonce,pin) in PROOFS.items():
        path=LOCAL/f'training80-r29-other11-original-warning-probes-{parent}-{nonce}.json'
        reply=json.loads(api['pin'](path,pin))
        observed=reply['observed']
        require(reply['rc']==0 and reply['replay_allowed'] is False and observed['parent']==parent and observed['read_id']==nonce
                and observed['Slurm_mutations']==0 and observed['GPU_GO'] is observed['device_admission'] is False,
                'original exact read-only namespace or result differs')
        for target in observed['targets']:
            node=target['node']
            old_pin,auth=old[node]
            [cpu]=target['old_CPU_terminal']
            [gpu]=target['previous_GPU_rows']
            old_request=auth['request']
            require(node not in targets and node!=EXCLUDED_NODE and auth['parent']==parent and auth['node']==node
                    and target['old_request']==old_request and target['original_durable_match'] is True
                    and target['prior_no_GPU_srun_proven'] is target['original_CPU_C0'] is target['may_prepare_separate_zero_GRES'] is True
                    and target['old_GPU_namespace_exists'] is False and target['old_GPU_label_matches']==[]
                    and target['scontrol_target_steps']==target['squeue_target_steps']==[]
                    and re.fullmatch(r'[a-f0-9]{64}',target['original_durable_outcome_SHA256'])
                    and len(cpu)==7 and cpu[0].startswith(parent+'.')
                    and cpu[1:6]==['training80-r25-fresh-'+old_request[:12],'COMPLETED','0:0',node,CPU_TRES]
                    and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',cpu[6])
                    and len(gpu)==9 and gpu[0]==target['previous_GPU'] and gpu[2:4]==['COMPLETED','0:0'] and gpu[5:7]==[node,GPU_TRES]
                    and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',gpu[8]),
                    'original old zero-GRES C0, old no-send or dual view unproved')
            request=uuid5(NAMESPACE,f'{pin}/{parent}/{node}/{old_request}').hex
            targets[node]=dict(parent=parent,old_request=old_request,request=request,cpu=cpu,gpu=gpu,original_reply_SHA256=old_pin,
                               prior_durable_SHA256=target['original_durable_outcome_SHA256'],auth=auth)
    new={item['request'] for item in targets.values()}
    require(set(targets)==set(old)-{EXCLUDED_NODE} and len(new)==11 and EXCLUDED_REQUEST not in new
            and not new&{record[1]['request'] for record in old.values()}
            and [sum(row['parent']==p for row in targets.values()) for p in PROOFS]==[7,4],
            'exact eleven-node scope or unique requests changed')
    return api,diag,targets


def expected_claim(diag,targets):
    return dict(scope=STEM,state='PREPARED_ONLY_NEW_DEVICE_AND_TERMINAL_ADMISSION_REQUIRED',launcher=WRAPPER.relative_to(ROOT).as_posix(),
                original_footer_SHA256=diag['FOOTER_SHA'],original_warning_SHA256=diag['OLD_STDERR'],
                proof_SHA256={parent:pin for parent,(_,pin) in PROOFS.items()},excluded_node=EXCLUDED_NODE,excluded_request=EXCLUDED_REQUEST,
                GPU_per_node=8,Slurm_cap_seconds=11100,max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False,
                fresh_warning_policy='REJECT_ALL_NONEMPTY_STDERR',targets={node:dict(parent=v['parent'],prior_GPU=v['gpu'][0],
                    prior_warning_CPU=v['cpu'][0],old_request=v['old_request'],request=v['request'],original_reply_SHA256=v['original_reply_SHA256'],
                    prior_durable_SHA256=v['prior_durable_SHA256']) for node,v in targets.items()})


def bound_api():
    original,diag,targets=load_inputs()
    env=dict(original,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC=targets)
    for name,fn in original.items():
        if isinstance(fn,FunctionType) and fn.__globals__ is original:
            bound=FunctionType(fn.__code__,env,name,fn.__defaults__,fn.__closure__)
            bound.__kwdefaults__=fn.__kwdefaults__
            env[name]=bound
    previous_renderer=env['render']

    def load(draft=False):
        base,conf,ordinary,plan,read,old,_=original['load']()
        raw=original['pin'](DRAFT if draft and not CLAIM.exists() else CLAIM)
        require(json.loads(raw)==expected_claim(diag,targets),'published eleven-target source binding differs')
        frozen={}
        for node,target in targets.items():
            row=old[node]
            require((row['parent'],row['node'],row['previous'],row['new_request'])
                    ==(target['parent'],node,target['gpu'][0],target['old_request'])
                    and target['gpu'][1]==f"training80-dcu-standby-{row['request'][:12]}-{node}",
                    'old ordinary bridge worker or failed request differs')
            frozen[node]=dict(row,new_request=target['request'])
        return base,conf,ordinary,plan,read,frozen,digest(raw)

    def render(context,row,account,terminal_pin,publication):
        node=row['node']; target=targets[node]
        old=target['old_request']
        result=previous_renderer(context,row,account,terminal_pin,publication)
        bind=dict(OLD_REQUEST=old,OLD_AUDIT=f'training80-r25-last40-ordinary-guarded-bf16-v1-{node}-{old}',
                  OLD_CPU=target['cpu'],OLD_GUARD_SHA=target['prior_durable_SHA256'],OLD_BODY=target['auth'])
        launch=''.join(f'{name}={value!r}\n' for name,value in bind.items())+result['launch']
        launch=original['unique'](launch,'\ndef run_guarded():',HOST_CHECK+'\ndef run_guarded():')
        launch=original['unique'](launch,'([old_label], [gpu_label])',f"([old_label], [gpu_label], ['training80-dcu-standby-{old[:12]}-{node}'])")
        launch=original['unique'](launch,'            check_previous(account(PREVIOUS))',
                                   '            check_previous(account(PREVIOUS))\n            verify_prior_warning_no_send()')
        require(launch.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<launch.index('            verify_prior_warning_no_send()')
                <launch.index('outcome["zero_GRES_call_attempted"] = True')
                <launch.index('require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,'),
                'fresh empty-stderr original device guard differs')
        compile(launch,'<strict R29 prior-no-send locked successor>','exec')
        result['launch']=launch
        return result

    def publication(context,revision):
        require(Path(__file__).resolve()==WRAPPER,'exact installed eleven-node source required')
        original['publication'](context,revision)
        _,publisher=context[3][-2]['modules']()
        for path in (WRAPPER,CLAIM,DIAG,diag['CLAIM']):
            ref=revision+':'+path.relative_to(ROOT).as_posix()
            require(publisher['git']('cat-file','blob',ref)==original['pin'](path),'published eleven-node successor bytes differ')

    env.update(load=load,render=render,publication=publication)
    return env


if __name__=='__main__':
    bound_api()['main']()

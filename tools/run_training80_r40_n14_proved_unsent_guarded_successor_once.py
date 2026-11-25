"""Give n14 one new guarded attempt after the old payload demonstrably never left."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r40-n14-proved-unsent-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r40_n14_proved_unsent_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
ORIGIN=('tools/run_training80_r39_n14_n16_guarded_successor_once.py',
        'c1c8da0a328eac485933eceb7fd6784143402f2e6c008da568d887116fc195f6')
NODE='f12r3n14'
PREVIOUS='1194900.1374'
ORIGINAL_CPU='1194900.1373'
REJECTED='a9b94bc5a40150b5bd68eb37e1be159b'
READ_ID='a75f4cf1400d4719aa8546624e9847a7'
INTENT_SHA='8a135e71d1edd0a2bd4b02fbabc9b8a7b45f560495e735aa5880c64f8e61ce53'
REPLY_SHA='a53fcc88f34b172efd946f8693cb7369054964a0706677703d0792404543b254'
READ_SHA='8110c7936b3423720fd8c1103be667031ce3354e69fef016005643bcd397eda0'
REQUEST=uuid5(UUID('c99e4169-2e4e-480e-9fa5-22c9ae4d0b18'),f'{NODE}/{PREVIOUS}/{REJECTED}/{REPLY_SHA}').hex


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def load():
    source=ROOT/ORIGIN[0]
    require(source.resolve(strict=True)==source and digest(source.read_bytes())==ORIGIN[1],'exact origin launcher differs')
    origin=runpy.run_path(str(source))
    budget,old_load,old_api=origin['api']()
    first,r34,early,n02,context,rows,old_claim=old_load()
    pin=budget['read_pinned']
    prior=rows[NODE]
    require(prior['previous']==PREVIOUS and prior['new_request']==REJECTED and prior['clearance'][0]==ORIGINAL_CPU,
            'original n14 scope changed')
    old_prefix=LOCAL/f"{origin['STEM']}-{NODE}-{REJECTED}"
    path=lambda suffix:old_prefix.with_name(old_prefix.name+'-'+suffix+'.json')
    intent=json.loads(pin(path('intent'),INTENT_SHA))
    reply=json.loads(pin(path('reply'),REPLY_SHA))
    transport=reply['transport']
    proof_path=LOCAL/f"{origin['STEM']}-{NODE}-read-{READ_ID}.json"
    proof=json.loads(pin(proof_path,READ_SHA))
    require(reply==dict(intent,transport=transport) and intent['request']==transport['request']==REJECTED
            and intent['node']==NODE and intent['terminal_read_id']==READ_ID and intent['terminal_read_SHA256']==READ_SHA
            and intent['replay_allowed'] is False and intent['GPU_requested']==8
            and transport['remote_outcome']=='PAYLOAD_NOT_SENT' and transport['response'] is None
            and all(transport[key] is False for key in ('authority_verified','payload_send_attempted','payload_flushed','remote_received'))
            and not os.path.lexists(path('send')),'prior scope cannot positively prove that the original payload never left')
    installed_claim=origin['CLAIM']
    pin(installed_claim,intent['claim_SHA256'])
    old_context,old_reader,_,old_render,old_dispatch,_=old_api((first,r34,early,n02,context,rows,old_claim),intent['claim_SHA256'])
    old_terminal=old_dispatch.__globals__['terminal'].__globals__['terminal_identity']
    account,pins=old_terminal(prior,READ_ID,proof)
    require(proof['source_SHA256']==digest(old_reader(old_context,prior,READ_ID).encode())
            and digest(old_render(old_context,prior,account,pins,intent['publication'])['launch'].encode())==intent['program_SHA256'],
            'authenticated original proof or original frozen GPU source differs')
    term,group=pins
    frozen=budget['controller'](context[1][6]['controller.py'])
    seconds=frozen['node_seconds'](NODE,10800,REQUEST)
    row=dict(prior,new_request=REQUEST,renewal_seconds=seconds,pinned_GPU=account,pinned_terminal=term,cgroup_SHA256=group,
             old_scope=f"{origin['STEM']}-{NODE}-{REJECTED}")
    claim=dict(scope=STEM,state='PREPARED_ONLY_DISTINCT_FRESH_READ_AND_DEVICE_ADMISSION',
               launcher=WRAPPER.relative_to(ROOT).as_posix(),node=NODE,parent=row['parent'],prior_GPU=PREVIOUS,prior_CPU=ORIGINAL_CPU,
               rejected_request=REJECTED,request=REQUEST,rejected_payload_never_sent=True,
               rejected_intent_SHA256=INTENT_SHA,rejected_reply_SHA256=REPLY_SHA,previous_terminal_read_SHA256=READ_SHA,
               previous_worker_SHA256=row['worker_SHA256'],previous_cgroup_SHA256=group,previous_terminal_SHA256=term,
               upstream_launcher_SHA256=ORIGIN[1],controller_SHA256=budget['CONTROLLER'][1],worker_seconds=seconds,
               Slurm_time_cap_seconds=seconds+300,GPU=8,max_invocations=1,ledger_mutations=0,replay_allowed=False)
    return origin,budget,first,r34,early,n02,context,{NODE:row},claim


def runtime(material,sha):
    origin,budget,first,r34,early,n02,context,rows,claim=material
    fn=first['_fn']
    tools=dict(early)
    independent=early['read_program']
    rep=early['unique']
    def read_program(ctx,row,nonce):
        actual=independent(ctx,row,nonce)
        return rep(actual,'    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,',
                   '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'
                   '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,')
    tools['read_program']=read_program
    old_render=early['render']
    def render(ctx,row,account,old_pin,publication):
        body=fn(old_render,STEM=STEM)(ctx,row,account,old_pin,publication)
        guard=body['launch']
        old_label=f'training80-dcu-standby-{REJECTED[:12]}-{NODE}'
        guard=rep(guard,'([old_label], [gpu_label])',f'([old_label], [gpu_label], [{old_label!r}])')
        anchor='            check_previous(account(PREVIOUS))'
        gate=(f"            require(not os.path.lexists(RESULTS / {('training80-dcu-standby-'+REJECTED)!r})\n"
              f"                    and not os.path.lexists(RESULTS / {row['old_scope']!r}), 'prior locally unsent namespace exists')\n")
        guard=rep(guard,anchor,gate+anchor)
        require(guard.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<guard.index('prior locally unsent namespace exists')
                <guard.index('outcome["zero_GRES_call_attempted"] = True'),'original namespace check moved outside shared lock')
        body['launch']=guard
        return body
    tools['render']=render
    direct=fn(budget['api'],STEM=STEM,G34=origin['LAST40'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,reader,read,render_new,dispatch,observe=direct((first,r34,tools,n02,context,rows,claim),sha)
    earlier=dispatch.__globals__['publication']
    def publication(ctx,revision):
        earlier(ctx,revision)
        _,publisher=ctx[3][-2]['modules']()
        for dep in (ORIGIN,origin['NEXT']):
            path,sha=dep
            require(publisher['git']('cat-file','blob',revision+':'+path)==budget['read_pinned'](ROOT/path,sha),
                    'published actual original launcher or budget gate differs')
    return ctx,reader,read,render_new,fn(dispatch,publication=publication),observe


def namespace_fixture(source,row):
    tree=ast.parse(source)
    [match]=[line for line in ast.walk(tree) if isinstance(line,ast.Expr) and 'prior locally unsent namespace exists' in ast.unparse(line)]
    check=compile(ast.Module(body=[match],type_ignores=[]),'<actual locked no-send n14 namespace gate>','exec')
    for kind in ('none','attempt','gpu'):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            if kind=='attempt':(root/row['old_scope']).mkdir()
            if kind=='gpu':(root/('training80-dcu-standby-'+REJECTED)).mkdir()
            try:
                exec(check,dict(RESULTS=root,os=os,require=require))
            except ValueError as error:
                require(kind!='none' and str(error)=='prior locally unsent namespace exists','wrong namespace verdict')
            else:
                require(kind=='none','prior namespace gate accepted a collision')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require(bool(args.publication)==bool(args.terminal_receipt),'exact fresh predecessor receipt required')
    material=load()
    budget,first,r34=material[1:4]
    row=material[-2][NODE]
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==material[-1],'exact n14 no-send successor differs')
    sha=digest(raw)
    ctx,reader,read,render,dispatch,observe=runtime(material,sha)
    if args.check:
        check=reader(ctx,row,'7'*32)
        cgroup=json.dumps(dict(node=NODE,step=PREVIOUS,worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                               same_UID=True,exact_job_and_step_cgroup=True)).encode()
        program=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)
        first['cgroup_fixture'](program['launch'],row,cgroup,require)
        namespace_fixture(program['launch'],row)
        seconds=budget['spawn_fixture'](program['launch'],row,ctx[1][6]['controller.py'])
        result=dict(state='PREPARED_ONLY',worker_seconds=seconds,request=REQUEST,actual_spawn_positive=1,actual_spawn_vetoes=2,
                    locked_cgroup_and_CPU_positive=1,locked_cgroup_and_CPU_vetoes=2,locked_namespace_positive=1,locked_namespace_vetoes=2,
                    reader_SHA256=digest(check.encode()),host_SHA256=digest(program['launch'].encode()),Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,row,args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,row)
    else:
        result=dispatch(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

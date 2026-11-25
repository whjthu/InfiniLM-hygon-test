"""Renew one ordinary worker after both the original return and an unsent attempt."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r79-n19-proved-unsent-guarded-bf16-v1'
WRAPPER = Path(__file__).resolve()
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
NODE = 'f13r2n19'
OLD_REQUEST = '1d7c428a2f995c088c4d8e49c19ee260'
OLD_SCOPE = 'training80-r78-r62-r64-r65-five-guarded-bf16-v1-f13r2n19-' + OLD_REQUEST
OLD_BASE = 'local-only/' + OLD_SCOPE
INTENT = (OLD_BASE+'-intent.json','11b1185f6049cd839137db278aeca62be386d64efc27710a8fb2811c5fefb305')
REPLY = (OLD_BASE+'-reply.json','930538ceffde0cd3c4136c4374607e5fa0e3cab9be1da5e30cd1bef966983d08')
RECONCILIATION = ('local-only/training80-r78-n19-original-no-send-independent-authoritative-absence-e1f91ee7e28647a1a750b39dd8e805d5/receipt.json',
                  '7c34e06b13ce07f4987642d747503fa797d10e7bb32c6b91cd7f9accd14375a2')
R78 = ('tools/run_training80_r78_r62_r64_r65_five_guarded_successor_once.py',
       'd7d97677933b177d0d8bd510d3a62c75212ce41c5a213abb9ec21267efdc2327')
R45 = ('tools/run_training80_r45_six_2117_guarded_successor_once.py',
       '067934fedb0e6b89a59ac8f5f3c65ac763ed55784680e880bcf6b1dc74644786')
PROTECTED = {'f13r2n07','f13r2n08','f13r3n13'}
TOKEN = ROOT / (OLD_BASE + '-send.json')


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def pin(spec):
    path = ROOT / spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path, 'noncanonical frozen evidence path')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1], 'exact frozen evidence differs')
    return raw


def no_old_token(exists=os.path.lexists):
    require(not exists(TOKEN), 'old R78 send token is present; original reconciliation required')


def load():
    pin(R78)
    base=runpy.run_path(str(ROOT/R78[0]))
    old_load,old_runtime,_=base['entry']()
    previous,unused,material=old_load()
    budget,first,r34,early,n02,context,rows,shadow,old_claim=material
    old=rows[NODE]
    require(old['new_request']==OLD_REQUEST and old['previous']=='1194900.1566'
            and old['clearance'][0]=='1194900.1565' and NODE not in PROTECTED
            and not shadow, 'exact R78 worker route differs')
    old_owner=old_claim['targets'][NODE]
    require(old_owner['queue_ownership']=='ADOPTED_ORDINARY58_EXACT_ANCESTOR'
            and NODE not in context[0]['local_holds'](), 'current work owns this ordinary node')
    no_old_token()
    intent=json.loads(pin(INTENT))
    reply=json.loads(pin(REPLY))
    tx=reply['transport']
    require(all(reply[key]==value for key,value in intent.items()) and intent['node']==NODE
            and intent['request']==tx['request']==OLD_REQUEST and intent['replay_allowed'] is False
            and tx['remote_outcome']=='PAYLOAD_NOT_SENT' and tx['rc']==1 and tx['response'] is None
            and all(tx[key] is False for key in ('authority_verified','payload_send_attempted','payload_flushed','remote_received')),
            'no positive exact client no-send evidence')
    prior_ctx,_,_,prior_render,_,_=old_runtime(previous,unused,material,intent['claim_SHA256'])
    exact=prior_render(prior_ctx,old,old['pinned_GPU'],(old['pinned_terminal'],old['cgroup_SHA256']),intent['publication'])
    require(sha256(exact['launch'].encode()).hexdigest()==intent['program_SHA256']
            and f'SCOPE={OLD_SCOPE!r}' in exact['launch'], 'old rendered scope differs')

    checked=json.loads(pin(RECONCILIATION))
    observed=checked['response']
    namespace=observed['namespace']
    require(checked['rc']==0 and checked['authority_verified'] is checked['remote_received'] is True
            and checked['remote_outcome']=='RESPONSE_OBSERVED' and checked['request']==observed['request']
            and observed['node']==NODE and observed['original_no_send_scope']==OLD_REQUEST
            and observed['actual_new_GPU']==0
            and observed['original_exact_rows']==observed['controller_node_rows']==observed['queue_node_steps']==[]
            and namespace['original_intent_root']==namespace['worker_root']=={'exists':False}
            and namespace['intent_members']==dict.fromkeys(('intent.json','fresh-probe-intent.json','fresh-probe.json',
                    'gpu-send-intent.json','outcome.json'),{'exists':False})
            and namespace['worker_members']==dict.fromkeys(('submission-receipt.json',NODE+'/expected.json',
                    NODE+'/worker.json',NODE+'/terminal.json'),{'exists':False})
            and observed['accounting']==[old['clearance'],old['pinned_GPU'][:4]+old['pinned_GPU'][5:7]+old['pinned_GPU'][8:]],
            'independent controller, accounting or original namespace proof differs')
    controller=budget['controller'](context[1][6]['controller.py'])
    request=uuid5(UUID('4bc88952-16ab-4e05-8172-ce747cada437'), f'{NODE}/{old["previous"]}/{OLD_REQUEST}').hex
    seconds=controller['node_seconds'](NODE,10800,request)
    require(request not in {OLD_REQUEST,old['request']} and 10500<=seconds<=10800, 'new request or runtime differs')
    row=dict(old,new_request=request,renewal_seconds=seconds)
    prior=dict(request=OLD_REQUEST,scope=OLD_SCOPE,intent_SHA256=INTENT[1],reply_SHA256=REPLY[1],
               reconciliation_SHA256=RECONCILIATION[1],old_send_token_present=False,replay_allowed=False)
    claim=dict(old_claim,scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(), parent_serial_order=[NODE],
               original_exact_node_receipt_SHA256={NODE:old_claim['original_exact_node_receipt_SHA256'][NODE]},
               source_pins=dict(old_claim['source_pins'], **dict((R78,R45))),
               targets={NODE:dict(old_owner,new_request=request,new_worker_seconds=seconds,
                                  Slurm_time_cap_seconds=seconds+300,closed_prior_transport=prior)},
               priority_scopes_preserved=sorted(set(old_claim['priority_scopes_preserved'])|PROTECTED|(set(rows)-{NODE})),
               prior_authoritative_absence_receipt_SHA256=RECONCILIATION[1])
    context=(*context[:5],{NODE:context[5][NODE]},context[6])
    return old_runtime,previous,unused,(budget,first,r34,early,n02,context,{NODE:row},{},claim),prior


def runtime(old_runtime,previous,unused,material,prior,claim_sha):
    first=material[1]
    clone=first['_fn']
    pin(R45)
    established=runpy.run_path(str(ROOT/R45[0]))
    inherited=clone(old_runtime,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,source,read,render,dispatch,observe=inherited(previous,unused,material,claim_sha)
    original_publication=dispatch.__globals__['publication']
    original_fence=dispatch.__globals__['exact_no_other_claims']

    def publication(actual, revision):
        original_publication(actual,revision)
        _,publisher=actual[3][-2]['modules']()
        for spec in (R78,R45):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                    'published exact original or closed-transport guard differs')

    def guarded_render(actual, row, accounting, terminal_pins, revision):
        program=render(actual,row,accounting,terminal_pins,revision)
        program['launch']=established['guarded_shadow'](
            program['launch'],prior,NODE,material[3]['unique'])
        return program

    def fence(ordinary, authority, row):
        original_fence(ordinary,authority,row)
        no_old_token()

    return (ctx,source,read,guarded_render,
            clone(dispatch,publication=publication,exact_no_other_claims=fence,render=guarded_render),
            clone(observe,render=guarded_render),established['shadow_fixture'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--show-run-order',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=(NODE,))
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require((args.node or args.check or args.show_run_order) and bool(args.publication)==bool(args.terminal_receipt),
            'one exact node and fresh predecessor receipt required')
    old_runtime,previous,unused,material,prior=load()
    budget,first,_,_,n02,_,rows,claim_shadow,claim=material
    raw=CLAIM.read_bytes()
    require(json.loads(raw)==claim and not claim_shadow, 'published one-node scope differs')
    claim_sha=sha256(raw).hexdigest()
    row=rows[NODE]
    if args.show_run_order:
        result=dict(state='PREPARED_ONLY',ordered=[{key:row[key] for key in
                    ('node','parent','previous','new_request','renewal_seconds')}],max_concurrent_send=1,replay_allowed=False)
    else:
        ctx,source,read,render,dispatch,observe,shadow_fixture=runtime(old_runtime,previous,unused,material,prior,claim_sha)
        if args.check:
            fixed=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)
            cgroup=json.dumps(dict(node=NODE,step=row['previous'],worker_pid=row['previous_worker_pid'],
                 worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
            cpin=sha256(cgroup).hexdigest()
            fixture=render(ctx,dict(row,cgroup_SHA256=cpin),row['pinned_GPU'],(row['pinned_terminal'],cpin),None)
            first['cgroup_fixture'](fixture['launch'],row,cgroup,require)
            seconds=budget['spawn_fixture'](fixture['launch'],row,ctx[1][6]['controller.py'])
            shadow_fixture(fixed['launch'],row,prior)
            no_old_token(lambda path:False)
            try:
                no_old_token(lambda path:True)
            except ValueError:
                pass
            else:
                raise ValueError('present old token accepted')
            n02['fixture_holds']()
            result=dict(state='PREPARED_ONLY',node=NODE,request=row['new_request'],seconds=seconds,
                        locked_namespace_and_label_vetoes=5,local_token_vetoes=1,
                        read_SHA256=sha256(source(ctx,row,'1'*32).encode()).hexdigest(),SSH=0,Git=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            result=read(ctx,row,args.observe_predecessor)
        elif args.observe_original:
            result=observe(ctx,row)
        else:
            result=dispatch(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=claim_sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

"""Use one frozen ordinary successor inventory and six exact guarded singleton requests."""

from datetime import datetime
import json
from pathlib import Path
import runpy
import sys
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
EARLY = ROOT / 'tools/run_training80_r25_early_ordinary_guarded_refill_once.py'
EARLY_SHA = 'f64000177ec337dad0b097007440a0a727ebf4e99712f68bd4603d527922b9fe'
NEXT = ROOT / 'tools/run_training80_r25_next_ordinary_guarded_refill_once.py'
NEXT_SHA = 'aa27c6421012b179b3587f08b02064649250880d06ecce28f2f1937e625d2af4'
STEM = 'training80-r25-remaining-ordinary-first6-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r25_remaining_ordinary_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
MANIFEST = LOCAL / 'training80-r25-remaining49-ordinary-incumbent-inventory-v1.json'
CHECKPOINT = ROOT / 'results/bw1000/2026-11-25/training80-1140-ordinary58-independent-counter-checkpoint-445.json'
CHECKPOINT_SHA = 'c6c9d46cd57f243f0e6bd84c98f01f2e4ef57a41e3f4f526dfa88786d87c6d92'
REQUESTS = {
    'f13r3n09': '4281f3bd808144ee9896da3ed2161f56',
    'f13r4n14': '40470f49a6a04158bdd6d137a18be768',
    'f12r2n10': 'e26b0d27c7774fcbaabafa663bdc6b43',
    'f13r1n07': 'ce0a9fea84f14b1b9666750b95a66c21',
    'f13r2n05': 'a5198b5155364212a6cdb404742ad933',
    'f13r4n05': '26b8abf6af6a4da2b3916d091188e94f',
}


def sources():
    from hashlib import sha256

    for path,sha in ((EARLY,EARLY_SHA),(NEXT,NEXT_SHA)):
        if path.resolve(strict=True)!=path or sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('published singleton base changed')
    early,next_api=runpy.run_path(str(EARLY)),runpy.run_path(str(NEXT))
    early['pin'](next_api['EARLY_CLAIM'],next_api['EARLY_CLAIM_SHA'])
    return early,next_api


def build_manifest(early,next_api,old_plan):
    require,pin=early['require'],early['pin']
    previous=json.loads(pin(next_api['CALENDAR'],next_api['CALENDAR_SHA']))
    checkpoint=json.loads(pin(CHECKPOINT,CHECKPOINT_SHA))
    owners=set(early['SPEC'])|set(next_api['SPEC'])
    ack=previous['ordinary_not_yet_independently_observed']
    n02=previous['extra_n02']['node']
    require(len(owners)==7 and ack==['f12r3n20','f13r4n16'] and previous['ordinary_count']==56
            and checkpoint['independently_numbered_observed_nodes']==56 and checkpoint['verified_full_eight_nodes']==53,
            '56-node observation scope differs')
    proof={}
    for reference in checkpoint['proof']:
        name=reference['file']
        require(Path(name).name==name and reference['SHA256'] not in proof,'bad immutable observation index')
        raw=pin(LOCAL/name,reference['SHA256'])
        wrapper=json.loads(raw)
        require(len(raw)==reference['bytes'] and wrapper['rc']==0 and wrapper['response']['GPU_steps_created']==0,
                'original proof is not a successful read-only observation')
        proof[reference['SHA256']]=(name,wrapper['response']['nodes'])
    result=[]
    for item in previous['ordinary']:
        node=item['node']
        if node in owners:
            continue
        checkpoint_node=checkpoint['nodes'][node]
        matches=[(sha,proof[sha][0],proof[sha][1][node]) for sha in checkpoint_node['proof_SHA256']]
        sha,name,record=max(matches,key=lambda r:r[2]['growth_DCU'])
        group=item['group']
        [ancestor]=[old for old in old_plan[-1][group-1] if old['node']==node]
        start,seconds=record['worker_start_epoch'],record['worker_seconds']
        prep=datetime.fromisoformat(item['check_from_Beijing']).timestamp()
        end=datetime.fromisoformat(item['predicted_end_Beijing']).timestamp()
        pci=sorted(set(record['first']) | set(record['second']))
        cards=record['growth_DCU']
        require(record['previous']==checkpoint_node['step']==item['running_original']
                and record['request']==checkpoint_node['request']==item['request'] and record['parent']==ancestor['parent']
                and cards==item['verified_positive_DCU']==checkpoint_node['positive_DCU'] and cards in (7,8)
                and len(pci)==8 and (cards==7 or (set(record['first'])==set(record['second']) and record['pair_state']=='PASS'))
                and abs(start+seconds-end)<.000002 and abs(prep+900-end)<.000002,
                'original worker counter, exact ancestry or calendar differs')
        result.append(dict(node=node,parent=record['parent'],previous=record['previous'],request=record['request'],
                           worker_SHA256=record['worker_SHA256'],expected_SHA256=record['expected_SHA256'],submit_SHA256=record['submit_SHA256'],
                           cgroup_SHA256=None,pci=pci,worker_start_epoch=start,worker_seconds=seconds,previous_worker_pid=record['worker_PID'],
                           planned_end_epoch=end,prepare_epoch=prep,prep_Beijing=item['check_from_Beijing'],budget_Beijing=item['predicted_end_Beijing'],
                           ledger_old=ancestor['previous'],ledger_request=old_plan[4]['request_ids'][group-1],published_group=group,
                           original_counter_cards=cards,source_file=name,source_SHA256=sha,actual_terminal=False))
    require(len(result)==49 and len({row['node'] for row in result})==49
            and n02=='f13r1n02'
            and not {row['node'] for row in result} & (owners|set(ack)|{n02})
            and [row['node'] for row in result[:6]]==list(REQUESTS)
            and sum(row['original_counter_cards']==8 for row in result)==46,
            'remaining ordinary scope or urgency order differs')
    return dict(schema='training80.r25.remaining49-ordinary-incumbents.v1', checkpoint_SHA256=CHECKPOINT_SHA,
                calendar_SHA256=next_api['CALENDAR_SHA'], original_observed=56, already_claimed_subset=sorted(owners),
                ack_only_hold=ack, protected_n02=n02, inputs=sorted(proof), prior_full_counter_nodes=46,
                prior_partial_counter_nodes=3, scopes_bound_to_exact_new_requests=list(REQUESTS), actual_terminals=0,
                ledger_mutations=0, rows=result)


def claim_fields(early,manifest,manifest_sha):
    pick=manifest['rows'][:6]
    early['require']([row['node'] for row in pick]==list(REQUESTS) and all(row['original_counter_cards']==8 for row in pick),
                     'only the first six fully observed incumbents can receive a new binding')
    targets={}
    for row in pick:
        node=row['node']
        targets[node]=dict(parent=row['parent'],previous_step=row['previous'],prior_standby_request=row['request'],
                           original_group_request=row['ledger_request'],original_group=row['published_group'],
                           prior_device_mode='ORDINARY_BRIDGE_BF16',prior_submit_SHA256=row['submit_SHA256'],request=REQUESTS[node])
    return dict(scope=STEM,state='PREPARED_ONLY_ACTUAL_TERMINAL_REQUIRED',scope_mode='ONE_NODE_EXACT_ORDINARY_BRIDGE_STANDBY',
                launcher=WRAPPER.relative_to(ROOT).as_posix(),original_adoption_SHA256=early['ADOPTION'],base_source_SHA256=EARLY_SHA,
                preceding_scope_source_SHA256=NEXT_SHA,remaining49_local_manifest_SHA256=manifest_sha,max_invocations_per_node=1,
                GPU_per_target=8,ledger_mutations=0,replay_allowed=False,targets=targets)


def assemble():
    early,next_api=sources()
    plan=early['toolbox'](early['ORDINARY'])['plan']()
    manifest=build_manifest(early,next_api,plan)
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    claim=(json.dumps(claim_fields(early,manifest,early['digest'](raw)),sort_keys=True,indent=2)+'\n').encode()
    with MANIFEST.open('xb') as handle:
        handle.write(raw)
    with DRAFT.open('xb') as handle:
        handle.write(claim)
    print(json.dumps(dict(nodes=49,live_requests=0,scopes=6,manifest_SHA256=early['digest'](raw),claim_SHA256=early['digest'](claim)),sort_keys=True))


def bound_api():
    early,next_api=sources()
    env=dict(early,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC=REQUESTS)
    for name,fn in early.items():
        if isinstance(fn,FunctionType) and fn.__globals__.get('__file__')==str(EARLY):
            clone=FunctionType(fn.__code__,env,name,fn.__defaults__,fn.__closure__)
            clone.__kwdefaults__=fn.__kwdefaults__
            env[name]=clone
    original_publication=env['publication']

    def load(draft=False):
        base=early['toolbox'](early['BASE'])
        config=base['prepared'](json.loads(early['pin'](ROOT/early['BASE_CLAIM'][0],early['BASE_CLAIM'][1])))
        ordinary=early['toolbox'](early['ORDINARY'])
        old_plan=ordinary['plan']()
        read=early['toolbox'](early['R25'])
        raw=early['pin'](DRAFT if draft and not CLAIM.exists() else CLAIM)
        claim=json.loads(raw)
        pin=claim['remaining49_local_manifest_SHA256']
        manifest=json.loads(early['pin'](MANIFEST,pin))
        early['require'](claim==claim_fields(early,manifest,pin) and len(manifest['rows'])==49
                         and manifest['checkpoint_SHA256']==CHECKPOINT_SHA and manifest['actual_terminals']==0,
                         'fixed archive or six-target claim differs')
        if draft:
            early['require'](manifest==build_manifest(early,next_api,old_plan),'49-node archive does not independently recalculate')
        targets={row['node']:dict(row,new_request=REQUESTS[row['node']]) for row in manifest['rows'][:6]}
        return base,config,ordinary,old_plan,read,targets,early['digest'](raw)

    def publication(context,revision):
        original_publication(context,revision)
        _,publisher=context[3][-2]['modules']()
        for known,sha in ((EARLY,EARLY_SHA),(NEXT,NEXT_SHA)):
            ref=revision+':'+known.relative_to(ROOT).as_posix()
            early['require'](publisher['git']('cat-file','blob',ref)==early['pin'](known,sha),'published inherited singleton bytes differ')

    env['load']=load
    env['publication']=publication
    return env


if __name__=='__main__':
    if sys.argv[1:]==['--assemble-local-manifest']:
        assemble()
    else:
        bound_api()['main']()

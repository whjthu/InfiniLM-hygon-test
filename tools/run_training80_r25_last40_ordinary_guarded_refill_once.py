"""Bind the final 40 fully observed ordinary incumbents to the deployed singleton guard."""

import json
from pathlib import Path
import runpy
import sys
from types import FunctionType
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
FIRST6 = ROOT / 'tools/run_training80_r25_remaining_ordinary_guarded_refill_once.py'
FIRST6_SHA = '020650b28165ca16ac94c8d937a466b3bb47a34ff3ce840fc2b402acae30dcc5'
FIRST6_CLAIM = ROOT / 'coordination/tasks/training80-r25-remaining-ordinary-first6-guarded-bf16-v1.json'
FIRST6_CLAIM_SHA = 'b36f7a978c2e0410bc102f75dc6493abdfd321ec1e4f429789244d915579af6f'
MANIFEST = LOCAL / 'training80-r25-remaining49-ordinary-incumbent-inventory-v1.json'
MANIFEST_SHA = '67508bee40a1359f9dd5ccd6409355afd6b25d2579f5deda9a3601ec80be7619'
STEM = 'training80-r25-last40-ordinary-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r25_last40_ordinary_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
NAMESPACE = UUID('3800d797-2def-453b-ad5e-634e2fb00b1d')
PARENTS = ('1103606','1194900')
PARTIAL = {'f12r3n17','f13r4n17','f13r1n08'}


def materials():
    from hashlib import sha256

    if FIRST6.resolve(strict=True)!=FIRST6 or sha256(FIRST6.read_bytes()).hexdigest()!=FIRST6_SHA:
        raise ValueError('published first-six guard source differs')
    prior=runpy.run_path(str(FIRST6))
    early,_=prior['sources']()
    first_claim=json.loads(early['pin'](FIRST6_CLAIM,FIRST6_CLAIM_SHA))
    manifest=json.loads(early['pin'](MANIFEST,MANIFEST_SHA))
    require=early['require']
    excluded=set(manifest['already_claimed_subset'])|set(first_claim['targets'])|set(manifest['ack_only_hold'])|{manifest['protected_n02']}
    rows={}
    for row in manifest['rows']:
        if row['original_counter_cards']==8 and row['node'] not in excluded:
            node=row['node']
            request=uuid5(NAMESPACE,f"{MANIFEST_SHA}/{row['parent']}/{node}/{row['previous']}").hex
            rows[node]=dict(row,new_request=request)
    require(len(rows)==40 and len({row['new_request'] for row in rows.values()})==40
            and sum(r['parent']==PARENTS[0] for r in rows.values())==16 and sum(r['parent']==PARENTS[1] for r in rows.values())==24
            and {row['node'] for row in manifest['rows'] if row['original_counter_cards']==7}==PARTIAL
            and not set(rows)&(excluded|PARTIAL|{'f12r2n10'}) and manifest['actual_terminals']==0
            and not {row['new_request'] for row in rows.values()} & {target['request'] for target in first_claim['targets'].values()},
            'remaining exact 40-node partition or protected scopes differ')
    return early,prior,rows


def batch_plan(rows):
    plan={}
    for parent in PARENTS:
        members=[row for row in rows.values() if row['parent']==parent]
        for offset in range(0,len(members),4):
            picked=members[offset:offset+4]
            plan[f'{parent}-{1+offset//4:02d}']={
                'parent':parent,'initial_budget_Beijing':picked[0]['budget_Beijing'],'last_budget_Beijing':picked[-1]['budget_Beijing'],
                'nodes':[{'node':r['node'],'old':r['previous'],'request':r['new_request'],'prepare':r['prep_Beijing'],
                          'predicted_end':r['budget_Beijing']} for r in picked]}
    return plan


def expected_claim(early,rows):
    return dict(scope=STEM,state='PREPARED_ONLY_ACTUAL_TERMINAL_REQUIRED',scope_mode='ONE_NODE_EXACT_ORDINARY_BRIDGE_STANDBY',
                launcher=WRAPPER.relative_to(ROOT).as_posix(),original_adoption_SHA256=early['ADOPTION'],
                inventory_SHA256=MANIFEST_SHA,first6_source_SHA256=FIRST6_SHA,first6_claim_SHA256=FIRST6_CLAIM_SHA,
                inherited_source_SHA256=early['digest'](Path(early['__file__']).read_bytes()),
                parent_batches=batch_plan(rows),GPU_per_target=8,max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False,
                targets={node:dict(parent=row['parent'],predecessor=row['previous'],previous_request=row['request'],
                           request=row['new_request'],prior_submit_SHA256=row['submit_SHA256']) for node,row in rows.items()})


def bound_api():
    early,prior,rows=materials()
    env=dict(early,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC=rows)
    for name,fn in early.items():
        if isinstance(fn,FunctionType) and fn.__globals__.get('__file__')==early['__file__']:
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
        early['require'](json.loads(raw)==expected_claim(early,rows),'fixed forty-target claim differs')
        return base,config,ordinary,old_plan,read,rows,early['digest'](raw)

    def publication(context,revision):
        original_publication(context,revision)
        _,publisher=context[3][-2]['modules']()
        for known,sha in ((FIRST6,FIRST6_SHA),(FIRST6_CLAIM,FIRST6_CLAIM_SHA),
                          (ROOT/'tools/run_training80_r25_early_ordinary_guarded_refill_once.py',prior['EARLY_SHA'])):
            ref=revision+':'+known.relative_to(ROOT).as_posix()
            early['require'](publisher['git']('cat-file','blob',ref)==early['pin'](known,sha),'published inherited guard differs')

    env['load']=load
    env['publication']=publication
    return env


if __name__=='__main__':
    if sys.argv[1:]==['--assemble-local-claim']:
        early,_,rows=materials()
        raw=(json.dumps(expected_claim(early,rows),sort_keys=True,indent=2)+'\n').encode()
        with DRAFT.open('xb') as stream:
            stream.write(raw)
        print(json.dumps(dict(claim_SHA256=early['digest'](raw),targets=len(rows),batches=len(batch_plan(rows)),GPU=0)))
    elif sys.argv[1:]==['--list-batches']:
        _,_,rows=materials()
        print(json.dumps(batch_plan(rows),sort_keys=True))
    else:
        bound_api()['main']()

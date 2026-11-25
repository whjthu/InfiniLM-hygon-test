"""Bind the next four observed ordinary incumbents to the deployed singleton guard."""

from datetime import datetime
import json
from pathlib import Path
import re
import runpy
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
EARLY = ROOT / 'tools/run_training80_r25_early_ordinary_guarded_refill_once.py'
EARLY_SHA = 'f64000177ec337dad0b097007440a0a727ebf4e99712f68bd4603d527922b9fe'
EARLY_CLAIM = ROOT / 'coordination/tasks/training80-r25-early-ordinary-guarded-bf16-v1.json'
EARLY_CLAIM_SHA = '2deae8bfea9f83d15b5398ae97db78b0fbc69459f9ae64a8e5a7031715240417'
STEM = 'training80-r25-next-ordinary-g3-g4-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r25_next_ordinary_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
CALENDAR = LOCAL / 'training80-r22-next-preparation-intents-20261125.json'
CALENDAR_SHA = 'f7d9a79afde6f4cf167f45c7b7da74a63eef2a735dda0754da5bff47d0a708d2'
GROWTH = {
    3: ('training80-1140-vnext-small-ack-progress-0634fb0c23e845c5ae2a8a72446a7f85.json',
        'd6f3cde448878a027fb4943a3c9cb9f01e629b97130106546b1902d299ddb550'),
    4: ('training80-1140-vnext-small-ack-progress-921442dc018e4cf394b9fd7e58d07451.json',
        'da7a0511bfca76c8381554bb90bfc0a04afce683b00624a81ce6c3a3ab04c401'),
}
SPEC = {
    'f12r3n02': (3,'1194900.1263','1194900.1191','30dae9426df15ccabc11b0f5c075b466','908d9247d75444df843b71a5ef394319'),
    'f13r1n05': (4,'1103606.3229','1103606.3114','67bea5b066c059318c5b8966fbdf05ae','8ea89a455312494db0aaf2eccd4792fa'),
    'f14r3n06': (3,'1194900.1262','1194900.1192','30dae9426df15ccabc11b0f5c075b466','ace6627d6cb5484084f9fc8fa435ea4b'),
    'f13r1n15': (4,'1103606.3227','1103606.3111','67bea5b066c059318c5b8966fbdf05ae','05f2f1c0eabe4f7480521e8760e6f750'),
}


def get_rows(early, old_plan):
    pin, require = early['pin'],early['require']
    calendar = json.loads(pin(CALENDAR,CALENDAR_SHA))
    ordinary = calendar['ordinary']
    excluded = calendar['ordinary_not_yet_independently_observed']
    completed_g12 = set(early['SPEC'])
    require(len(ordinary)==calendar['ordinary_count']==56 and excluded==['f12r3n20','f13r4n16']
            and not (completed_g12 | set(excluded) | {calendar['extra_n02']['node']}) & set(SPEC),
            'known 56-node coverage, prior scope or unknown fences changed')
    first = [row for row in ordinary if row['node'] not in completed_g12][:len(SPEC)]
    require([item['node'] for item in first]==list(SPEC),'new scope is not the exact first four after G1/G2')
    progress={}
    for group,(name,sha) in GROWTH.items():
        data=json.loads(pin(LOCAL/name,sha))
        require(data['rc']==0 and data['response']['GPU_steps_created']==0 and data['replay_allowed'] is False,
                'original independent no-write growth read differs')
        progress[group]=data['response']['nodes']
    rows={}
    for item in first:
        node=item['node']
        group,step,ancestor,claim_id,request=SPEC[node]
        rec=progress[group][node]
        start,budget=rec['worker_start_epoch'],rec['worker_seconds']
        prep=datetime.fromisoformat(item['check_from_Beijing']).timestamp()
        end=datetime.fromisoformat(item['predicted_end_Beijing']).timestamp()
        parent=step.split('.')[0]
        label_pci=sorted(rec['first'])
        require(group==item['group'] and rec['previous']==item['running_original']==step
                and rec['parent']==parent and rec['request']==item['request'] and rec['worker_PID']>0
                and rec['growth_DCU']==item['verified_positive_DCU']==8 and rec['pair_state']=='PASS'
                and len(label_pci)==len(set(label_pci))==8 and set(rec['second'])==set(label_pci)
                and all(re.fullmatch(r'[a-f0-9]{64}',rec[key]) for key in ('worker_SHA256','expected_SHA256','submit_SHA256'))
                and abs(start+budget-end)<.000002 and abs(prep+900-end)<.000002
                and old_plan[4]['request_ids'][group-1]==claim_id
                and any(row['node']==node and row['parent']==parent and row['previous']==ancestor for row in old_plan[-1][group-1]),
                'frozen incumbent, growth, timing or adopted original group differs')
        rows[node]=dict(node=node,parent=parent,previous=step,request=rec['request'],worker_SHA256=rec['worker_SHA256'],
                        expected_SHA256=rec['expected_SHA256'],submit_SHA256=rec['submit_SHA256'],cgroup_SHA256=None,
                        pci=label_pci,worker_start_epoch=start,worker_seconds=budget,previous_worker_pid=rec['worker_PID'],
                        planned_end_epoch=end,prepare_epoch=prep,prep_Beijing=item['check_from_Beijing'],
                        budget_Beijing=item['predicted_end_Beijing'],ledger_old=ancestor,ledger_request=claim_id,
                        new_request=request,published_group=group)
    require(len({row['new_request'] for row in rows.values()})==len(rows)
            and not {row['new_request'] for row in rows.values()} & ({row['request'] for row in rows.values()} | {entry[-1] for entry in early['SPEC'].values()}),
            'new unique requests overlap an old namespace')
    return rows


def expected_claim(early, rows):
    return dict(scope=STEM,state='PREPARED_ONLY_ACTUAL_TERMINAL_REQUIRED',scope_mode='ONE_NODE_EXACT_ORDINARY_BRIDGE_STANDBY',
                launcher=WRAPPER.relative_to(ROOT).as_posix(),original_adoption_SHA256=early['ADOPTION'],
                source_SHA256=EARLY_SHA,source_claim_SHA256=EARLY_CLAIM_SHA,calendar_SHA256=CALENDAR_SHA,
                max_invocations_per_node=1,GPU_per_target=8,ledger_mutations=0,replay_allowed=False,
                targets={node:dict(parent=row['parent'],previous_step=row['previous'],prior_standby_request=row['request'],
                         original_group_request=row['ledger_request'],original_group=row['published_group'],prior_device_mode='ORDINARY_BRIDGE_BF16',
                         prior_submit_SHA256=row['submit_SHA256'],request=row['new_request']) for node,row in rows.items()})


def base_api():
    from hashlib import sha256

    if EARLY.resolve(strict=True)!=EARLY or sha256(EARLY.read_bytes()).hexdigest()!=EARLY_SHA:
        raise ValueError('deployed G1/G2 wrapper bytes differ')
    early=runpy.run_path(str(EARLY))
    early['pin'](EARLY_CLAIM,EARLY_CLAIM_SHA)
    return early


def bound_api():
    early=base_api()
    env=dict(early,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC=SPEC)
    # Clone the audited function globals as one scope so read/intent/reply paths and publication all agree.
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
        rows=get_rows(early,old_plan)
        raw=early['pin'](DRAFT if draft and not CLAIM.exists() else CLAIM)
        early['require'](json.loads(raw)==expected_claim(early,rows),'published exact four-target claim differs')
        return base,config,ordinary,old_plan,read,rows,early['digest'](raw)

    def publication(context,revision):
        original_publication(context,revision)
        _,publisher=context[3][-2]['modules']()
        for known,sha in ((EARLY,EARLY_SHA),(EARLY_CLAIM,EARLY_CLAIM_SHA)):
            ref=revision+':'+known.relative_to(ROOT).as_posix()
            early['require'](publisher['git']('cat-file','blob',ref)==early['pin'](known,sha),'published inherited G1/G2 bytes differ')

    env['load']=load
    env['publication']=publication
    return env


if __name__=='__main__':
    bound_api()['main']()

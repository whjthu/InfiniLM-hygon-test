"""Prepare four fixed running incumbents for their next single guarded renewal."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r35-first-four-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r35_first_four_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
EARLY = ('tools/run_training80_r25_early_ordinary_guarded_refill_once.py',
         'f64000177ec337dad0b097007440a0a727ebf4e99712f68bd4603d527922b9fe')
N02 = ('tools/run_training80_n02_3212_guarded_refill_once.py',
       'e6dd2c4ee93cb502f3a3cde068e8a1e220a99bb9fdb72bbe5a2cde1769a54b66')
R34 = ('tools/run_training80_r34_four_terminal_guarded_refill_once.py',
       'd3f72317a56fc31363a14d2b9c017fa612c9781238e1ca3124fd3cd983e5b4e8')
CAL = ('training80-r34-1727-next-90min-original-worker-preparation-v1.json',
       '5d2bb1629266c6676badaaa87b4ef12edf2b2dec91280e76802521fc5ecadd7c')
N02_NODE = 'f13r1n02'
SPECS = {
    N02_NODE: ('1103606.3311', '1103606.3310',
               'b48c62cd52b3a58771a80c5c775e0c0b3bd16d7b67b3db0b723686730cd1616e', 10500),
    'f13r2n08': ('1103606.3313', '1103606.3312',
                '27402b3a60e98f1f6f451d2c9491561706c2f1c61bfcf6eccd3b4091c6b4514a', 10575),
    'f14r3n04': ('1194900.1323', '1194900.1322',
                'd6e29ec0d25fcaa759010416bb7e24f7590e1b61f07e6cb7cb947237259a8740', 10650),
    'f13r3n04': ('1194900.1325', '1194900.1324',
                '15a909ce74631aefec26e1e36014263b017904abd1714cb542e345b95d9ab505', 10725),
}


def _fn(function, **globals_):
    return FunctionType(function.__code__, dict(function.__globals__, **globals_), function.__name__,
                        function.__defaults__, function.__closure__)


def modules():
    fixed = ROOT / R34[0]
    if fixed.resolve(strict=True) != fixed or sha256(fixed.read_bytes()).hexdigest() != R34[1]:
        raise ValueError('frozen R34 source differs')
    r34 = runpy.run_path(str(fixed))
    for file, sha in (EARLY, N02):
        r34['pinned'](ROOT / file, sha)
    return r34, runpy.run_path(str(ROOT / EARLY[0])), runpy.run_path(str(ROOT / N02[0]))


def load():
    r34, old, n02 = modules()
    pin, require, digest = (r34[x] for x in ('pinned','require','digest'))
    context = old['load']()
    cal = json.loads(pin(LOCAL / CAL[0], CAL[1]))
    require(list(SPECS) == [r['node'] for r in cal['scheduled_preparation_90min'][:4]], 'four earliest exact scopes changed')
    rows, targets = {}, {}
    for evidence in cal['scheduled_preparation_90min'][:4]:
        node = evidence['node']
        step, cpu, reply_sha, renewal_seconds = SPECS[node]
        request = evidence['request']
        source = evidence['counter_receipt']
        receipt = json.loads(pin(LOCAL / source, evidence['counter_receipt_SHA256']))
        worker = receipt['result']['nodes'][node]
        require(evidence['step'] == worker['previous'] == step and worker['request'] == request
                and worker['status'] == 'RUNNING_WITH_ORIGINAL_BUDGET' and worker['pair_state'] == 'PASS'
                and len(worker['first']) == len(worker['second']) == worker['growth_DCU'] == 8
                and receipt['rc'] == 0 and worker['worker_start_epoch'] + worker['worker_seconds'] == evidence['expected_budget_end_epoch']
                and evidence['expected_budget_end_epoch'] - 900 == evidence['prepare_epoch'],
                'actual original worker/counter/budget proof changed')
        reply = (f'{n02["STEM"]}-reply.json' if node == N02_NODE else f'{old["STEM"]}-{node}-{request}-reply.json')
        outcome = json.loads(pin(LOCAL / reply, reply_sha))['transport']
        body = outcome['response']
        clearance = body['fresh_zero_GRES_accounting']
        require(outcome['rc'] == 0 and outcome['remote_outcome'] == 'RESPONSE_OBSERVED' and request == outcome['request'] == body['request']
                and body['GPU_srun_attempted'] is body['mutex_acquired'] is body['zero_GRES_call_attempted'] is True
                and body['probe_stderr_SHA256'] == digest(b'') and body['probe_step'] == cpu == clearance[0]
                and clearance[2:6] == ['COMPLETED','0:0',node,'cpu=1,mem=256M,node=1'], 'old exact successful probe/launch changed')
        new = uuid5(UUID('7ffe74da-dfa0-4111-bd16-6b3291ae9a43'), f'{node}/{step}/{request}').hex
        rows[node] = dict(node=node,parent=worker['parent'],previous=step,request=request,new_request=new,clearance=clearance,
                          cgroup_SHA256=None,worker_start_epoch=worker['worker_start_epoch'],worker_seconds=worker['worker_seconds'],
                          worker_SHA256=worker['worker_SHA256'],expected_SHA256=worker['expected_SHA256'],submit_SHA256=worker['submit_SHA256'],
                          previous_worker_pid=worker['worker_PID'],pci=sorted(worker['first']),
                          planned_end_epoch=evidence['expected_budget_end_epoch'],prepare_epoch=evidence['prepare_epoch'],renewal_seconds=renewal_seconds)
        targets[node] = dict(original_GPU=step,original_zero_GRES=cpu,prior_request=request,new_request=new,parent=worker['parent'],
                             prepare_epoch=evidence['prepare_epoch'],planned_worker_budget_end=evidence['expected_budget_end_epoch'],
                             worker_SHA256=worker['worker_SHA256'],expected_SHA256=worker['expected_SHA256'],
                             submit_SHA256=worker['submit_SHA256'],counter_receipt_SHA256=evidence['counter_receipt_SHA256'],
                             original_launcher_reply_SHA256=reply_sha,worker_seconds=renewal_seconds)
    claim = dict(scope=STEM,state='PREPARED_ONLY_ACTUAL_C0_AND_FRESH_DEVICE_REQUIRED',parent_ids=['1103606','1194900'],
                 source_calendar_SHA256=CAL[1],GPU_per_node=8,max_invocations_per_node=1,targets=targets,
                 launcher=WRAPPER.relative_to(ROOT).as_posix(),role_gate_N02='I01.Kimi-K3',ledger_mutations=0,replay_allowed=False)
    return r34, old, n02, context, rows, claim


def entry(material, claim_sha):
    r34,old,n02,context,rows,_=material
    require,pin,digest=(r34[k] for k in ('require','pinned','digest'))
    rep=old['unique']
    def budget(node,limit,request):
        row=rows[node]
        require(limit==10800 and request==row['new_request'] and 10500<=row['renewal_seconds']<=limit,'new worker budget changed')
        return row['renewal_seconds']
    context=(context[0],(*context[1][:7],budget),*context[2:-1],claim_sha)

    def read_program(ctx,row,nonce):
        source=old['read_program'](ctx,row,nonce)
        qualification = r'''
def qualify(x):
    out=group(x)
    folder=base/('training80-dcu-standby-'+x['request'])/x['node']
    worker,_=read(folder/'worker.json')
    returned,_=read(folder/'terminal.json')
    cgroup,cpin=read(folder/'worker-cgroup.json')
    submitted,spin=read(folder.parent/'submission-receipt.json')
    prior=x['clearance']
    rr=[r for line in query('sacct','-nP','-j',prior[0],'-o',fmt)
          if len(r:=line.split('|'))==9 and r[0]==prior[0]]
    cpu=rr[0] if len(rr)==1 else None
    out['original_terminal_and_targets_empty']=bool(
        out['original_terminal_and_targets_empty'] and cgroup and worker and
        returned and returned['epoch']>=x['planned_end_epoch'] and
        cgroup['node']==x['node'] and cgroup['step']==x['previous'] and cgroup['worker_pid']==worker['worker_pid']==x['previous_worker_pid']
        and cgroup['worker_start_ticks']>0 and cgroup['same_UID'] is cgroup['exact_job_and_step_cgroup'] is True
        and spin==x['submit_SHA256'] and submitted and submitted['request']==x['request'] and submitted['synthetic'] is True
        and cpu and cpu[:4]+cpu[5:7]+cpu[8:]==prior)
    out.update(worker_cgroup_SHA256=cpin,submit_SHA256=spin,original_zero_GRES_accounting=cpu)
    return out
'''
        source=rep(source,'print(json.dumps(dict(read_id=',qualification+'\nprint(json.dumps(dict(read_id=')
        source=rep(source,'targets=[group(item) for item in items]','targets=[qualify(item) for item in items]')
        compile(source,'<exact four successor predecessor read>','exec')
        return source

    def terminal_identity(row,nonce,proof):
        account,dpin=old['terminal_identity'](row,nonce,proof)
        [target]=proof['observed']['targets']
        cpin=target['worker_cgroup_SHA256']
        old_cpu=target['original_zero_GRES_accounting']
        require(proof['stderr_SHA256']==digest(b'') and target['submit_SHA256']==row['submit_SHA256']
                and old_cpu[:4]+old_cpu[5:7]+old_cpu[8:]==row['clearance'] and re.fullmatch(r'[a-f0-9]{64}',cpin or ''),
                'current natural predecessor C0 plus cgroup/old zero-GRES proof absent')
        return account,(dpin,cpin)

    def render(ctx,row,account,pins,revision):
        terminal,cpin=pins
        inherited=_fn(old['render'],STEM=STEM)
        guard=inherited(ctx,row,account,terminal,revision)
        source=guard['launch']
        anchor='            check_previous(account(PREVIOUS))'
        check=(
            "            prior_cgroup_path = original_dir / NODE / 'worker-cgroup.json'\n"
            "            require(prior_cgroup_path.resolve(strict=True)==prior_cgroup_path and prior_cgroup_path.stat().st_uid==os.getuid(), 'original worker cgroup owner differs')\n"
            "            with prior_cgroup_path.open('rb') as prior_cgroup_handle:\n"
            "                prior_cgroup_bytes = prior_cgroup_handle.read(8193)\n"
            f"            require(0<len(prior_cgroup_bytes)<=8192 and sha256(prior_cgroup_bytes).hexdigest()=={cpin!r}, 'original worker cgroup bytes differ')\n"
            "            prior_cgroup = json.loads(prior_cgroup_bytes)\n"
            "            require(prior_cgroup['node']==NODE and prior_cgroup['step']==PREVIOUS and prior_cgroup['worker_pid']==old_worker['worker_pid']\n"
            "                    and prior_cgroup['worker_start_ticks']>0 and prior_cgroup['same_UID'] is prior_cgroup['exact_job_and_step_cgroup'] is True, 'original worker cgroup identity differs')\n"
            f"            require(account({row['clearance'][0]!r}) == {row['clearance']!r}, 'original paired zero-GRES terminal differs')\n")
        guard['launch']=rep(source,anchor,check+anchor)
        compile(guard['launch'],'<exact four locked successor>', 'exec')
        return guard

    def fence(ordinary,authority,row):
        node=row['node']
        if node==N02_NODE:
            require(node not in context[0]['live_authority'](context[1]), 'n02 unexpectedly assigned to ordinary wave')
            n02['live_inference_priority']()
        else:
            ancestors=old['exact_no_other_claims'](ordinary,authority,context[5][node])
            require(all(a[1]!=row['previous'] and a[2]!=row['new_request'] for a in ancestors),
                    'current successor already governed by live ordinary ledger')

    def publication(ctx,revision):
        function=_fn(old['publication'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        function(ctx,revision)
        _,publisher=ctx[3][-2]['modules']()
        for file,digest_expected in (EARLY,N02,R34):
            require(publisher['git']('cat-file','blob',revision+':'+file)==pin(ROOT/file,digest_expected),'published replay fence dependency differs')

    common=dict(STEM=STEM)
    read_path=_fn(old['read_path'],**common)
    slot_paths=_fn(old['slot_paths'],**common)
    read=_fn(old['observe_predecessor'],read_program=read_program,read_path=read_path)
    terminal=_fn(old['terminal'],read_program=read_program,terminal_identity=terminal_identity)
    dispatch=_fn(old['dispatch'],slot_paths=slot_paths,read_path=read_path,publication=publication,terminal=terminal,
                 render=render,exact_no_other_claims=fence)
    observe=_fn(old['observe_original'],STEM=STEM,slot_paths=slot_paths,read_path=read_path,
                terminal_identity=terminal_identity,render=render)
    return context,read_program,read,render,dispatch,observe


def cgroup_fixture(launch,row,data,require):
    tree=ast.parse(launch)
    guard=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run_guarded')
    locked=next(node for node in ast.walk(guard) if isinstance(node,ast.With)
                and any(isinstance(item.optional_vars,ast.Name) and item.optional_vars.id=='lock' for item in node.items))
    start=next(i for i,node in enumerate(locked.body) if ast.unparse(node).startswith('prior_cgroup_path ='))
    stop=next(i for i,node in enumerate(locked.body) if 'original paired zero-GRES terminal differs' in ast.unparse(node))
    snippet=compile(ast.Module(body=locked.body[start:stop+1],type_ignores=[]),'<rendered cgroup/zero-GRES gate>', 'exec')
    for mode in ('match','drift','cpu'):
        with TemporaryDirectory() as tmp:
            base=Path(tmp).resolve()
            cell=base/row['node']
            cell.mkdir()
            path=cell/'worker-cgroup.json'
            path.write_bytes(data if mode!='drift' else data+b' ')
            env=dict(original_dir=base,NODE=row['node'],PREVIOUS=row['previous'],require=require,json=json,sha256=sha256,
                     os=SimpleNamespace(getuid=lambda:path.stat().st_uid),old_worker={'worker_pid':row['previous_worker_pid']},
                     account=lambda step:row['clearance'] if mode!='cpu' and step==row['clearance'][0] else None)
            try:
                exec(snippet,env)
            except ValueError as error:
                expected={'drift':'original worker cgroup bytes differ','cpu':'original paired zero-GRES terminal differs'}
                require(str(error)==expected.get(mode),'wrong shared lock cgroup or CPU veto')
            else:
                require(mode=='match','unsafe worker cgroup or CPU proof accepted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=SPECS)
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    material=load()
    r34,old,n02,_,rows,claim=material
    require,pin,digest=(r34[k] for k in ('require','pinned','digest'))
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one fixed node and exact actual predecessor receipt required')
    raw=pin(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact successor claim differs')
    sha=digest(raw)
    ctx,rr,read,render,dispatch,observe=entry(material,sha)
    if args.check:
        results={}
        for node,row in rows.items():
            out=rr(ctx,row,'1'*32)
            account=[row['previous'],f"training80-dcu-standby-{row['request'][:12]}-{node}",'COMPLETED','0:0',str(row['worker_seconds']),
                     node,old['GPU_TRES'],'2000-01-01T00:00:00','2000-01-01T03:00:00']
            sample=dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                        same_UID=True,exact_job_and_step_cgroup=True)
            raw_cgroup=json.dumps(sample).encode()
            program=render(ctx,row,account,('a'*64,digest(raw_cgroup)),None)
            baseline=_fn(old['render'],STEM=STEM)(ctx,row,account,'a'*64,None)
            old['host_fixture'](baseline['launch'],row,account,ctx[1][6])
            cgroup_fixture(program['launch'],row,raw_cgroup,require)
            results[node]=dict(request=row['new_request'],worker_seconds=program['worker_seconds'],
                               reader_fixture_SHA256=digest(out.encode()),rendered_fixture_SHA256=digest(program['launch'].encode()))
        n02['fixture_holds']()
        result=dict(state='PREPARED_ONLY',nodes=results,host_staging_positive=4,host_staging_vetoes=8,
                    locked_cgroup_and_cpu_positive=4,locked_cgroup_and_cpu_vetoes=8,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

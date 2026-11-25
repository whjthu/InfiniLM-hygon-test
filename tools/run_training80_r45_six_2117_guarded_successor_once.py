"""Guarded one-shot standby handoffs for six independently returned workers."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from types import SimpleNamespace
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r45-six-2117-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r45_six_2117_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
R44 = ('tools/run_training80_r44_four_2053_guarded_successor_once.py',
       'eb321d52a97d27a7b5707b3c0fd2c8bf2e22c525cda2c902923fc25b4f4b91ed')
R39 = ('tools/run_training80_r39_last40_twenty_three_guarded_successor_once.py',
       '2fa7e7ec704fb2665bf8ea9cb43a07decafe1fd828ee43b21da399bb3efc05d5')
READER = ('local-only/read_training80_r45_2117_exact_originals_once.py',
          '341b84c2a85b2baf45e6904640d8ee25ea049d1bd2ee352794b94e2c7e43af03')
ORIGINALS = {
    '1103606': ('local-only/training80-r45-2117-exact-originals-1103606-7f8bbe3f6e4d4a9686b92ef54ab8c251.json',
                '95552bab30777fe50a5a3bf3d885253bfef21dc3a680c28dcec83a8e0433fa92'),
    '1194900': ('local-only/training80-r45-2117-exact-originals-1194900-af67c921468a40c3a0b043bda25298f9.json',
                'b707b4b00f8983858d7dd2c709a5e49c058a923921f6e776442665ff3a114c59'),
}
ORDER = ('f13r1n05', 'f13r3n04', 'f14r3n06', 'f12r3n02', 'f14r3n04', 'f13r2n19')
UNSENT_NODES = {'f14r3n04', 'f13r2n19'}
ABSENT_CGROUP = ('f13r1n01', 'f13r2n01', 'f13r2n02', 'f13r3n01')
OWNER_LOCKS = ('f12r2n10', 'f12r2n11', 'f13r1n12', 'f13r2n06', 'f13r3n13')
EMPTY = sha256(b'').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(spec):
    path = ROOT / spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True) == path, 'exact local evidence path differs')
    with path.open('rb') as handle:
        raw = handle.read(262145)
    require(0 < len(raw) <= 262144 and digest(raw) == spec[1], 'exact local evidence bytes differ')
    return raw


def module(spec):
    pinned(spec)
    return runpy.run_path(str(ROOT / spec[0]))


def load():
    r44 = module(R44)
    budget, first, r34, early, n02, r44_context, _, _ = r44['load']()
    successor_context = budget['load']()[4]
    last40 = module(R39)['entry']()[0]()[5]
    routes = {}
    for node in ORDER:
        owners = [context[5][node] for context in (r44_context, successor_context, last40) if node in context[5]]
        require(len(owners) == 1, f'{node} exact accepted ancestry differs')
        routes[node] = owners[0]
    context = (*r44_context[:5], routes, r44_context[6])
    controller = budget['controller'](context[1][6]['controller.py'])
    reader = module(READER)
    inputs = reader['inputs'](reader['base_api']())
    require([row['node'] for row in inputs] == list(ORDER), 'exact independent reader cohort differs')
    rows, targets, shadow = {}, {}, {}
    for parent, receipt in ORIGINALS.items():
        packet = json.loads(pinned(receipt))
        observed = packet['observed']
        expected = [row for row in inputs if row['parent'] == parent]
        require(packet['rc'] == 0 and packet['parent'] == parent and packet['stderr_SHA256'] == EMPTY
                and packet['replay_allowed'] is False
                and packet['Slurm_mutations'] == packet['physical_admissions'] == observed['Slurm_mutations'] == observed['physical_admissions'] == 0
                and observed['plans_only'] is observed['requires_fresh_independent_device_and_dispatch_checks'] is True
                and packet['selected_nodes'] == [item['node'] for item in expected]
                and len(expected) == len(observed['targets']), 'exact independent parent-only evidence differs')
        source = reader['render'](reader['base_api'](), expected, observed['read_id'])
        require(packet['source_SHA256'] == digest(source.encode()), 'actual source for parent receipt differs')
        for old, proof in zip(expected, observed['targets'], strict=True):
            node, step, request = old['node'], old['previous'], old['request']
            require(proof['node'] == node and proof['prior_numbered'] == step and proof['prior_request'] == request
                    and proof['parent'] == parent and proof['scope'] == 'NATURAL' and proof['physical_clear'] is False
                    and proof['qualified_for_separate_fresh_physical'] is proof['original_natural_return'] is True
                    and proof['controller_numeric'] == proof['queue_numeric'] == [],
                    'actual natural return or current scheduler views differ')
            gpu = proof['original_accounting']
            cpu, = proof['original_CPU']
            files = proof['files_SHA256']
            start, seconds = old['worker_start_epoch'], old['worker_seconds']
            require(len(gpu) == len(cpu) == 9 and gpu[:4] == [step, f'training80-dcu-standby-{request[:12]}-{node}', 'COMPLETED', '0:0']
                    and gpu[5:7] == [node, early['GPU_TRES']] and gpu[7] < gpu[8] and int(gpu[4]) >= seconds
                    and cpu[:4]+cpu[5:7]+cpu[8:] == old['old_CPU'] and cpu[7] <= cpu[8]
                    and set(files) == {'cgroup','expected','submission','terminal','worker'}
                    and all(re.fullmatch(r'[a-f0-9]{64}', value or '') for value in files.values())
                    and files['worker'] == old['worker_SHA256'] and files['expected'] == old['expected_SHA256']
                    and files['submission'] == old['submit_SHA256'] and
                    (old['cgroup_SHA256'] is None or files['cgroup'] == old['cgroup_SHA256'])
                    and len(set(proof['pci'])) == 8 and proof['pci'] == old['pci']
                    and proof['worker_pid'] == old['previous_worker_pid'] > 0
                    and proof['worker_start_epoch'] == start and proof['worker_seconds'] == seconds
                    and proof['returned_epoch'] >= start+seconds, 'old paired CPU, PCI, worker or cgroup evidence differs')
            prior = old['prior_transport_not_sent']
            if prior:
                require(node in UNSENT_NODES and proof.get('previous_unsent_attempt') == dict(
                    request=prior['request'], prior_reply_SHA256=prior['reply_SHA256'], prior_intent_SHA256=prior['intent_SHA256'],
                    gpu_label_ids=[], cpu_label_ids=[], gpu_namespace_exists=False, control_namespace_exists=False,
                    history_and_namespaces_empty=True), 'prior positive no-send history is still open')
                shadow[node] = prior
            else:
                require(node not in UNSENT_NODES and 'previous_unsent_attempt' not in proof, 'unexpected former launch')
            new = uuid5(UUID('104935e1-c914-4c1f-b346-2f0882e31e1c'), f'{node}/{step}/{request}').hex
            renewal = controller['node_seconds'](node, 10800, new)
            require(new not in {request, prior['request'] if prior else None}, 'fixed new request overlaps original')
            rows[node] = dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'], clearance=old['old_CPU'], worker_start_epoch=start, worker_seconds=seconds,
                planned_end_epoch=start+seconds,prepare_epoch=start+seconds-900,pci=proof['pci'],
                worker_SHA256=files['worker'],expected_SHA256=files['expected'],submit_SHA256=files['submission'],
                cgroup_SHA256=files['cgroup'],previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal)
            targets[node] = dict(parent=parent,original_GPU=step,original_CPU=cpu[0],original_request=request,
                new_request=new,actual_prior_seconds=seconds,new_worker_seconds=renewal,
                Slurm_time_cap_seconds=renewal+300,original_worker_files=files,closed_prior_transport=prior)
    require(set(rows) == set(ORDER) and set(shadow) == UNSENT_NODES and not set(rows) & set(ABSENT_CGROUP + OWNER_LOCKS)
            and len({row['new_request'] for row in rows.values()}) == 6, 'strict successor/hold scope differs')
    claim = dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
        launcher=WRAPPER.relative_to(ROOT).as_posix(),original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()},
        original_local_reader_SHA256=READER[1],source_R44_SHA256=R44[1],source_R39_SHA256=R39[1],
        controller_SHA256=budget['CONTROLLER'][1],targets=targets,parent_serial_order=list(ORDER),
        previous_missing_cgroup_preserved=list(ABSENT_CGROUP),other_priority_states_preserved=list(OWNER_LOCKS),
        ready_GPU_count=0,GPU_per_node=8,max_invocations_per_node=1,max_concurrent_send=1,ledger_mutations=0,replay_allowed=False)
    return budget, first, r34, early, n02, context, rows, shadow, claim


def guarded_shadow(source, previous, node, unique):
    stub = previous['request'][:12]
    cpu = 'training80-r25-fresh-'+stub
    gpu = 'training80-dcu-standby-'+stub+'-'+node
    root = 'training80-dcu-standby-'+previous['request']
    extra = (
        '            previous_no_send_rows = [r.split("|") for r in query("sacct", "-n", "-P", "-j", PARENT, "-o", "JobIDRaw,JobName%110")]\n'
        '            require(any(len(r)==2 and r[0]==PARENT for r in previous_no_send_rows), "prior no-send accounting lacks exact parent")\n'
        f'            require(not any(len(r)==2 and r[1] in ({cpu!r},{gpu!r}) for r in previous_no_send_rows)\n'
        f'                    and not os.path.lexists(RESULTS / {root!r}) and not os.path.lexists(RESULTS / {previous["scope"]!r}),\n'
        '                    "prior no-send original namespace or label became visible")\n'
    )
    boundary = '            check_previous(account(PREVIOUS))'
    rendered = unique(source, boundary, extra+boundary)
    compile(rendered, '<prior unsent scope under the same shared HOST lock>', 'exec')
    return rendered


def runtime(material, claim_sha):
    budget, first, r34, early, n02, context, rows, shadow, claim = material
    fn = first['_fn']
    reader_context = dict(early)
    def read_source(ctx, row, nonce):
        source = early['read_program'](ctx,row,nonce)
        anchor = '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,'
        return early['unique'](source,anchor,
            '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'+anchor)
    reader_context['read_program'] = read_source
    original_render = early['render']
    def render_shadow(ctx, row, acct, terminal, revision):
        result = fn(original_render,STEM=STEM)(ctx,row,acct,terminal,revision)
        if row['node'] in shadow:
            result['launch'] = guarded_shadow(result['launch'],shadow[row['node']],row['node'],early['unique'])
        return result
    reader_context['render'] = render_shadow
    guarded = fn(budget['api'],STEM=STEM,G34=module(R44)['R29'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx, source, read, render, dispatch, observe = guarded((first,r34,reader_context,n02,context,rows,claim),claim_sha)
    inherited = dispatch.__globals__['publication']
    def publication(actual_context, revision):
        inherited(actual_context,revision)
        _, publisher = actual_context[3][-2]['modules']()
        for spec in (R44,R39,module(R44)['NEXT']):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pinned(spec),
                    'published guarded route or special prior scope differs')
    return ctx, source, read, render, fn(dispatch,publication=publication), observe


def shadow_fixture(launch, row, shadow):
    tree = ast.parse(launch)
    guard = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name=='run_guarded')
    lock = next(item for item in ast.walk(guard) if isinstance(item, ast.With)
                and any(isinstance(x.optional_vars, ast.Name) and x.optional_vars.id=='lock' for x in item.items))
    body = [ast.unparse(item) for item in lock.body]
    start = next(i for i,text in enumerate(body) if text.startswith('previous_no_send_rows ='))
    assert start < next(i for i,text in enumerate(body) if text.startswith('check_previous('))
    code = compile(ast.Module(body=lock.body[start:start+3],type_ignores=[]),'<rendered locked original no-send guard>', 'exec')
    stub = shadow['request'][:12]
    for case in ('clean','cpu','gpu','parent_missing','control_dir','gpu_dir'):
        parent = row['parent']
        report = [] if case == 'parent_missing' else [parent+'|parent-original']
        if case == 'cpu':
            report.append(parent+'.1|training80-r25-fresh-'+stub)
        if case == 'gpu':
            report.append(parent+'.1|training80-dcu-standby-'+stub+'-'+row['node'])
        existing = shadow['scope'] if case == 'control_dir' else ('training80-dcu-standby-'+shadow['request'] if case == 'gpu_dir' else None)
        environment = dict(PARENT=parent,RESULTS=Path('.'),query=lambda *args:report, require=require,
                           os=SimpleNamespace(path=SimpleNamespace(lexists=lambda path:path.name == existing)))
        try:
            exec(code,environment)
        except ValueError:
            require(case!='clean','matching prior unsent scope unexpectedly rejected')
        else:
            require(case=='clean','visible prior scope or incomplete accounting accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--check',action='store_true')
    choices.add_argument('--show-run-order',action='store_true')
    choices.add_argument('--observe-predecessor')
    choices.add_argument('--publication')
    choices.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=ORDER)
    parser.add_argument('--terminal-receipt',nargs=2)
    args = parser.parse_args()
    local = args.check or args.show_run_order
    require((local or args.node) and bool(args.publication) == bool(args.terminal_receipt),
            'one exact node and a fresh terminal receipt are required to send')
    material = load()
    budget, first, r34, _, n02, _, rows, shadow, claim = material
    raw = r34['pinned'](DRAFT if local and not CLAIM.exists() else CLAIM)
    require(json.loads(raw) == claim, 'published six-node guard scope differs')
    claim_sha = digest(raw)
    if args.show_run_order:
        result = dict(state='PREPARED_ONLY',ordered=[
            {key:rows[node][key] for key in ('node','parent','previous','new_request','renewal_seconds')} for node in ORDER],
            max_concurrent_send=1,replay_allowed=False)
    else:
        ctx, source, read, render, dispatch, observe = runtime(material,claim_sha)
        if args.check:
            fixtures = {}
            for node,row in rows.items():
                fresh = source(ctx,row,'1'*32)
                cgroup = json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],
                     worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
                host = render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)
                first['cgroup_fixture'](host['launch'],row,cgroup,require)
                seconds = budget['spawn_fixture'](host['launch'],row,ctx[1][6]['controller.py'])
                if node in shadow:
                    shadow_fixture(host['launch'],row,shadow[node])
                fixtures[node] = dict(request=row['new_request'],worker_seconds=seconds,fresh_reader_SHA256=digest(fresh.encode()))
            n02['fixture_holds']()
            result = dict(state='PREPARED_ONLY',targets=fixtures,HOST_spawn_positive=6,HOST_spawn_vetoes=12,
                locked_cgroup_CPU_positive=6,locked_cgroup_CPU_vetoes=12,locked_no_send_positive=2,locked_no_send_vetoes=10,
                SSH=0,Git=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            result = read(ctx,rows[args.node],args.observe_predecessor)
        elif args.observe_original:
            result = observe(ctx,rows[args.node])
        else:
            result = dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=claim_sha),sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()

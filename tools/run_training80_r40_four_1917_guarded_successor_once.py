"""Bind four exact split-parent releases to independent guarded successors."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r40-1917-four-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r40_four_1917_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R36=('tools/run_training80_r36_last40_two_guarded_successor_once.py',
     '15036fd95a3f45146c4d7e9b7f357b7b38f0cf29e948096dfff29576ee8e0ea7')
R29=('tools/run_training80_r29_other11_warning_guarded_refill_once.py',
     'c3162682176376fcd84099c09372ec3e3010de54c106e04278ed3752e132159b')
TERMINALS={
 '1103606':('training80-r40-1917-four-original-read-c94fc5d3405a4d51a6a52bdbc312d55e.json',
            '69c697d134274791592900394d64a1b026890d90acf9b64747304bbeaded73a8',
            ('f13r1n04','f13r4n08','f13r4n07')),
 '1194900':('training80-r40-1917-four-original-read-7d2b181cf9d5450bae65bc4acc4a2130.json',
            'd4e2ec4042fc1bdfe19bc6ec45a3fec1899c4be8ad135f04de318f2e2c218ba5',('f12r3n06',)),
}
COUNTER=('training80-r31-other11-four-later-counters-5c926c1957144f039ce1bd973aa50df1.json',
         '512b8a989e172c1608054e3fb6c9119618cf09a13705e7ddf683f9e6b8f0e1cb')
ROUTES={
 'f13r1n04':('1103606.3359','1103606.3358','a3a86e83d1bc14e74dc0d9ef53b9620d3226abb2dc44c6fe9b67b862974a1c0c'),
 'f12r3n06':('1194900.1379','1194900.1378','c4b8c4cf3cc5b5db69f80db1497e1ce159ebd85c617c01b00789aa42f16b6ef6'),
 'f13r4n08':('1103606.3366','1103606.3365','0be99c6237b2ece683872e78306cd9e2359c92ccf8128180eb83fbb20fdfd585'),
 'f13r4n07':('1103606.3364','1103606.3363','5ce05236c3dce914d31cb9a70f81ae97a2c113b7f219e34e4e4cb297d8271ffb'),
}
NODES=tuple(ROUTES)


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path,expected):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'fixed source or receipt path changed')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==expected,'fixed source or receipt bytes changed')
    return raw


def qualified_inputs(config):
    counters=json.loads(pin(LOCAL/COUNTER[0],COUNTER[1]))
    scope=counters['result']['nodes']
    require(counters['rc']==0 and counters['result']['GPU_steps_created']==0 and set(scope)==set(NODES),'fixed R31 counter receipt differs')
    actual, gap={},{}
    for parent,(filename,sha,expected_nodes) in TERMINALS.items():
        packet=json.loads(pin(LOCAL/filename,sha))
        body=packet['observed']
        require(packet['rc']==0 and packet['replay_allowed'] is False
                and packet['Slurm_mutations']==packet['physical_admissions']==body['Slurm_mutations']==body['physical_admissions']==0
                and body['plans_only'] is body['requires_fresh_independent_device_and_dispatch_checks'] is True
                and body['read_id'] in filename and tuple(r['node'] for r in body['targets'])==expected_nodes,
                'one immutable single-parent original read differs')
        for r in body['targets']:
            node=r['node']
            previous,cpu,reply_sha=ROUTES[node]
            v=scope[node]
            hashes=r['files_SHA256']
            require(r['parent']==v['parent']==parent and r['request']==v['request']==config[5][node]['new_request']
                    and r['numbered_predecessor']==v['previous']==previous and r['old_cpu_accounting'][0]==cpu
                    and r['qualified_original_release'] is r['natural_return'] is True and r['physical_clear'] is False
                    and r['controller_rows']==r['queue_rows']==[] and v['growth_DCU']==8 and v['pair_state']=='PASS'
                    and r['pid']==v['worker_PID'] and r['worker_start_epoch']==v['worker_start_epoch'] and r['worker_seconds']==v['worker_seconds']
                    and hashes['worker']==v['worker_SHA256'] and hashes['expected']==v['expected_SHA256'] and hashes['submission']==v['submit_SHA256']
                    and sorted(v['first'])==sorted(v['second'])==sorted(r['pci'])==sorted(config[5][node]['pci']),
                    'sealed original, fixed R31 worker or same-device authority differs')
            actual[node]=dict(gpu=r['old_gpu_accounting'],cpu=r['old_cpu_accounting'],worker_records=hashes,
                              controller_rows=r['controller_rows'],queue_rows=r['queue_rows'],natural_return=r['natural_return'],
                              original_released_for_fresh_physical_preflight=r['qualified_original_release'],
                              worker_pid=r['pid'],worker_start_ticks=r['cgroup_start_ticks'],worker_seconds=r['worker_seconds'],
                              worker_start_epoch=r['worker_start_epoch'],returned_epoch=r['returned_epoch'])
            gap[node]=dict(last40_prior_request=r['request'],parent=parent,numbered_predecessor=previous,
                           launch_budget_seconds=r['worker_seconds'],old_CPU=r['old_cpu_accounting'][:4]+r['old_cpu_accounting'][5:7]+r['old_cpu_accounting'][8:],
                           prior_reply_SHA256=reply_sha)
    require(set(actual)==set(gap)==set(NODES),'candidate count or separate parents differ')
    return actual,gap


def entry():
    path=ROOT/R36[0]
    raw=pin(path,R36[1])
    inherited=runpy.run_path(str(path))
    scope=dict(inherited,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,LAST40=R29,NODES=NODES,
               __file__=__file__,qualified_inputs=qualified_inputs)
    tree=ast.parse(raw)
    item=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    loader=ast.get_source_segment(raw.decode(),item)
    a=loader.index('    terminal=json.loads(read(LOCAL/TERMINAL[0],TERMINAL[1]))')
    z=loader.index('    rows,targets={},{}',a)
    loader=loader[:a]+'    verified,gap=qualified_inputs(config)\n'+loader[z:]
    replacements={
        "        actual=view['nodes'][node]":"        actual=verified[node]",
        'source_terminal_SHA256=TERMINAL[1],\n               source_gap_SHA256=CROSSWALK[1],last40_ancestry_SHA256=LAST40[1],runtime_guard_SHA256=NEXT[1],':
            'separate_parent_terminal_SHA256=split_receipts,\n               source_counter_SHA256=pinned_counter,prior_r29_route_SHA256=LAST40[1],runtime_guard_SHA256=NEXT[1],',
    }
    for prev,new in replacements.items():
        require(loader.count(prev)==1,'existing exact successor source boundary differs')
        loader=loader.replace(prev,new,1)
    scope.update(split_receipts={p:t[1] for p,t in TERMINALS.items()},pinned_counter=COUNTER[1])
    exec(compile(loader,'<four exact terminal receipts by parent>', 'exec'),scope)
    prior_api=inherited['api']
    def api(material,claim_sha):
        fn=material[1]['_fn']
        bound=fn(prior_api,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,LAST40=R29,__file__=__file__)
        context,reader,read,render,dispatch,observe=bound(material,claim_sha)
        earlier=dispatch.__globals__['publication']
        def publication(ctx,revision):
            earlier(ctx,revision)
            _,publisher=ctx[3][-2]['modules']()
            require(publisher['git']('cat-file','blob',revision+':'+R36[0])==raw,'published exact original launcher differs')
        return context,reader,read,render,fn(dispatch,publication=publication),observe
    scope['api']=api
    main_ast=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    main_src=ast.get_source_segment(raw.decode(),main_ast)
    for count in ('actual_HOST_spawn','locked_cgroup_and_CPU'):
        old=f'{count}_positive=2,{count}_vetoes=4'
        new=f'{count}_positive=len(nodes),{count}_vetoes=2*len(nodes)'
        require(main_src.count(old)==1,'exact inherited fixture summary differs')
        main_src=main_src.replace(old,new,1)
    exec(compile(main_src,'<four singleton original dispatch paths>', 'exec'),scope)
    return scope['load'],scope['main']


if __name__=='__main__':
    _,main=entry()
    main()

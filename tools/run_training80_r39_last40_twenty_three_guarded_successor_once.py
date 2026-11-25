"""Single-node standby handoffs for the 23 independently returned last40 workers."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r39-last40-23-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r39_last40_twenty_three_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R36=('tools/run_training80_r36_last40_two_guarded_successor_once.py',
     '15036fd95a3f45146c4d7e9b7f357b7b38f0cf29e948096dfff29576ee8e0ea7')
TERMINAL=('training80-1841-last40-23-exact-original-terminal-2c42be59d32540b9b90cd2c33e320186/receipt.json',
          '51d252a0a3e3f61cf70b3d98fc9fded092f1c81ad0a37933758ddb8d2b312b3b')
CROSSWALK=('training80-1841-last40-23-exact-incumbent-crosswalk-v1.json',
           '475a8ae01c62ce6d6529e3882b9370e048929cab170b4108a787b9d1161c7e8e')
EXPECTED={
 '1103606':('f13r1n03','f13r1n13','f13r1n20','f13r2n04','f13r2n06','f13r2n09','f13r3n14'),
 '1194900':('f12r1n17','f12r1n18','f12r3n04','f12r3n07','f12r3n08','f12r3n09','f12r3n12','f12r3n15',
            'f12r3n16','f12r3n19','f13r2n19','f13r2n20','f13r3n03','f13r3n10','f13r4n15','f14r3n03'),
}
EXCLUDED=('f12r1n16','f12r2n10','f12r3n14','f13r1n11','f13r1n12','f13r2n10','f13r2n11','f14r3n05')
NODES=tuple(n for nodes in EXPECTED.values() for n in nodes)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def pinned(path,expected):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'exact source or private receipt path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==expected,'exact source or private receipt bytes differ')
    return raw


def project(response,gap):
    require(response['Slurm_mutations']==0 and response['physical_admitted_count']==0
            and response['qualified_original_count']==len(NODES)==23
            and set(response['nodes'])==set(gap)==set(NODES) and not set(NODES)&set(EXCLUDED),
            '23 original-only independent terminal cohort differs')
    recovered={}
    for parent,nodes in EXPECTED.items():
        for node in nodes:
            fact=response['nodes'][node]
            original=gap[node]
            require(original['parent']==parent and original['node']==node and original['authentic_launch_proven'] is True
                    and fact['qualified_original_release'] is fact['natural_return'] is True and fact['physical_clear'] is False
                    and fact['controller_rows']==fact['queue_rows']==[]
                    and fact['old_gpu_accounting'][0]==original['numbered_predecessor']
                    and fact['old_gpu_accounting'][1]==original['expected_accounting_label']
                    and sorted(fact['pci'])==sorted(original['route']['required_same_PCI']),
                    'exact qualified node, PCI, or natural return is not in the parent cut')
            recovered[node]=dict(gpu=fact['old_gpu_accounting'],cpu=fact['old_cpu_accounting'],worker_records=fact['files_SHA256'],
                                 controller_rows=fact['controller_rows'],queue_rows=fact['queue_rows'],natural_return=fact['natural_return'],
                                 original_released_for_fresh_physical_preflight=fact['qualified_original_release'],
                                 worker_pid=fact['pid'],worker_start_ticks=fact['cgroup_start_ticks'],worker_seconds=fact['worker_seconds'],
                                 worker_start_epoch=fact['worker_start_epoch'],returned_epoch=fact['returned_epoch'])
    return recovered


def entry():
    path=ROOT/R36[0]
    raw=pinned(path,R36[1])
    inherited=runpy.run_path(str(path))
    scope=dict(inherited,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,TERMINAL=TERMINAL,CROSSWALK=CROSSWALK,NODES=NODES,
               __file__=__file__,project=project)
    tree=ast.parse(raw)
    legacy=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    original=ast.get_source_segment(raw.decode(),legacy)
    changes={
        "and view['new_GPU']==0 and view['physical_clear'] is False and set(view['nodes'])==set(gap)==set(NODES),":
            "and view['Slurm_mutations']==0 and view['physical_admitted_count']==0 and set(view['nodes'])==set(gap)==set(NODES),",
        "    rows,targets={},{}":"    projected=project(view,gap)\n    rows,targets={},{}",
        "        actual=view['nodes'][node]":"        actual=projected[node]",
        """        reply_sha=declared.get('prior_reply_SHA256') or {
            'f13r2n11':'a9a98d6ad6e44ed3931e356f3d39d8214b0b4cfb48e10309c01d14f8ce39544a',
            'f14r3n05':'db97d96b3b48d9d9bf67e36b3bf2b2585151baf562dc24562bb2f6c389a8e8a5',
        }[node]""":"        reply_sha=declared['original_launch_reply_SHA256']",
    }
    for before,after in changes.items():
        require(original.count(before)==1,'inherited exact successor evidence adapter changed')
        original=original.replace(before,after,1)
    exec(compile(original,'<23 actual last40 terminal worker evidence>', 'exec'),scope)
    loader=scope['load']
    def load():
        material=loader()
        contract=material[-2]
        contract.update(source_immutable_two_node_launcher_SHA256=R36[1],excluded_existing_scopes=list(EXCLUDED),
                        ready_GPU_count=0,dispatch_scope='ONE_NODE_PER_PROCESS_AND_GLOBAL_SHARED_LOCK')
        return material
    inherited_api=inherited['api']
    def api(material,claim_sha):
        fn=material[1]['_fn']
        bound=fn(inherited_api,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,reader,read,render,dispatch,observe=bound(material,claim_sha)
        earlier=dispatch.__globals__['publication']
        def publication(ctx,revision):
            earlier(ctx,revision)
            _,publisher=ctx[3][-2]['modules']()
            require(publisher['git']('cat-file','blob',revision+':'+R36[0])==raw,'published exact original launcher differs')
        return context,reader,read,render,fn(dispatch,publication=publication),observe
    scope.update(load=load,api=api)
    main_ast=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    main_src=ast.get_source_segment(raw.decode(),main_ast)
    for count in ('actual_HOST_spawn','locked_cgroup_and_CPU'):
        old=f'{count}_positive=2,{count}_vetoes=4'
        new=f'{count}_positive=len(nodes),{count}_vetoes=2*len(nodes)'
        require(main_src.count(old)==1,'inherited fixture reporting changed')
        main_src=main_src.replace(old,new,1)
    exec(compile(main_src,'<23 node-by-node fixture and controller entry>', 'exec'),scope)
    return load,api,scope['main']


def show_run_order(load):
    material=load()
    rows=material[-3]
    raw=pinned(LOCAL/TERMINAL[0],TERMINAL[1])
    released=json.loads(raw)['response']['nodes']
    ordered=sorted(rows,key=lambda node:released[node]['returned_epoch'])
    print(json.dumps(dict(state='PREPARED_ONLY_EACH_NODE_REQUIRES_A_FRESH_READ_AND_PHYSICAL_PASS',
                          ordered=[dict(node=n,parent=rows[n]['parent'],previous=rows[n]['previous'],request=rows[n]['new_request'],
                                        renewal_seconds=rows[n]['renewal_seconds']) for n in ordered],
                          max_concurrent_dispatch=1,replay_allowed=False),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    load,_,main=entry()
    if sys.argv[1:]==['--show-run-order']:
        show_run_order(load)
    else:
        main()

"""Bind the exact R33 n13 natural release to one independent standby successor."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r41-n13-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r41_n13_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R36=('tools/run_training80_r36_last40_two_guarded_successor_once.py',
     '15036fd95a3f45146c4d7e9b7f357b7b38f0cf29e948096dfff29576ee8e0ea7')
R33=('tools/run_training80_r33_n13_preauth_guarded_refill_once.py',
     '24495989c454e060a3aa44057af83ce2244f26e5bb44b9dad441c377a3867372')
TERMINAL=('training80-r41-n13-next-original-read-7513bbdbab8d45a69797264addc9a315.json',
          'c6ed96685b4c71f4ef112b5d18c91368a8808984d503b8fcb511cf57390320cb')
COUNTER=('training80-r33-n13-exact-paired-counters-76a3b44a8e2d45139ebfdd4d898197df.json',
         'f5673be6fcf57e620a94f29916c85d480f16438ff4933b214fd25356dabdf1cb')
OLD_REPLY='5de10c50b788f1a75c0cf4ec61ed105b88687bfd73f304b068d7059621c45f1f'
NODE='f12r3n13'
STEP='1194900.1385'
CPU='1194900.1384'
NODES=(NODE,)


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def pinned(path,expected):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'fixed source or proof path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==expected,'fixed source or proof bytes differ')
    return raw


def qualified(config):
    terminal=json.loads(pinned(LOCAL/TERMINAL[0],TERMINAL[1]))
    body=terminal['observed']
    [r]=body['targets']
    counter=json.loads(pinned(LOCAL/COUNTER[0],COUNTER[1]))
    v=counter['result']['nodes'][NODE]
    files=r['files_SHA256']
    require(terminal['rc']==0 and terminal['scope']=='n13-next' and terminal['replay_allowed'] is False
            and terminal['Slurm_mutations']==terminal['physical_admissions']==body['Slurm_mutations']==body['physical_admissions']==0
            and r['node']==NODE and r['parent']==v['parent']=='1194900' and r['original_numbered']==v['previous']==STEP
            and r['old_cpu_accounting'][0]==CPU and r['controller_rows']==r['queue_rows']==[]
            and r['qualified_original_release'] is True and r['physical_clear'] is False
            and r['request']==v['request']==config[5][NODE]['new_request'] and r['pid']==v['worker_PID']
            and r['worker_start_epoch']==v['worker_start_epoch'] and r['worker_seconds']==v['worker_seconds']
            and counter['rc']==0 and v['growth_DCU']==8 and v['pair_state']=='PASS'
            and files['worker']==v['worker_SHA256'] and files['expected']==v['expected_SHA256'] and files['submission']==v['submit_SHA256']
            and sorted(r['pci'])==sorted(v['first'])==sorted(v['second'])==sorted(config[5][NODE]['pci']),
            'actual n13 terminal, exact R33 claim ancestry or same PCI differs')
    actual=dict(gpu=r['old_gpu_accounting'],cpu=r['old_cpu_accounting'],worker_records=files,controller_rows=r['controller_rows'],
                queue_rows=r['queue_rows'],natural_return=True,original_released_for_fresh_physical_preflight=r['qualified_original_release'],
                worker_pid=r['pid'],worker_start_ticks=r['cgroup_start_ticks'],worker_seconds=r['worker_seconds'],
                worker_start_epoch=r['worker_start_epoch'],returned_epoch=r['returned_epoch'])
    gap=dict(last40_prior_request=r['request'],parent=r['parent'],numbered_predecessor=STEP,launch_budget_seconds=r['worker_seconds'],
             old_CPU=r['old_cpu_accounting'][:4]+r['old_cpu_accounting'][5:7]+r['old_cpu_accounting'][8:],prior_reply_SHA256=OLD_REPLY)
    return {NODE:actual},{NODE:gap}


def entry():
    path=ROOT/R36[0]
    raw=pinned(path,R36[1])
    old=runpy.run_path(str(path))
    scope=dict(old,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,LAST40=R33,NODES=NODES,__file__=__file__,
               qualified=qualified,source_terminal=TERMINAL[1],source_counter=COUNTER[1])
    tree=ast.parse(raw)
    origin=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    loader=ast.get_source_segment(raw.decode(),origin)
    a=loader.index('    terminal=json.loads(read(LOCAL/TERMINAL[0],TERMINAL[1]))')
    z=loader.index('    rows,targets={},{}',a)
    loader=loader[:a]+'    verified,gap=qualified(config)\n'+loader[z:]
    changes={
        "        actual=view['nodes'][node]":"        actual=verified[node]",
        'source_terminal_SHA256=TERMINAL[1],\n               source_gap_SHA256=CROSSWALK[1],last40_ancestry_SHA256=LAST40[1],runtime_guard_SHA256=NEXT[1],':
            'source_terminal_SHA256=source_terminal,\n               source_counter_SHA256=source_counter,prior_R33_route_SHA256=LAST40[1],runtime_guard_SHA256=NEXT[1],',
    }
    for before,after in changes.items():
        require(loader.count(before)==1,'existing exact terminal adapter changed')
        loader=loader.replace(before,after,1)
    exec(compile(loader,'<single confirmed n13 release>', 'exec'),scope)
    inherited=old['api']
    def api(material,claim_sha):
        fn=material[1]['_fn']
        bind=fn(inherited,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,LAST40=R33,__file__=__file__)
        context,reader,read,render,dispatch,observe=bind(material,claim_sha)
        previous=dispatch.__globals__['publication']
        def publication(ctx,revision):
            previous(ctx,revision)
            _,publisher=ctx[3][-2]['modules']()
            require(publisher['git']('cat-file','blob',revision+':'+R36[0])==raw,'published exact parent launcher differs')
        return context,reader,read,render,fn(dispatch,publication=publication),observe
    scope['api']=api
    main_ast=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    main_src=ast.get_source_segment(raw.decode(),main_ast)
    for count in ('actual_HOST_spawn','locked_cgroup_and_CPU'):
        match=f'{count}_positive=2,{count}_vetoes=4'
        replacement=f'{count}_positive=len(nodes),{count}_vetoes=2*len(nodes)'
        require(main_src.count(match)==1,'existing inherited fixture summary differs')
        main_src=main_src.replace(match,replacement,1)
    exec(compile(main_src,'<exact n13 executable entry>', 'exec'),scope)
    return scope['load'],scope['main']


if __name__=='__main__':
    _,main=entry()
    main()

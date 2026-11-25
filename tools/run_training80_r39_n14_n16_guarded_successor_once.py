"""Freeze n14 and n16 handoffs without assuming either worker has returned."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r39-n14-n16-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r39_n14_n16_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
NEXT=('tools/run_training80_r35_next_three_guarded_successor_once.py',
      'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
LAST40=('tools/run_training80_r25_last40_ordinary_guarded_refill_once.py',
        '575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e')
N16=('tools/run_training80_r29_n16_warning_diagnostic_guarded_refill_once.py',
     '88d8ece33936d23e48a3c5ff1bd16b0ef504dd236f5ae5060eeacf5731befab5')
PLAN=('training80-1815-exact-gap-and-next-budget-plan-v1.json',
      '3df0f818ef09a99a08327592653eb24781acfb18a89aea41ab0659ec6f9a9501')
SPECIAL='f12r1n16'
SPECS={
    'f12r3n14':('1194900.1374','1194900.1373','37afde67d6e2fcf386619c55e27a881e86b249b4427136f526b9344ded6156f8'),
    SPECIAL:('1194900.1377','1194900.1376','cff0ce2c71ab4b1c114e83e3bf83f355b54bb18089e0752d35be1c7934904cca'),
}


def api():
    path=ROOT/NEXT[0]
    raw=path.read_bytes()
    if path.resolve(strict=True)!=path or sha256(raw).hexdigest()!=NEXT[1]:
        raise ValueError('exact controller-budget guard differs')
    next_scope=runpy.run_path(str(path))
    first=next_scope['module'](next_scope['FIRST'])
    n16=next_scope['module'](N16)
    fn=first['_fn']
    require=next_scope['require']
    prior_n16=n16['bound_api']()['load']()
    [actual_special]=prior_n16[5]
    require(actual_special==SPECIAL and prior_n16[5][SPECIAL]['new_request']==n16['REQUEST']
            and prior_n16[5][SPECIAL]['parent']=='1194900','exact prior diagnostic route changed')

    def original_request(node,ctx):
        return prior_n16[5][node]['new_request'] if node==SPECIAL else ctx[5][node]['new_request']

    def receipt_family(node,ordinary):
        return n16['STEM'] if node==SPECIAL else ordinary['STEM']

    # Reuse the frozen actual-budget and paired-counter validation for just
    # these calendar rows. The n16 old dispatch used its separate diagnosis.
    env=dict(next_scope,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__,SPECS=SPECS,G34=LAST40,
             original_request=original_request,receipt_family=receipt_family)
    tree=ast.parse(raw)
    load_ast=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='load')
    source=ast.get_source_segment(raw.decode(),load_ast)
    changes={
        "cal['scheduled_preparation_90min'][4:7]":"cal['scheduled_preparation_90min'][12:14]",
        "request==worker['request']==ctx[5][node]['new_request']":"request==worker['request']==original_request(node,ctx)",
        "{ancestor['STEM']}-{node}-{request}-reply.json":"{receipt_family(node,ancestor)}-{node}-{request}-reply.json",
        'old_G3G4_scope_SHA256=':'old_last40_scope_SHA256=',
        'current exact three-node sequence changed':'current exact n14/n16 sequence changed',
    }
    for before,after in changes.items():
        require(source.count(before)==1,'inherited case selection drifted')
        source=source.replace(before,after,1)
    exec(compile(source,'<n14 and n16 immutable case selection>', 'exec'),env)
    bound=fn(next_scope['api'],STEM=STEM,G34=LAST40,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)

    def load():
        material=env['load']()
        plan=json.loads(next_scope['read_pinned'](LOCAL/PLAN[0],PLAN[1]))
        exact={r['node']:r for r in plan['after_first_eleven_exact_preparation'] if r['node'] in SPECS}
        require(set(exact)==set(SPECS)
                and all(item['step']==material[-2][node]['previous'] and item['parent']=='1194900'
                        and item['worker_budget_seconds']==material[-2][node]['worker_seconds']
                        for node,item in exact.items()),
                'exact planning subset differs')
        require(material[4][5][SPECIAL]['new_request']==n16['OLD_REQUEST']
                and material[4][5][SPECIAL]['previous']==prior_n16[5][SPECIAL]['previous'],
                'original last40 booked ancestry or rejected n16 request differs')
        material[-1].update(prior_n16_distinct_diagnostic_route_SHA256=N16[1],source_plan_SHA256=PLAN[1],
                             fixed_budget_runtime_SHA256=NEXT[1],excluded_business_nodes=['f12r2n10','f13r1n12'])
        return material

    def runtime(material,claim_sha):
        context,reader,read,render,dispatch,observe=bound(material,claim_sha)
        inherited_publish=dispatch.__globals__['publication']
        def publication(ctx,revision):
            inherited_publish(ctx,revision)
            _,publisher=ctx[3][-2]['modules']()
            for dep in (NEXT,N16):
                file,sha=dep
                require(publisher['git']('cat-file','blob',revision+':'+file)==next_scope['read_pinned'](ROOT/file,sha),
                        'published n16 ancestor or exact budget runtime differs')
        return context,reader,read,render,fn(dispatch,publication=publication),observe

    return next_scope,load,runtime


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
    inherited,load,runtime=api()
    require,digest=inherited['require'],inherited['digest']
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),
            'one fixed node and an actual old receipt are required')
    material=load()
    first,r34,early,_,_,rows,claim=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact n14/n16 publication differs')
    sha=digest(raw)
    context,reader,read,render,dispatch,observe=runtime(material,sha)
    if args.check:
        nodes={}
        for node,row in rows.items():
            program=reader(context,row,'6'*32)
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                   same_UID=True,exact_job_and_step_cgroup=True)).encode()
            gpu=[row['previous'],f"training80-dcu-standby-{row['request'][:12]}-{node}",'COMPLETED','0:0',str(row['worker_seconds']),
                 node,early['GPU_TRES'],'2000-01-01T00:00:00','2000-01-01T03:00:00']
            guard=render(context,row,gpu,('a'*64,digest(cgroup)),None)
            first['cgroup_fixture'](guard['launch'],row,cgroup,require)
            seconds=inherited['spawn_fixture'](guard['launch'],row,context[1][6]['controller.py'])
            nodes[node]=dict(prior_GPU=row['previous'],request=row['new_request'],worker_seconds=seconds,
                             reader_SHA256=digest(program.encode()),host_SHA256=digest(guard['launch'].encode()))
        result=dict(state='PREPARED_ONLY',nodes=nodes,actual_HOST_spawn_positive=2,actual_HOST_spawn_vetoes=4,
                    locked_cgroup_and_CPU_positive=2,locked_cgroup_and_CPU_vetoes=4,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(context,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(context,rows[args.node])
    else:
        result=dispatch(context,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

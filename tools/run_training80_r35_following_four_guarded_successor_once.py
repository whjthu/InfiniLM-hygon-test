"""Prepare four already observed R29 workers for separate guarded renewals."""

import argparse
import ast
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r35-following-four-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r35_following_four_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
NEXT = ('tools/run_training80_r35_next_three_guarded_successor_once.py',
        'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
FIRST6 = ('tools/run_training80_r25_remaining_ordinary_guarded_refill_once.py',
          '020650b28165ca16ac94c8d937a466b3bb47a34ff3ce840fc2b402acae30dcc5')
FLEET = ('training80-post-renewal-scheduler-b1d79e514fc84917a76fb0e83aedb68c/receipt.json',
         '5c26090833df7d43dc34cd12e4b34f26073afd90e7f79512273dab870665b4ce')
SPECS = {
    'f13r4n14': ('1194900.1333', '1194900.1332', '9a29ad621982c2ed786bd4f7fec7656d94d9fb70f30135b3f8befb8ee4238807'),
    'f13r3n09': ('1194900.1331', '1194900.1330', '775db10cdf22fdfa3f6b2a42b46f2cf4654bd2836bd79a50d7ba1c220735f70f'),
    'f13r1n07': ('1103606.3319', '1103606.3318', 'e001c7bebf968d315097b5cedeb86a408a56a79a2efe7f1dce4e8615f1c127ae'),
    'f13r2n05': ('1103606.3321', '1103606.3320', '61239e6f4c78a8711f3b9c713d3e99b6675bb8aa84a3864a06a0a2eddd239947'),
}


def exact_api():
    from hashlib import sha256

    file, expected=NEXT
    path=ROOT/file
    raw=path.read_bytes()
    if path.resolve(strict=True)!=path or sha256(raw).hexdigest()!=expected:
        raise ValueError('exact controller-budget guarded source differs')
    inherited=runpy.run_path(str(path))
    first=inherited['module'](inherited['FIRST'])
    fn=first['_fn']
    require=inherited['require']
    env=dict(inherited,SPECS=SPECS,G34=FIRST6,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    # Only the exact calendar selection and its predecessor-source name differ.
    module=ast.parse(raw)
    original=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='load')
    source=ast.get_source_segment(raw.decode(),original)
    swaps={
        "cal['scheduled_preparation_90min'][4:7]":"cal['scheduled_preparation_90min'][7:11]",
        'old_G3G4_scope_SHA256=':'old_first6_scope_SHA256=',
        'current exact three-node sequence changed':'current exact four-node sequence changed',
    }
    for a,b in swaps.items():
        require(source.count(a)==1,'inherited singleton input selection differs')
        source=source.replace(a,b,1)
    exec(compile(source,'<exact four newly absent candidates from the pinned calendar>', 'exec'),env)
    build=fn(inherited['api'],STEM=STEM,G34=FIRST6,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)

    def load():
        r=env['load']()
        status=json.loads(inherited['read_pinned'](LOCAL/FLEET[0],FLEET[1]))
        response=status['response']
        absent=response['snapshot']['no_numbered']
        require(status['rc']==0 and status['remote_outcome']=='RESPONSE_OBSERVED'
                and response['snapshot']['numbered_cards']==384
                and all(node in absent[row['parent']] for node,row in r[-2].items()),
                'newly missing cohort differs from exact 18:15 scheduler evidence')
        r[-1]['later_scheduler_receipt_SHA256']=FLEET[1]
        r[-1]['runtime_budget_guard_SHA256']=NEXT[1]
        return r

    def bind(material,claim_sha):
        ctx,read_program,read,render,dispatch,observe=build(material,claim_sha)
        prior_publish=dispatch.__globals__['publication']
        def publication(context,revision):
            prior_publish(context,revision)
            _,publisher=context[3][-2]['modules']()
            require(publisher['git']('cat-file','blob',revision+':'+NEXT[0])==raw,'published exact budget guard differs')
        return ctx,read_program,read,render,fn(dispatch,publication=publication),observe

    return inherited,load,bind


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
    inherited,load,bind=exact_api()
    require,digest=inherited['require'],inherited['digest']
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and actual old receipt required')
    material=load()
    first,r34,early,_,_,rows,claim=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact four successors changed')
    sha=digest(raw)
    ctx,reader,read,render,dispatch,observe=bind(material,sha)
    if args.check:
        output={}
        for node,row in rows.items():
            source=reader(ctx,row,'3'*32)
            gpu=[row['previous'],f"training80-dcu-standby-{row['request'][:12]}-{node}",'COMPLETED','0:0',str(row['worker_seconds']),
                 node,early['GPU_TRES'],'2000-01-01T00:00:00','2000-01-01T03:00:00']
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                   same_UID=True,exact_job_and_step_cgroup=True)).encode()
            guard=render(ctx,row,gpu,('a'*64,digest(cgroup)),None)
            first['cgroup_fixture'](guard['launch'],row,cgroup,require)
            seconds=inherited['spawn_fixture'](guard['launch'],row,ctx[1][6]['controller.py'])
            output[node]=dict(request=row['new_request'],worker_seconds=seconds,original_GPU=row['previous'],
                              reader_fixture_SHA256=digest(source.encode()),host_fixture_SHA256=digest(guard['launch'].encode()))
        result=dict(nodes=output,state='PREPARED_ONLY',actual_HOST_spawn_positive=4,actual_HOST_spawn_vetoes=8,
                    locked_cgroup_and_CPU_positive=4,locked_cgroup_and_CPU_vetoes=8,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

"""Freeze separate successors for the proved n17 and n10 worker returns."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r43-two-specialist-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r43_two_specialist_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R42=('tools/run_training80_r42_three_2006_guarded_successor_once.py',
     'adc19c6d77ba95fc82951ccab543f337436cb203b8d5ed640f25b0d31bbef2c1')
R34=('tools/run_training80_r34_four_terminal_guarded_refill_once.py',
     'd3f72317a56fc31363a14d2b9c017fa612c9781238e1ca3124fd3cd983e5b4e8')
N17=('tools/run_training80_n17_3301_guarded_refill_once.py',
     '069cab103227559d17c36b8ca3ae3bc12a9dcc610d9591dbb9b82fed1b8126db')
TERMINAL=('training80-r42-two-next-original-read-939a46110239422ba72b61ff0c1e1e1e.json',
          '5fe772aead8638ca4d32e785363900245b7b9b825a8c10e323d43ac662739e3a')
COUNTER=('training80-r34-five-renewal-paired-counters-b44209849726432eb5081818b8fa4002.json',
         'b037278fa0781d4197dde17baa1c065981f74a265bff9404aad4c4197328d489')
SPECS={
 'f13r1n17':('1103606.3377','1103606.3376','e9685e1f281134f396fa6d337de5b31e306227676d8a6df47dead2a44912d2a5'),
 'f13r1n10':('1103606.3379','1103606.3378','41053a419058d3d53a5f8254d3bc986232661fc2132567f2af11400d2032146a'),
}
PARENT='1103606'


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact source or proof path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==expected,'exact source or proof bytes changed')
    return raw


def module(spec):
    relative,sha=spec
    path=ROOT/relative
    pin(path,sha)
    return runpy.run_path(str(path))


def load():
    inherited,r34,n17=module(R42),module(R34),module(N17)
    special,_,_,last,previous,_,legacy_predecessors,legacy,_,_=r34['load']()
    n17_material=n17['inputs']()
    original=n17_material[8]
    require(n17['NODE']=='f13r1n17' and n17['PREVIOUS']=='1103606.3301'
            and original['previous']==n17['PREVIOUS'],'exact n17 long-running original ancestry differs')
    ancestor,budget=module(inherited['R34']),module(inherited['BUDGET'])
    first=budget['module'](budget['FIRST'])
    prior_four,early,n02=first['modules']()
    live=early['load']()
    controller=budget['controller'](live[1][6]['controller.py'])
    packet=json.loads(pin(LOCAL/TERMINAL[0],TERMINAL[1]))
    facts=packet['observed']
    counter=json.loads(pin(LOCAL/COUNTER[0],COUNTER[1]))
    points=counter['result']['nodes']
    require(packet['rc']==0 and packet['replay_allowed'] is False
            and packet['Slurm_mutations']==packet['physical_admissions']==facts['Slurm_mutations']==facts['physical_admissions']==0
            and counter['rc']==0 and counter['result']['GPU_steps_created']==0 and set(points)==set(SPECS)
            and [r['node'] for r in facts['targets']]==list(SPECS),'original two-node terminal scope differs')
    rows,claims,routes={},{},{}
    for fact in facts['targets']:
        node=fact['node'];step,cpu,reply_sha=SPECS[node];v=points[node]
        gpu,zero,files=fact['old_gpu_accounting'],fact['old_cpu_accounting'],fact['files_SHA256']
        is_n17=node==n17['NODE']
        family,request=(n17['STEM'],n17['REQUEST']) if is_n17 else (r34['STEM'],last['targets'][node]['request'])
        wire=json.loads(pin(LOCAL/f'{family}-{node}-{request}-reply.json',reply_sha))['transport']
        out=wire['response']
        utc_cpu=zero[:4]+zero[5:7]+zero[8:]
        approved_cpu=out['fresh_zero_GRES_accounting']
        require(wire['rc']==0 and wire['remote_outcome']=='RESPONSE_OBSERVED' and wire['authority_verified'] is wire['remote_received'] is True
                and out['GPU_srun_attempted'] is True and wire['request']==out['request']==fact['request']==v['request']==request
                and out['node']==node and out['parent']==v['parent']==fact['parent']==PARENT
                and out['probe_step']==cpu==zero[0] and utc_cpu==approved_cpu
                and gpu[:4]==[step,'training80-dcu-standby-'+request[:12]+'-'+node,'COMPLETED','0:0']
                and gpu[5:7]==[node,early['GPU_TRES']] and gpu[7]<gpu[8]
                and fact['numbered_predecessor']==v['previous']==step and fact['controller_rows']==fact['queue_rows']==[]
                and fact['qualified_original_release'] is fact['natural_return'] is True and fact['physical_clear'] is False
                and fact['pid']==v['worker_PID'] and fact['worker_start_epoch']==v['worker_start_epoch'] and fact['cgroup_start_ticks']>0
                and fact['worker_seconds']==v['worker_seconds']==out['receipt']['worker_seconds']
                and fact['returned_epoch']>=fact['worker_start_epoch']+fact['worker_seconds']
                and v['growth_DCU']==8 and v['pair_state']=='PASS' and sorted(fact['pci'])==sorted(v['first'])==sorted(v['second'])
                and files['worker']==v['worker_SHA256'] and files['expected']==v['expected_SHA256'] and files['submission']==v['submit_SHA256'],
                'actual returned worker, paired CPU or exact original launch differs')
        initial=legacy[node]; historical=legacy_predecessors[node]['accounting'][0]
        old_success=n17['PREVIOUS'] if is_n17 else previous[node]['accounting'][0]
        routes[node]=dict(node=node,old_initial_step=initial,old_scope_step=historical,previous=step,another_prior=old_success)
        successor=uuid5(UUID('816a9a8e-7b37-4f47-b6f6-89e7fdd649ae'),f'{node}/{step}/{request}').hex
        seconds=controller['node_seconds'](node,10800,successor)
        end=v['worker_start_epoch']+v['worker_seconds']
        rows[node]=dict(node=node,parent=PARENT,previous=step,request=request,new_request=successor,clearance=utc_cpu,
                        pinned_GPU=gpu,pinned_terminal=files['terminal'],worker_start_epoch=v['worker_start_epoch'],
                        worker_seconds=v['worker_seconds'],planned_end_epoch=end,prepare_epoch=end-900,pci=fact['pci'],
                        worker_SHA256=files['worker'],expected_SHA256=files['expected'],submit_SHA256=files['submission'],
                        cgroup_SHA256=files['cgroup'],previous_worker_pid=fact['pid'],renewal_seconds=seconds)
        claims[node]=dict(prior_GPU=step,prior_CPU=cpu,prior_request=request,new_request=successor,worker_seconds=seconds,
                          prior_route_SHA256=N17[1] if is_n17 else R34[1],old_launch_reply_SHA256=reply_sha,old_worker_files=files,
                          protected_history=[initial,historical,old_success],Slurm_time_cap_seconds=seconds+300)
    ctx=(*live[:5],routes,live[6])
    claim=dict(scope=STEM,state='PREPARED_ONLY_FRESH_PHYSICAL_AND_PRIORITY_REQUIRED',parent=PARENT,targets=claims,
               source_terminal_SHA256=TERMINAL[1],source_counter_SHA256=COUNTER[1],runtime_SHA256=R42[1],
               controller_SHA256=budget['CONTROLLER'][1],launcher=WRAPPER.relative_to(ROOT).as_posix(),
               other_reserved_nodes_excluded=['f12r2n10','f13r1n12','f13r3n13'],GPU_per_node=8,max_invocations_per_node=1,
               ledger_mutations=0,replay_allowed=False)
    return inherited,(ancestor,budget,first,prior_four,special,early,n02,ctx,rows,claim)


def runtime(inherited,material,claim_sha):
    _,_,first,_,owners,_,_,_,rows,_=material
    fn=first['_fn']
    call=fn(inherited['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,reader,read,render,dispatch,observe=call(material,claim_sha)
    original_publish=dispatch.__globals__['publication']
    local=dispatch.__globals__['exact_no_other_claims']
    def publication(c,revision):
        original_publish(c,revision)
        _,publisher=c[3][-2]['modules']()
        for path,sha in (R42,R34,N17):
            require(publisher['git']('cat-file','blob',revision+':'+path)==pin(ROOT/path,sha),'published original specialist source differs')
    def confirm(ordinary,old_plan,row):
        observed=local(ordinary,old_plan,row)
        node=row['node']
        owners['require_no_slots'](ctx[0]['JOURNAL'],node,ctx[5][node]['another_prior'],(0,1,2))
        return observed
    return ctx,reader,read,render,fn(dispatch,publication=publication,exact_no_other_claims=confirm),observe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    p.add_argument('--node',choices=SPECS)
    p.add_argument('--terminal-receipt',nargs=2)
    args=p.parse_args()
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one node and exact original receipt required')
    inherited,material=load()
    _,budget,first,base,_,_,_,_,rows,claim=material
    raw=base['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact two-specialist claim differs')
    sha=digest(raw)
    ctx,source,read,render,dispatch,observe=runtime(inherited,material,sha)
    if args.check:
        nodes={}
        for node,row in rows.items():
            reader=source(ctx,row,'f'*32)
            sample=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                  same_UID=True,exact_job_and_step_cgroup=True)).encode()
            program=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(sample)),None)
            first['cgroup_fixture'](program['launch'],row,sample,require)
            seconds=budget['spawn_fixture'](program['launch'],row,ctx[1][6]['controller.py'])
            nodes[node]=dict(request=row['new_request'],worker_seconds=seconds,reader_SHA256=digest(reader.encode()))
        result=dict(state='PREPARED_ONLY',nodes=nodes,actual_HOST_spawn_positive=2,actual_HOST_spawn_vetoes=4,
                    locked_cgroup_and_CPU_positive=2,locked_cgroup_and_CPU_vetoes=4,SSH=0,Git=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

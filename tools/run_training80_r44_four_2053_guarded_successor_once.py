"""Renew only the four passed original worker releases, one guarded node at a time."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r44-four-2053-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r44_four_2053_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
NEXT=('tools/run_training80_r35_next_three_guarded_successor_once.py',
      'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
R29=('tools/run_training80_r29_other11_warning_guarded_refill_once.py',
     'c3162682176376fcd84099c09372ec3e3010de54c106e04278ed3752e132159b')
ORIGINALS={
 '1103606':('training80-r44-2053-originals-1103606-a56ecb0fba6d4cdfa60e2e5503c326dc.json',
            '10440535172e20ce811d28ef8a76ac2b0067c3fb0f4b56da6055a322e74b02c3',5),
 '1194900':('training80-r44-2053-originals-1194900-8784d04a019242869b75d3ea11db2600.json',
            '11b3cd5bf676fa128475d980da99747f6e3431b3b5b97c7b901d454a9d88ed05',3),
}
R35_COUNTER=('training80-r35-first-four-actual-counter-read-226b38d1a5ca40a0beb4bce80c65fd88/receipt.json',
             'a510da0a17319835cc294f5f4afb12029b093eed2ccf39b67af8903e2be2d340')
R31_COUNTER=('training80-r29-other11-warning-guarded-bf16-v1-f12r3n10-independent-e167232095e341e6a02f9a8e37b3b255.json',
             '8f4cd3c23692094444f097d64dcc6f3b810203612d6621dad685c48523bbc4b8')
NODES={
 'f13r1n02':('1103606.3387','1103606.3386','3552c1337bdc3b9a6074b082f747198037c5b7f18340c7d7a684d7855f52d2ef'),
 'f13r2n08':('1103606.3389','1103606.3388','e9851aac91196981ca828171195ab38d5d2d668177f551e9052bdac87dd7e0c9'),
 'f14r3n04':('1194900.1391','1194900.1390','576e3fcfc8d1b9597c45967a8dd06066d9f215f23eca55becdbff0d868bd1eb8'),
 'f12r3n10':('1194900.1381','1194900.1380','ae844dac9f4e1b0ef54c2534634004813cacc882a0e95b14de525a384b121617'),
}
HELD_WARNED={'f13r1n01':'1103606.3357','f13r2n01':'1103606.3360','f13r2n02':'1103606.3361','f13r3n01':'1194900.1383'}
HELD_OTHER=('f12r2n10','f12r2n11','f13r1n12','f13r2n06','f13r2n19','f13r3n13')
R29_NODE='f12r3n10'
N02='f13r1n02'


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok,why):
    if not ok:
        raise ValueError(why)


def pin(path,sha):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact source or receipt path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==sha,'exact source or receipt bytes differ')
    return raw


def module(spec):
    path=ROOT/spec[0]
    pin(path,spec[1])
    return runpy.run_path(str(path))


def json_input(spec):
    path,sha,*_=spec
    return json.loads(pin(LOCAL/path,sha))


def load():
    budget=module(NEXT)
    first=budget['module'](budget['FIRST'])
    r34,early,n02,ctx,old_rows,_=first['load']()
    other=module(R29)
    ordinary=other['bound_api']()['load']()
    routes=dict(ctx[5],**{R29_NODE:ordinary[5][R29_NODE]})
    context=(*ctx[:5],routes,ctx[6])
    base=budget['controller'](context[1][6]['controller.py'])
    c35=json_input(R35_COUNTER)
    c31=json_input(R31_COUNTER)
    require(c35['rc']==c31['rc']==0 and c35['result']['GPU_steps_created']==c31['observed']['new_scheduling_operations']==0,
            'actual eight-card predecessors differ')
    rows,claims,rejected={}, {}, {}
    for parent,spec in ORIGINALS.items():
        packet=json_input(spec)
        data=packet['observed']
        require(packet['rc']==0 and packet['parent']==parent and packet['replay_allowed'] is False
                and packet['Slurm_mutations']==packet['physical_admissions']==data['Slurm_mutations']==data['physical_admissions']==0
                and data['plans_only'] is data['requires_fresh_independent_device_and_dispatch_checks'] is True
                and len(data['targets'])==spec[2] and packet['selected_nodes']==[v['node'] for v in data['targets']],
                'expected exact parent/read original denominator differs')
        for fact in data['targets']:
            node=fact['node']
            require(fact['parent']==parent and fact['physical_clear'] is False,'read-only scope or parent differs')
            if node not in NODES:
                [cpu]=fact['rejected_CPU']
                require(node in HELD_WARNED and fact['scope']=='WARNED' and fact['files_SHA256']['cgroup'] is None
                        and fact['qualified_for_separate_fresh_physical'] is False and fact['original_natural_return'] is False
                        and fact['controller_numeric']==fact['queue_numeric']==[]
                        and fact['durable_no_send_matches'] is True and fact['rejected_GPU_label_rows']==[]
                        and fact['rejected_GPU_namespace_exists'] is fact['rejected_GPU_intent_exists'] is False
                        and cpu[0]==HELD_WARNED[node] and cpu[2:4]==['COMPLETED','0:0'],'older failed qualification changed')
                rejected[node]=dict(prior_GPU=fact['prior_numbered'],paired_warning_CPU=cpu[0],state='HOLD_ORIGINAL_CGROUP_ABSENT')
                continue
            step,cpu_id,reply_sha=NODES[node]
            independent=node==R29_NODE
            evidence=c31['observed'] if independent else c35['result']['nodes'][node]
            a,b=evidence['first'],evidence['second']
            request=fact['prior_request']
            source=other['STEM'] if independent else first['STEM']
            wire=json_input((f'{source}-{node}-{request}-reply.json',reply_sha))['transport']
            out=wire['response'];gpu=fact['original_accounting'];files=fact['files_SHA256']
            [cpu]=fact['original_CPU']
            old_budget=evidence['worker_budget_seconds' if independent else 'worker_seconds']
            start=evidence['actual_worker_start_epoch' if independent else 'worker_start_epoch']
            actual_cpu=cpu[:4]+cpu[5:7]+cpu[8:]
            require(fact['scope']=='NATURAL' and fact['qualified_for_separate_fresh_physical'] is fact['original_natural_return'] is True
                    and fact['controller_numeric']==fact['queue_numeric']==[]
                    and step==fact['prior_numbered']==evidence['step' if independent else 'previous']
                    and gpu[:4]==[step,'training80-dcu-standby-'+request[:12]+'-'+node,'COMPLETED','0:0']
                    and gpu[5:7]==[node,early['GPU_TRES']] and len(gpu)==9 and gpu[7]<gpu[8]
                    and len(cpu)==9 and cpu[0]==out['probe_step']==cpu_id and cpu[2:4]==['COMPLETED','0:0']
                    and cpu[5:7]==[node,'cpu=1,mem=256M,node=1'] and actual_cpu==out['fresh_zero_GRES_accounting']
                    and wire['rc']==0 and wire['remote_outcome']=='RESPONSE_OBSERVED'
                    and wire['authority_verified'] is wire['remote_received'] is out['GPU_srun_attempted'] is True
                    and wire['request']==out['request']==evidence['request']==request
                    and request==(ordinary[5][node] if independent else old_rows[node])['new_request']
                    and fact['worker_pid']>0 and (independent or fact['worker_pid']==evidence['worker_PID'])
                    and start==fact['worker_start_epoch'] and old_budget==out['receipt']['worker_seconds']==fact['worker_seconds']
                    and int(gpu[4])>=old_budget and fact['returned_epoch']>=start+old_budget
                    and len(a)==len(b)==len(fact['pci'])==8 and set(a)==set(b)==set(fact['pci'])
                    and all(b[p]['calls']>a[p]['calls'] and b[p]['epoch']>a[p]['epoch'] for p in a)
                    and set(files)=={'cgroup','expected','submission','terminal','worker'} and all(re.fullmatch(r'[a-f0-9]{64}',v) for v in files.values())
                    and files['worker']==evidence['worker_SHA256'] and files['expected']==evidence['expected_SHA256']
                    and (files['cgroup']==evidence['worker_cgroup_SHA256'] if independent else files['submission']==evidence['submit_SHA256']),
                    'exact complete original C0, worker files, paired previous CPU or owner authority differs')
            new=uuid5(UUID('abfe70aa-26bd-45cb-94aa-3a14bee1adee'),f'{node}/{step}/{request}').hex
            seconds=base['node_seconds'](node,10800,new)
            rows[node]=dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,pinned_terminal=files['terminal'],
                            clearance=actual_cpu,worker_start_epoch=start,worker_seconds=old_budget,planned_end_epoch=start+old_budget,
                            prepare_epoch=start+old_budget-900,pci=sorted(a),worker_SHA256=files['worker'],expected_SHA256=files['expected'],
                            submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],previous_worker_pid=fact['worker_pid'],renewal_seconds=seconds)
            claims[node]=dict(parent=parent,original_GPU=step,original_CPU=cpu_id,original_request=request,new_request=new,
                              actual_prior_seconds=old_budget,new_worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300,
                              original_worker_files=files,old_launch_reply_SHA256=reply_sha,
                              priority='I01.Kimi-K3 first' if node==N02 else 'existing adoption and active business holds')
    require(set(rows)==set(NODES) and set(rejected)==set(HELD_WARNED) and not set(rows)&set(HELD_OTHER),
            'either enabled or excluded subset differs')
    claim=dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',launcher=WRAPPER.relative_to(ROOT).as_posix(),
               original_parent_receipt_SHA256={p:v[1] for p,v in ORIGINALS.items()},targets=claims,rejected=rejected,other_states_preserved=list(HELD_OTHER),
               original_R29_route_SHA256=R29[1],original_R35_route_SHA256=budget['FIRST'][1],controller_SHA256=budget['CONTROLLER'][1],
               GPU_per_node=8,max_invocations_per_node=1,max_concurrent_send=1,ledger_mutations=0,replay_allowed=False)
    return budget,first,r34,early,n02,context,rows,claim


def runtime(material,sha):
    budget,first,r34,early,n02,context,rows,claim=material
    fn=first['_fn']
    extended=dict(early)
    def read_source(ctx,row,nonce):
        source=early['read_program'](ctx,row,nonce)
        return early['unique'](source,'    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,',
                               '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'
                               '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,')
    extended['read_program']=read_source
    guarded=fn(budget['api'],STEM=STEM,G34=R29,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,source,read,render,dispatch,observe=guarded((first,r34,extended,n02,context,rows,claim),sha)
    existing=dispatch.__globals__['publication']
    def publication(c,revision):
        existing(c,revision)
        _,publisher=c[3][-2]['modules']()
        for path,pin_sha in (NEXT,budget['FIRST']):
            require(publisher['git']('cat-file','blob',revision+':'+path)==pin(ROOT/path,pin_sha),
                    'published actual old role or controller budget guard differs')
    return ctx,source,read,render,fn(dispatch,publication=publication),observe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--check',action='store_true')
    m.add_argument('--observe-predecessor')
    m.add_argument('--publication')
    m.add_argument('--observe-original',action='store_true')
    p.add_argument('--node',choices=NODES)
    p.add_argument('--terminal-receipt',nargs=2)
    args=p.parse_args()
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and fresh terminal receipt required')
    material=load()
    budget,first,r34,_,n02,_,rows,claim=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact guarded four-node plan differs')
    sha=digest(raw)
    ctx,source,read,render,dispatch,observe=runtime(material,sha)
    if args.check:
        status={}
        for node,row in rows.items():
            read_fixture=source(ctx,row,'1'*32)
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],
                                   worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
            host=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)
            first['cgroup_fixture'](host['launch'],row,cgroup,require)
            seconds=budget['spawn_fixture'](host['launch'],row,ctx[1][6]['controller.py'])
            status[node]=dict(request=row['new_request'],worker_seconds=seconds,fixture_SHA256=digest(read_fixture.encode()))
        n02['fixture_holds']()
        result=dict(state='PREPARED_ONLY',nodes=status,held=claim['rejected'],physical_admissions=0,
                    actual_HOST_spawn_positive=4,actual_HOST_spawn_vetoes=8,locked_cgroup_and_CPU_positive=4,locked_cgroup_and_CPU_vetoes=8,
                    SSH=0,Git=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

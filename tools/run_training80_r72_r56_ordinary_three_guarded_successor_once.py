"""Renew the three R56 ordinary owners after independently verified natural returns."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r72-r56-ordinary-three-guarded-bf16-v1'
WRAPPER = ROOT/'tools/run_training80_r72_r56_ordinary_three_guarded_successor_once.py'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R56 = ('tools/run_training80_r56_four_0215_guarded_successor_once.py',
       'a7ac9f59a288b6d475eb6a8a314305cf2b9ac07f55ef39fe9afb9998399c9716')
R56_CLAIM = ('coordination/tasks/training80-r56-four-0215-guarded-bf16-v1.json',
             '6569d623e58bad6d767cbf9e08a07ef2c84f282c8fc301c7671af203ac456bcb')
LOCAL_R56 = 'local-only/training80-r56-four-0215-guarded-bf16-v1'
CUSTODY_ROOT = 'local-only/training80-r56-three-original-physical-custody-readback-69bfca7295f049c88b98d146da3cf5d4'
CUSTODY = (CUSTODY_ROOT+'/receipt.json','246f3cb363d52fb0792f353ee68b04bdaf8007d54aea963fef9bd4e1d243b1b7')
CUSTODY_INTENT = (CUSTODY_ROOT+'/intent.json','c8ec5ba881fecdfdd5dc0ef86e36485abc16b404046424ffd7757e6ebc9ed252')
CUSTODY_SEND = (CUSTODY_ROOT+'/send.json','3b0d17f61b7bf101c22c895231b02e436658fdd5da705a63359533367c663448')
COUNTERS = (LOCAL_R56+'-actual-batch-counters-e1d4972c126c4a4297a4aeb0d2feaf99/receipt.json',
            '8b52ffae9568cf2afd0ffad6af337f0e7b4e0e947406b0b5b03b34a3e45c0207')
TERMINALS = {
    'f13r1n13': (LOCAL_R56+'-f13r1n13-independent-b9642c7385274c45b4862f82e04db4c7.json',
                 '95ce047bf1b3b0224ec682729c2eceb1cef1a06e32145bb38ba3cd9cee02367b'),
    'f13r1n03': (LOCAL_R56+'-f13r1n03-independent-f7ab5cb54225467eb2d397914ffaccf6.json',
                 '0c3622deb7747457b254dfc02366c1770a9f125a536184893150f9f2187bb1b2'),
    'f14r3n03': (LOCAL_R56+'-f14r3n03-independent-83791ed881a245fd8f028a652cda59fb.json',
                 '5effcd6782604efab2699381b44f46dce11cbfc6a1b461f0c75a6f5624746e30'),
}
ORDER = ('f13r1n13','f13r1n03','f14r3n03')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pinned(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact source path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'exact source bytes differ')
    return raw


def seven_columns(row):
    return row[:4]+row[5:7]+row[8:]


def load():
    pinned(R56)
    base=runpy.run_path(str(ROOT/R56[0]))
    previous,_,material=base['entry']()[0]()
    budget,first,r34,early,n02,context,originals,shadow,old_claim=material
    require(json.loads(pinned(R56_CLAIM))==old_claim and set(ORDER)==set(base['ORDER'])-{'f12r1n18'}
            and all(old_claim['targets'][n]['queue_ownership']=='ADOPTED_ORDINARY58_EXACT_ANCESTOR' for n in ORDER),
            'exact R56 ordinary owners differ')
    intent,send,receipt=(json.loads(pinned(spec)) for spec in (CUSTODY_INTENT,CUSTODY_SEND,CUSTODY))
    evidence=receipt['response']
    require(receipt['rc']==0 and receipt['remote_outcome']=='RESPONSE_OBSERVED'
            and receipt['authority_verified'] is receipt['remote_received'] is receipt['payload_flushed'] is True
            and receipt['request']==intent['request']==send['request']==evidence['request']
            and send['payload_sha256']==receipt['payload_sha256']
            and evidence['Slurm_mutations']==evidence['physical_admissions']==0
            and [item['node'] for item in evidence['targets']]==list(ORDER),
            'independent original owner read phase differs')
    counter=json.loads(pinned(COUNTERS))
    controller=budget['controller'](context[1][6]['controller.py'])
    rows,targets={},{}
    for item in evidence['targets']:
        node=item['node']
        original=originals[node]
        parent=original['parent']
        request=original['new_request']
        seconds=original['renewal_seconds']
        packet=json.loads(pinned(TERMINALS[node]))
        observed=packet['observed']
        terminal,terminal_sha=observed['worker_terminal']
        outcome,_=observed['original_guard_outcome']
        [point]=[part['receipt']['nodes'][node] for part in counter['receipt_by_batch'] if part.get('rc')==0
                 and node in (part.get('receipt') or {}).get('nodes',{})]
        gpu,cpu=item['gpu_accounting'],item['cpu_accounting']
        require(item['original_request']==request==point['request']==outcome['request']==observed['request']
                and item['original_custody_verified'] is True and item['physical_clearance'] is False
                and not item['current_controller_steps'] and not item['current_queue_steps']
                and len(gpu)==len(cpu)==9 and gpu[0]==item['old_GPU']==observed['step']==point['previous']
                and gpu[1:4]==['training80-dcu-standby-'+request[:12]+'-'+node,'COMPLETED','0:0']
                and seven_columns(gpu)==observed['accounting'] and int(gpu[4])>=seconds and gpu[7]<gpu[8]
                and gpu[5:7]==[node,early['GPU_TRES']]
                and cpu[:4]==[item['old_CPU'],'training80-r25-fresh-'+request[:12],'COMPLETED','0:0']
                and cpu[5:7]==[node,'cpu=1,mem=256M,node=1'] and cpu[7]<=cpu[8] and cpu[8]!='Unknown'
                and seven_columns(cpu)==outcome['fresh_zero_GRES_accounting'] and outcome['probe_step']==cpu[0]
                and packet['rc']==0 and packet['replay_allowed'] is False and observed['state']=='EXACT_TERMINAL_ACCOUNTING'
                and observed['new_scheduling_operations']==0 and observed['parent']==point['parent']==parent
                and outcome['GPU_srun_attempted'] is outcome['mutex_acquired'] is True
                and terminal['node']==node and terminal['step']==gpu[0] and terminal['state']=='WORKER_RETURNED'
                and terminal['epoch']>=point['worker_start_epoch']+seconds and point['worker_seconds']==seconds
                and len(point['first'])==point['growth_DCU']==8 and set(point['first'])==set(point['second'])==set(point['count_deltas'])
                and all(point['second'][pci]['calls']-v['calls']==point['count_deltas'][pci]>0
                        and point['second'][pci]['epoch']>v['epoch'] for pci,v in point['first'].items())
                and item['original_PID']==point['worker_PID']>0
                and type(item['original_start_ticks']) is int and item['original_start_ticks']>0
                and item['terminal_SHA256']==terminal_sha and all(item[key]==point[key] for key in ('expected_SHA256','worker_SHA256','submit_SHA256'))
                and all(re.fullmatch(r'[a-f0-9]{64}',item[key]) for key in
                        ('cgroup_SHA256','terminal_SHA256','worker_SHA256','expected_SHA256','submit_SHA256')),
                f'{node}: original worker, terminal, paired CPU or durable cgroup custody differs')
        new=uuid5(UUID('fb0c1139-90d9-496e-8d7b-2e77a5dee2cf'),f"{node}/{gpu[0]}/{request}").hex
        duration=controller['node_seconds'](node,10800,new)
        require(new!=request and 10500<=duration<=10800,'new ordinary request or worker budget differs')
        rows[node]=dict(node=node,parent=parent,previous=gpu[0],request=request,new_request=new,pinned_GPU=gpu,
                        pinned_terminal=terminal_sha,clearance=seven_columns(cpu),worker_start_epoch=point['worker_start_epoch'],
                        worker_seconds=seconds,planned_end_epoch=point['planned_expiry_epoch'],prepare_epoch=point['prepare_epoch'],
                        pci=sorted(point['first']),worker_SHA256=item['worker_SHA256'],expected_SHA256=item['expected_SHA256'],
                        submit_SHA256=item['submit_SHA256'],cgroup_SHA256=item['cgroup_SHA256'],
                        previous_worker_pid=item['original_PID'],original_start_ticks=item['original_start_ticks'],renewal_seconds=duration)
        targets[node]=dict(parent=parent,original_GPU=gpu[0],original_CPU=cpu[0],original_request=request,actual_prior_seconds=seconds,
                           new_request=new,new_worker_seconds=duration,Slurm_time_cap_seconds=duration+300,queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',
                           original_worker_files={k:item[k+'_SHA256'] for k in ('cgroup','terminal','worker','expected','submit')},
                           original_start_ticks=item['original_start_ticks'])
    require(len({r['new_request'] for r in rows.values()})==len(ORDER),'duplicate exact new request')
    claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),state='PREPARED_ONLY_EXACT_FRESH_ORDINARY_ADMISSION_REQUIRED',
               original_source_SHA256=R56[1],original_claim_SHA256=R56_CLAIM[1],original_counter_SHA256=COUNTERS[1],
               original_terminal_SHA256={n:TERMINALS[n][1] for n in ORDER},original_custody_SHA256=CUSTODY[1],
               targets=targets,parent_serial_order=list(ORDER),queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',GPU_per_node=8,ready_GPU_count=0,
               priority_scopes_preserved=sorted(base['FENCES']|{'f12r1n18'}),max_concurrent_send=1,
               max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False)
    context=(*context[:5],{n:context[5][n] for n in ORDER},context[6])
    return previous,(budget,first,r34,early,n02,context,rows,shadow,claim)


def runtime(previous,material,claim_sha):
    clone=material[1]['_fn']
    inherit=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    context,source,read,render,dispatch,observe=inherit(material,claim_sha)
    publish=dispatch.__globals__['publication']

    def publication(actual,revision):
        publish(actual,revision)
        _,publisher=actual[3][-2]['modules']()
        for spec in (R56,R56_CLAIM):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pinned(spec),'original ordinary publication differs')

    def guarded_render(ctx,row,account,pins,revision):
        require(account==row['pinned_GPU'] and pins==(row['pinned_terminal'],row['cgroup_SHA256']),
                'current predecessor receipt differs from the pinned original custody')
        packet=render(ctx,row,account,pins,revision)
        text=packet['launch']
        prior="prior_cgroup['worker_start_ticks']>0"
        require(text.count(prior)==1,'existing locked cgroup comparison differs')
        packet['launch']=text.replace(prior,f"prior_cgroup['worker_start_ticks']=={row['original_start_ticks']}",1)
        compile(packet['launch'],'<locked ordinary successor original start ticks>','exec')
        return packet

    return context,source,read,guarded_render,clone(dispatch,render=guarded_render,publication=publication),clone(observe,render=guarded_render)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--show-run-order',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=ORDER)
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args=parser.parse_args()
    require((args.check or args.show_run_order or args.node) and bool(args.publication)==bool(args.terminal_receipt),
            'one exact node and a fresh predecessor receipt required')
    previous,material=load()
    raw=CLAIM.read_bytes()
    budget,first,_,_,n02,_,rows,_,claim=material
    require(json.loads(raw)==claim,'ordinary claim differs')
    pin=sha256(raw).hexdigest()
    if args.show_run_order:
        result=dict(state='PREPARED_ONLY',ordered=[{k:rows[n][k] for k in ('node','previous','new_request','renewal_seconds')} for n in ORDER],
                    max_concurrent_send=1,replay_allowed=False)
    else:
        context,source,read,render,dispatch,observe=runtime(previous,material,pin)
        if args.check:
            tested={}
            for n,r in rows.items():
                original=render(context,r,r['pinned_GPU'],(r['pinned_terminal'],r['cgroup_SHA256']),None)
                require(f"prior_cgroup['worker_start_ticks']=={r['original_start_ticks']}" in original['launch'],
                        'fixed production worker start ticks not rendered')
                try:
                    render(context,r,r['pinned_GPU'],(r['pinned_terminal'],'0'*64),None)
                except ValueError:
                    pass
                else:
                    raise ValueError('wrong original cgroup hash accepted')
                cgroup=json.dumps(dict(node=n,step=r['previous'],worker_pid=r['previous_worker_pid'],
                    worker_start_ticks=r['original_start_ticks'],same_UID=True,exact_job_and_step_cgroup=True)).encode()
                mock=dict(r,cgroup_SHA256=sha256(cgroup).hexdigest())
                rendered=render(context,mock,r['pinned_GPU'],(r['pinned_terminal'],mock['cgroup_SHA256']),None)
                first['cgroup_fixture'](rendered['launch'],mock,cgroup,require)
                seconds=budget['spawn_fixture'](rendered['launch'],mock,context[1][6]['controller.py'])
                tested[n]=dict(request=r['new_request'],worker_seconds=seconds,read_SHA256=sha256(source(context,r,'1'*32).encode()).hexdigest())
            n02['fixture_holds']()
            result=dict(state='PREPARED_ONLY',targets=tested,locked_cgroup_CPU_positive=3,locked_cgroup_CPU_vetoes=6,
                        bound_cgroup_hash_vetoes=3,HOST_spawn_positive=3,HOST_spawn_vetoes=6,SSH=0,Git=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            result=read(context,rows[args.node],args.observe_predecessor)
        elif args.observe_original:
            result=observe(context,rows[args.node])
        else:
            result=dispatch(context,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=pin),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

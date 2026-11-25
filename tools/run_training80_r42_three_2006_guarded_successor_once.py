"""Prepare separate guarded successors for the three proved returns at 20:06–20:10."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r42-2006-three-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r42_three_2006_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R34=('tools/run_training80_r34_three_warning_guarded_refill_once.py',
     '770fe7c9706a1a4e218de4f5c3340e8b1b6090bb9c3622d5eb3088a7a8be829a')
BUDGET=('tools/run_training80_r35_next_three_guarded_successor_once.py',
        'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
TERMINAL=('training80-r42-2006-three-original-read-bbbfe606b23142168ed8e57df71eb7e0.json',
          'e0c68f4eb3cf54ccddb5dd6b5390aaa2dfe3d12fa376825c733f5e2b4cbf8ecb')
COUNTER=('training80-r34-three-warning-paired-counters-90fc1eecf4f647b88c148e2ea6154182.json',
         '7f6d1982cc6a536424f23df05e961c701b9c0d721116639b4b9d87993d49e04a')
SPECS={
 'f13r4n05':('1103606.3371','1103606.3370','221b24f529ed117608b0231b8383b0f6d85003bb0d2085dfe158d18a91152ea5'),
 'f13r2n07':('1103606.3375','1103606.3374','f401b708ef8f4ac16ebddfe1e47cbfe8189731cd8b9a2a33a340366f06ee4966'),
 'f13r2n03':('1103606.3373','1103606.3372','09ea433e7cac7d0570c5ea009c3396706b37a4365b6af1d301cd2d0519080fa5'),
}
ORDINARY='f13r4n05'
T07='f13r2n07'
PARENT='1103606'


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(path,expected):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'exact proof path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==expected,'exact proof bytes differ')
    return raw


def module(spec):
    file,sha=spec
    path=ROOT/file
    pin(path,sha)
    return runpy.run_path(str(path))


def seven(row):
    return row[:4]+row[5:7]+row[8:]


def load():
    ancestor,budget=module(R34),module(BUDGET)
    first=budget['module'](budget['FIRST'])
    r34,early,n02=first['modules']()
    selected=ancestor['originals']()
    _,original,old_ordinary=ancestor['ordinary'](selected,'0'*64)
    specialist=ancestor['specialist']('f13r2n03',selected)
    r27,_,_,_,previous,_,legacy=specialist[:7]
    controller=budget['controller'](original[1][6]['controller.py'])
    wrapper=json.loads(pin(LOCAL/TERMINAL[0],TERMINAL[1]))
    terminal=wrapper['observed']
    counters=json.loads(pin(LOCAL/COUNTER[0],COUNTER[1]))
    c=counters['result']['nodes']
    require(wrapper['rc']==0 and wrapper['replay_allowed'] is False
            and wrapper['Slurm_mutations']==wrapper['physical_admissions']==terminal['Slurm_mutations']==terminal['physical_admissions']==0
            and counters['rc']==0 and counters['result']['GPU_steps_created']==0 and set(c)==set(SPECS)
            and [t['node'] for t in terminal['targets']]==list(SPECS),'exact terminal and paired counter manifests differ')
    rows,claims,ancestry={},{},{ORDINARY:old_ordinary}
    for fact in terminal['targets']:
        n=fact['node'];step,cpu,reply_sha=SPECS[n];v=c[n];old_request=v['request'];files=fact['files_SHA256']
        old=json.loads(pin(LOCAL/f"{ancestor['STEM']}-{n}-{old_request}-reply.json",reply_sha))['transport']
        out=old['response']; gpu=fact['old_gpu_accounting']; host_cpu=fact['old_cpu_accounting']
        require(old['rc']==0 and old['remote_outcome']=='RESPONSE_OBSERVED' and old['authority_verified'] is old['remote_received'] is True
                and old_request==old['request']==out['request']==fact['request']==selected[n]['request']
                and out['GPU_srun_attempted'] is True and out['node']==n and out['parent']==fact['parent']==PARENT
                and gpu[:4]==[step,'training80-dcu-standby-'+old_request[:12]+'-'+n,'COMPLETED','0:0']
                and gpu[5:7]==[n,early['GPU_TRES']] and gpu[7]<gpu[8]
                and out['probe_step']==cpu==host_cpu[0] and seven(host_cpu)==out['fresh_zero_GRES_accounting']
                and fact['numbered_predecessor']==v['previous']==step and fact['controller_rows']==fact['queue_rows']==[]
                and fact['qualified_original_release'] is fact['natural_return'] is True and fact['physical_clear'] is False
                and fact['pid']==v['worker_PID'] and fact['cgroup_start_ticks']>0 and fact['worker_start_epoch']==v['worker_start_epoch']
                and fact['worker_seconds']==v['worker_seconds']==out['receipt']['worker_seconds']
                and fact['returned_epoch']>=fact['worker_start_epoch']+fact['worker_seconds']
                and v['growth_DCU']==8 and v['pair_state']=='PASS' and sorted(fact['pci'])==sorted(v['first'])==sorted(v['second'])
                and files['worker']==v['worker_SHA256'] and files['expected']==v['expected_SHA256'] and files['submission']==v['submit_SHA256'],
                'natural return/CPU/history or exact old worker identity differs')
        if n==ORDINARY:
            require(old_ordinary['new_request']==old_request,'ordinary linked adopted route differs')
        else:
            ancestry[n]=dict(node=n,old_scope_step=selected[n]['previous'][0],old_initial_step=legacy[n],previous=step,
                             prior_request=old_request)
        request=uuid5(UUID('f4663219-373d-4242-ab71-1f6b25090fbc'),f'{n}/{step}/{old_request}').hex
        seconds=controller['node_seconds'](n,10800,request)
        end=fact['worker_start_epoch']+fact['worker_seconds']
        rows[n]=dict(node=n,parent=PARENT,previous=step,request=old_request,new_request=request,clearance=seven(host_cpu),
                     pinned_GPU=gpu,pinned_terminal=files['terminal'],worker_start_epoch=fact['worker_start_epoch'],
                     worker_seconds=fact['worker_seconds'],planned_end_epoch=end,prepare_epoch=end-900,
                     pci=fact['pci'],worker_SHA256=files['worker'],expected_SHA256=files['expected'],submit_SHA256=files['submission'],
                     cgroup_SHA256=files['cgroup'],previous_worker_pid=fact['pid'],renewal_seconds=seconds)
        claims[n]=dict(prior_GPU=step,prior_CPU=cpu,prior_request=old_request,new_request=request,worker_seconds=seconds,
                       prior_route='ORDINARY_ADOPTED' if n==ORDINARY else 'T07_ROLE_PRIORITY' if n==T07 else 'PROTECTED_SPECIALIST',
                       source_launch_reply_SHA256=reply_sha,prior_worker_files=files,Slurm_time_cap_seconds=seconds+300)
    context=(*original[:5],ancestry,original[6])
    claim=dict(scope=STEM,state='PREPARED_ONLY_FRESH_DEVICE_AND_PRIORITY_REQUIRED',parent=PARENT,targets=claims,
               original_terminal_SHA256=TERMINAL[1],original_counter_SHA256=COUNTER[1],original_route_SHA256=R34[1],
               controller_SHA256=budget['CONTROLLER'][1],GPU_per_node=8,max_invocations_per_node=1,replay_allowed=False,ledger_mutations=0,
               T07_priority_always_wins=True,launcher=WRAPPER.relative_to(ROOT).as_posix())
    return ancestor,budget,first,r34,r27,early,n02,context,rows,claim


def runtime(material,claim_sha):
    ancestor,budget,first,r34,r27,early,n02,context,rows,claim=material
    fn=first['_fn']
    early_new=dict(early)
    def terminal_reader(ctx,row,nonce):
        source=early['read_program'](ctx,row,nonce)
        return early['unique'](source,'    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,',
                               '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'
                               '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,')
    def authority(ordinary,plan,route):
        node=route['node']
        if node==ORDINARY:
            return early['exact_no_other_claims'](ordinary,plan,route)
        base,config=context[:2]
        require(node not in base['live_authority'](config) and node not in base['local_holds'](),
                'existing ordinary, T07 or other priority hold protects specialist node')
        r27['require_no_slots'](base['JOURNAL'],node,route['old_initial_step'],(1,2))
        r27['require_no_slots'](base['JOURNAL'],node,route['old_scope_step'],(0,1,2))
        r27['require_no_slots'](base['JOURNAL'],node,route['previous'],(0,1,2))
        return []
    early_new.update(read_program=terminal_reader,exact_no_other_claims=authority)
    guarded=fn(budget['api'],STEM=STEM,G34=R34,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,reader,read,render,dispatch,observe=guarded((first,r34,early_new,n02,context,rows,claim),claim_sha)
    verify=dispatch.__globals__['publication']
    ownership_file=ROOT/ancestor['SPECIAL']
    def publication(c,revision):
        verify(c,revision)
        _,publisher=c[3][-2]['modules']()
        for file,sha in ((ROOT/BUDGET[0],BUDGET[1]),(ownership_file,ancestor['SPECIAL_SHA'])):
            relative=file.relative_to(ROOT).as_posix()
            require(publisher['git']('cat-file','blob',revision+':'+relative)==pin(file,sha),'published actual routing or budget source differs')
    return ctx,reader,read,render,fn(dispatch,publication=publication),observe


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
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one node and actual original receipt required')
    material=load()
    _,budget,first,r34,_,early,_,_,rows,claim=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact three-node successor differs')
    sha=digest(raw)
    ctx,source,read,render,dispatch,observe=runtime(material,sha)
    if args.check:
        out={}
        for node,row in rows.items():
            reader=source(ctx,row,'d'*32)
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                   same_UID=True,exact_job_and_step_cgroup=True)).encode()
            prog=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)
            first['cgroup_fixture'](prog['launch'],row,cgroup,require)
            seconds=budget['spawn_fixture'](prog['launch'],row,ctx[1][6]['controller.py'])
            out[node]=dict(request=row['new_request'],worker_seconds=seconds,read_fixture_SHA256=digest(reader.encode()))
        result=dict(state='PREPARED_ONLY',nodes=out,actual_HOST_spawn_positive=3,actual_HOST_spawn_vetoes=6,
                    locked_cgroup_and_CPU_positive=3,locked_cgroup_and_CPU_vetoes=6,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

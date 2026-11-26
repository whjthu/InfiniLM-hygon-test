"""Prepare the seventeen closed R84 workers behind the ordinary one-send guard."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import time
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'local-only'
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r84-seventeen-0930-guarded-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
READER = ('local-only/read_training80_r84_0930_ordinary_frontiers_once.py',
          'aa9da5730c1b65bbbc8e62815e149625cea3b5e654c215f6335ac2a7df48687a')
ORIGINAL = ('local-only/training80-r84-0930-ordinary-frontiers-e91e20edf2b04fe7b92dc387140ba825.json',
            '666d8b7c31c595cd7df7c4dba5de47ef58a2bcb75cb234875c13444c5de7b776')
R70 = ('tools/run_training80_r70_n14_eagain_guarded_successor_once.py',
       'd23cf3ed638c93040793216a1767ad74df02893eb90319dde9a58b5efd393128')
R72 = ('tools/run_training80_r72_r56_ordinary_three_guarded_successor_once.py',
       'de62a5f6f6fe8c9de560e1b87d9f020fca6c2456aa2900958a23de97d3cb053c')
R74 = ('tools/run_training80_r74_r57_first_four_guarded_successor_once.py',
       '2897293fbd122b40e29a15ab8a0afa01cf543d473390243049b4930f6c7328a6')
R76 = ('tools/run_training80_r76_r58_r59_ten_guarded_successor_once.py',
       '45d1cf61027eb9d2317ce7b4c7cf58a7a2a86014fe1dad3888e5d4896f7e14dd')
EMPTY = sha256(b'').hexdigest()
ID = re.compile('[a-f0-9]{32}').fullmatch
REV = re.compile('[a-f0-9]{40}').fullmatch
DIGEST = re.compile('[a-f0-9]{64}').fullmatch


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def raw(path):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path, 'exact canonical receipt required')
    with path.open('rb') as stream:
        data = stream.read(262145)
    require(0 < len(data) <= 262144, 'bounded receipt required')
    return data


def pin(spec):
    data = raw(ROOT/spec[0])
    require(sha256(data).hexdigest()==spec[1], 'exact immutable evidence differs')
    return data


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def proof_row(item, old):
    node,parent,step,request = (old[key] for key in ('node','parent','previous','request'))
    files = item['files_SHA256']
    gpu = item['original_accounting']
    cpu, = item['original_CPU']
    start,seconds = old['worker_start_epoch'],old['worker_seconds']
    require(item['node']==node and item['parent']==parent and item['prior_numbered']==step
            and item['prior_request']==request and item['scope']=='NATURAL'
            and item['qualified_for_separate_fresh_physical'] is item['original_natural_return'] is True
            and item['physical_clear'] is False and item['controller_numeric']==item['queue_numeric']==[]
            and len(gpu)==len(cpu)==9 and gpu[:4]==[step,f'training80-dcu-standby-{request[:12]}-{node}','COMPLETED','0:0']
            and gpu[5:7]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1'] and gpu[7]<gpu[8] and int(gpu[4])>=seconds
            and cpu[:4]+cpu[5:7]+cpu[8:]==old['old_CPU'] and cpu[7]<=cpu[8]
            and set(files)=={'cgroup','expected','submission','terminal','worker'}
            and all(DIGEST(value or '') for value in files.values())
            and files['worker']==old['worker_SHA256'] and files['expected']==old['expected_SHA256']
            and files['cgroup']==old['cgroup_SHA256'] and len(set(item['pci']))==8 and item['pci']==old['pci']
            and item['worker_pid']>0 and item['worker_start_epoch']==start and item['worker_seconds']==seconds
            and item['returned_epoch']>=start+seconds, 'qualified predecessor identity, paired CPU or return differs')
    return gpu,files,item['worker_pid']


def load():
    reader = module(READER)
    core,base,groups,categories,_ = reader['load']()
    receipt = json.loads(pin(ORIGINAL))
    outcome = receipt['observed']
    nonce = outcome['read_id']
    require(ID(nonce) and receipt['rc']==0 and receipt['stderr_SHA256']==EMPTY
            and receipt['physical_admissions']==receipt['Slurm_mutations']==outcome['physical_admissions']==outcome['Slurm_mutations']==0
            and receipt['replay_allowed'] is False and set(outcome['parents'])==set(groups)
            and receipt['source_SHA256']==sha256(reader['render'](core,base,groups,nonce).encode()).hexdigest(),
            'original whole-group read provenance differs')
    cohort = module(R76)
    cohort_load, cohort_runtime,_ = cohort['entry']()
    previous,_,template = cohort_load()
    budget,first,r34,early,n02,context,_,_,_ = template
    owners = (module(R70)['load']()[1], module(R72)['load']()[1], module(R74)['entry']()[0]()[2], template)
    routes = {}
    accepted = {}
    for material in owners:
        for node,old in material[6].items():
            require(node not in routes, 'old ordinary owners overlap')
            routes[node] = material[5][5][node]
            accepted[node] = old
    controller = budget['controller'](context[1][6]['controller.py'])
    rows,targets,feed = {},{},{}
    for parent,earlier in groups.items():
        branch = outcome['parents'][parent]['response']
        require(branch['physical_admissions']==branch['Slurm_mutations']==0 and branch['read_id']==nonce
                and branch['plans_only'] is branch['requires_fresh_independent_device_and_dispatch_checks'] is True,
                'exact read-only parent response differs')
        for old,item in zip(earlier,branch['targets'],strict=True):
            node,step,request = (old[key] for key in ('node','previous','request'))
            source = accepted[node]
            require(node not in rows and source['parent']==parent==old['parent']==routes[node]['parent']
                    and source['new_request']==request and source['renewal_seconds']==old['worker_seconds']
                    and categories[node]=='EXACT_ORDINARY_ORIGINAL_READ','accepted original route differs')
            gpu,files,pid = proof_row(item,old)
            new = uuid5(UUID('1bdebfc6-f20c-4af6-aac7-e49340c2df02'),f'{node}/{step}/{request}').hex
            seconds = controller['node_seconds'](node,10800,new)
            require(new!=request and 10500<=seconds<=10800,'new exact worker identity or budget differs')
            rows[node] = dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'],clearance=old['old_CPU'],worker_start_epoch=old['worker_start_epoch'],
                worker_seconds=old['worker_seconds'],planned_end_epoch=old['planned_end_epoch'],
                prepare_epoch=old['planned_end_epoch']-900,pci=old['pci'],worker_SHA256=files['worker'],
                expected_SHA256=files['expected'],submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],
                previous_worker_pid=pid,renewal_seconds=seconds)
            feed[node] = old
            targets[node] = dict(parent=parent,original_GPU=step,original_CPU=old['old_CPU'][0],original_request=request,
                new_request=new,actual_prior_seconds=old['worker_seconds'],new_worker_seconds=seconds,
                Slurm_time_cap_seconds=seconds+300,original_worker_files=files,
                queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR')
    require(len(rows)==len({row['new_request'] for row in rows.values()})==17,'exact seventeen IDs differ')
    order = sorted(rows,key=lambda node:rows[node]['planned_end_epoch'])
    context = (*context[:5],{n:routes[n] for n in rows},context[6])
    claim = dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
        launcher=WRAPPER.relative_to(ROOT).as_posix(),original_aggregate_receipt_SHA256=ORIGINAL[1],
        original_local_reader_SHA256=READER[1],source_pins=dict((R70,R72,R74,R76)),targets=targets,
        parent_serial_order=order,ready_GPU_count=0,GPU_per_node=8,max_invocations_per_node=1,
        max_concurrent_send=1,max_nodes_per_fresh_parent_read=4,ledger_mutations=0,replay_allowed=False)
    material = (budget,first,r34,early,n02,context,rows,{},claim)
    return reader,core,base,feed,previous,cohort_runtime,material


def group_path(parent,nonce):
    require(ID(nonce) and int(nonce,16)>0,'a new exact read ID is required')
    return LOCAL/f'{STEM}-{parent}-group-{nonce}.json'


def group_read(reader,core,base,feed,members,nonce,revision):
    selected = [feed[node] for node in members]
    parent, = {r['parent'] for r in selected}
    require(1<=len(selected)<=4 and len(set(members))==len(selected) and REV(revision)
            and nonce not in {r['request'] for r in selected},'a bounded one-parent fresh group is required')
    holds = base['module'](base['R29'])['bound_api']()['load']()[0]['local_holds']()
    require(set(members).isdisjoint(holds),'current priority protection overlaps this group')
    source = reader['render'](core,base,{parent:selected},nonce)
    path = group_path(parent,nonce)
    with path.open('x',encoding='utf8') as stream:
        begin = time.monotonic()
        anchor = time.time()-begin
        result = reader['client'](base,{parent:selected},nonce,source)
        end = time.monotonic()
        record = dict(kind='R84_GROUP',publication=revision,read_id=nonce,parent=parent,members=members,
            received_monotonic=end,anchor=anchor,RTT=end-begin,reader_receipt=result['receipt'],
            reader_receipt_SHA256=result['SHA256'],replay_allowed=False)
        json.dump(record,stream,sort_keys=True)
    return dict(result,group_receipt=path.name,group_receipt_SHA256=sha256(raw(path)).hexdigest())


def dispatch_group(send,ctx,rows,members,nonce,digest,revision):
    require(1<=len(members)<=4 and len(set(members))==len(members) and set(members)<=set(rows)
            and len({rows[n]['parent'] for n in members})==1 and DIGEST(digest) and REV(revision),
            'exact one-parent send order required')
    parent = rows[members[0]]['parent']
    path = group_path(parent,nonce)
    proof = json.loads(pin((path.relative_to(ROOT).as_posix(),digest)))
    require(proof['members']==members and proof['publication']==revision,'send order or publication differs from actual fresh group')
    results = {}
    for node in members:
        try:
            require_fresh_group(proof,revision,time.monotonic(),time.time())
        except ValueError:
            return dict(results=results,remaining=[n for n in members if n not in results],state='REFRESH_GROUP_READ')
        try:
            result = send(ctx,rows[node],revision,(nonce,digest))
        except (ValueError, OSError, KeyError, TypeError) as error:
            return dict(results=results,current_node=node,detail_class=type(error).__name__,
                        unattempted_tail=members[len(results)+1:],state='CHECK_CURRENT_NODE_RECEIPTS_BEFORE_ANY_RETRY')
        results[node] = result
        if result['state'] not in {'GPU_NOT_SENT','SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING'}:
            return dict(results=results,current_node=node,unattempted_tail=members[len(results):],
                        state='ORIGINAL_OUTCOME_UNKNOWN_RECONCILE')
    return dict(results=results,state='ALL_SCOPED_RECEIPTS_RECORDED',remaining=[])


def require_fresh_group(proof,revision,now,wall):
    require(proof['publication']==revision and 0<=now-proof['received_monotonic']<=45
            and abs(wall-now-proof['anchor'])<2 and 0<=proof['RTT']<=45,'fresh same-parent receipt required')


def runtime(reader,core,base,feed,previous,cohort_runtime,material,claim_sha):
    clone = material[1]['_fn']
    inherited = clone(cohort_runtime,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    context,source,single_read,render,dispatch,observe = inherited(previous,None,material,claim_sha)
    publish = dispatch.__globals__['publication']
    single_path = dispatch.__globals__['read_path']
    single_terminal = dispatch.__globals__['terminal']
    single_identity = observe.__globals__['terminal_identity']

    def group_identity(row,nonce,proof,read_support=pin):
        require(proof['read_id']==nonce and proof['parent']==row['parent'] and proof['replay_allowed'] is False
                and REV(proof['publication']) and 1<=len(proof['members'])<=4
                and row['node'] in proof['members'] and len(set(proof['members']))==len(proof['members']),
                'exact group terminal receipt scope differs')
        names = proof['members']
        selected = [feed[node] for node in names]
        require(all(old['parent']==row['parent'] for old in selected)
                and nonce not in {r['new_request'] for r in material[6].values()},'group identities differ')
        name = f"{reader['STEM']}-{nonce}.json"
        require(proof['reader_receipt']==name,'exact supporting read file differs')
        receipt = json.loads(read_support((f"local-only/{name}",proof['reader_receipt_SHA256'])))
        payload = receipt['observed']
        parent = row['parent']
        require(receipt['rc']==0 and receipt['stderr_SHA256']==EMPTY and receipt['replay_allowed'] is False
                and receipt['Slurm_mutations']==receipt['physical_admissions']==payload['Slurm_mutations']==payload['physical_admissions']==0
                and payload['read_id']==nonce and set(payload['parents'])=={parent}
                and receipt['source_SHA256']==sha256(reader['render'](core,base,{parent:selected},nonce).encode()).hexdigest(),
                'actual fresh same-parent read differs')
        branch = payload['parents'][parent]['response']
        require(branch['read_id']==nonce and branch['Slurm_mutations']==branch['physical_admissions']==0
                and branch['plans_only'] is branch['requires_fresh_independent_device_and_dispatch_checks'] is True
                and [v['node'] for v in branch['targets']]==names,'fresh parent target roster differs')
        item = branch['targets'][names.index(row['node'])]
        gpu,files,pid = proof_row(item,feed[row['node']])
        require(gpu==row['pinned_GPU'] and pid==row['previous_worker_pid']
                and files['terminal']==row['pinned_terminal'] and files['cgroup']==row['cgroup_SHA256']
                and files['submission']==row['submit_SHA256'],'current same-parent worker identity differs')
        return gpu,(files['terminal'],files['cgroup'])

    def read_path(row,nonce):
        group = group_path(row['parent'],nonce)
        single = single_path(row,nonce)
        if os.path.lexists(group):
            require(not os.path.lexists(single),'read ID collides between group and singleton')
            return group
        return single

    def identity(row,nonce,proof):
        if proof.get('kind')=='R84_GROUP':
            return group_identity(row,nonce,proof)
        return single_identity(row,nonce,proof)

    def publication(ctx,revision):
        publish(ctx,revision)
        _,publisher = ctx[3][-2]['modules']()
        for spec in (R70,R72,R74,R76):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                    'exact published owner differs')

    def send(ctx,row,revision,term_args):
        def terminal(ctx,row,nonce,proof,*,now,wall):
            if proof.get('kind')!='R84_GROUP':
                return single_terminal(ctx,row,nonce,proof,now=now,wall=wall)
            require_fresh_group(proof,revision,now,wall)
            return group_identity(row,nonce,proof)
        return clone(dispatch,terminal=terminal,read_path=read_path,publication=publication)(ctx,row,revision,term_args)

    watch = clone(observe,read_path=read_path,terminal_identity=identity)
    return context,source,single_read,render,send,watch,group_identity


def group_fixture(reader,core,base,feed,rows,identity):
    receipt = json.loads(pin(ORIGINAL))
    snapshot = receipt['observed']
    nonce = '2'*32
    revision = '3'*40
    for parent,part in snapshot['parents'].items():
        branch = dict(part['response'],read_id=nonce,targets=part['response']['targets'][:3])
        names = [r['node'] for r in branch['targets']]
        prior = [feed[node] for node in names]
        source = reader['render'](core,base,{parent:prior},nonce)
        staged = dict(receipt,source_SHA256=sha256(source.encode()).hexdigest(),
                      observed=dict(snapshot,read_id=nonce,parents={parent:dict(response=branch)}))
        supporting = json.dumps(staged,sort_keys=True).encode()
        name = f"{reader['STEM']}-{nonce}.json"
        group = dict(kind='R84_GROUP',read_id=nonce,publication=revision,parent=parent,members=names,
                     reader_receipt=name,reader_receipt_SHA256=sha256(supporting).hexdigest(),
                     received_monotonic=25,anchor=1000,RTT=4,replay_allowed=False)
        for node in names:
            def load(spec):
                require(spec==(f'local-only/{name}',sha256(supporting).hexdigest()),'in-memory fixture hash differs')
                return supporting
            row = rows[node]
            require_fresh_group(group,revision,50,1050)
            require(identity(row,nonce,group,load)==(row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256'])),
                    'matching current group did not yield exact pins')
        for mode in ('stale','clock','RTT','publication','roster','paper','digest'):
            try:
                if mode == 'stale':
                    require_fresh_group(group,revision,71,1071)
                elif mode == 'clock':
                    require_fresh_group(group,revision,50,1053)
                elif mode == 'RTT':
                    require_fresh_group(dict(group,RTT=46),revision,50,1050)
                elif mode == 'publication':
                    require_fresh_group(group,'4'*40,50,1050)
                elif mode == 'roster':
                    identity(row,nonce,dict(group,members=list(reversed(names))),load)
                elif mode == 'paper':
                    identity(row,nonce,dict(group,reader_receipt='unauthenticated.json'),load)
                else:
                    identity(row,nonce,dict(group,reader_receipt_SHA256='0'*64),load)
            except ValueError:
                pass
            else:
                raise ValueError(f'bad batch receipt accepted: {mode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--show-run-order',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--observe-group')
    modes.add_argument('--dispatch-group',nargs=2,metavar=('READ_ID','SHA256'))
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node')
    parser.add_argument('--members',nargs='+')
    parser.add_argument('--published-revision')
    parser.add_argument('--terminal-receipt',nargs=2)
    args = parser.parse_args()
    reader,core,base,feed,previous,inherited,material = load()
    budget,first,_,_,n02,_,rows,_,claim = material
    claim_bytes = raw(CLAIM)
    require(json.loads(claim_bytes)==claim,'exact new claim differs')
    ctx,source,read,render,send,observe,identity = runtime(reader,core,base,feed,previous,inherited,material,sha256(claim_bytes).hexdigest())
    if args.check:
        for row in rows.values():
            source(ctx,row,'1'*32)
            pins = (row['pinned_terminal'],row['cgroup_SHA256'])
            render(ctx,row,row['pinned_GPU'],pins,None)
            datum = json.dumps(dict(node=row['node'],step=row['previous'],worker_pid=row['previous_worker_pid'],
                worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
            sha = sha256(datum).hexdigest()
            host = render(ctx,dict(row,cgroup_SHA256=sha),row['pinned_GPU'],(pins[0],sha),None)['launch']
            first['cgroup_fixture'](host,row,datum,require)
            budget['spawn_fixture'](host,row,ctx[1][6]['controller.py'])
        n02['fixture_holds']()
        group_fixture(reader,core,base,feed,rows,identity)
        output = dict(state='PREPARATION_ONLY',rows_checked=len(rows),group_proof_vetoes=14,
                      physical_admissions=0,SSH=0,Git=0,GPU=0)
    elif args.show_run_order:
        output = dict(state='PREPARED_ONLY',ordered=[dict(node=n,**claim['targets'][n]) for n in claim['parent_serial_order']],
                      max_concurrent_send=1,replay_allowed=False)
    elif args.observe_group:
        require(args.members and args.published_revision and not args.node and set(args.members)<=set(rows)
                and args.observe_group not in {r['new_request'] for r in rows.values()},'exact future group arguments differ')
        output = group_read(reader,core,base,feed,args.members,args.observe_group,args.published_revision)
    elif args.dispatch_group:
        require(args.members and args.published_revision and not args.node and not args.terminal_receipt,
                'use the same bounded group and its published revision')
        nonce,digest = args.dispatch_group
        output = dispatch_group(send,ctx,rows,args.members,nonce,digest,args.published_revision)
    else:
        require(args.node in rows and not args.members and not args.published_revision,'one exact node is required')
        row = rows[args.node]
        if args.observe_predecessor:
            output = read(ctx,row,args.observe_predecessor)
        elif args.observe_original:
            output = observe(ctx,row)
        else:
            require(args.terminal_receipt and REV(args.publication),'actual publication and fresh receipt required')
            output = send(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(output,sort_keys=True))


if __name__ == '__main__':
    main()

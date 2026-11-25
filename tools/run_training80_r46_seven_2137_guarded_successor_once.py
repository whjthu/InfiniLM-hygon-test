"""Renew seven independently returned workers through their existing ownership gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r46-seven-2137-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r46_seven_2137_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
PREVIOUS = ('tools/run_training80_r45_six_2117_guarded_successor_once.py',
            '067934fedb0e6b89a59ac8f5f3c65ac763ed55784680e880bcf6b1dc74644786')
FOLLOWING = ('tools/run_training80_r35_following_four_guarded_successor_once.py',
             'a9e149ccc1e0aac5265fc8ee105a5703186c2c62a00ab7ee6454392008567b6c')
WARNING = ('tools/run_training80_r36_two_warning_guarded_successor_once.py',
           '801531127d7351ca6e57dfab16c62ac6c1e3cd94ccce363ca8245de196831dd5')
LAST40 = ('tools/run_training80_r36_last40_two_guarded_successor_once.py',
          '15036fd95a3f45146c4d7e9b7f357b7b38f0cf29e948096dfff29576ee8e0ea7')
READER = ('local-only/read_training80_r46_2137_exact_originals_once.py',
          '92c68a4a8d435532ad8acdb1acb3a47f053cd1122823e13e8c32f7ab1c5cf6e5')
ORIGINALS = {
    '1103606': ('local-only/training80-r46-2137-exact-originals-1103606-f5d6568809124a6c8102d451a6951863.json',
                'e9f692e36174111187d940e357bb6b4e9028bd4f043783b856dbe90e321f0a05'),
    '1194900': ('local-only/training80-r46-2137-exact-originals-1194900-632f17433881444a9e9bdca0776db079.json',
                '9da40a569658675bbe432217cdd8e7e00a0a1b58d8320d5f160601ddafa22303'),
}
ORDER = ('f13r2n05','f13r1n07','f13r1n11','f13r2n10','f13r2n11','f13r3n09','f13r4n14')
SPECIAL = {'f13r1n11','f13r2n10'}
FENCES = {'f13r1n19','f13r1n12','f12r2n10','f12r2n11','f13r3n13'}
EMPTY = sha256(b'').hexdigest()


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pinned(spec):
    path = ROOT / spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact local evidence path differs')
    with path.open('rb') as stream:
        raw = stream.read(262145)
    require(0 < len(raw) <= 262144 and digest(raw)==spec[1],'exact local evidence bytes differ')
    return raw


def module(spec):
    pinned(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def load():
    previous = module(PREVIOUS)
    r44 = previous['module'](previous['R44'])
    budget, first, r34, early, n02, context, _, _ = r44['load']()
    _, load_following, _ = module(FOLLOWING)['exact_api']()
    following = load_following()
    warning = module(WARNING)['load']()
    last = module(LAST40)['load']()
    following_routes, last_routes = following[4][5], last[5][5]
    require(set(following[-2]) == set(ORDER)-SPECIAL-{'f13r2n11'}
            and set(warning[-2]) == SPECIAL and 'f13r2n11' in last[-3],
            'actual prior route ownership differs')
    routes = {node:following_routes[node] for node in following[-2]}
    routes['f13r2n11'] = last_routes['f13r2n11']
    context = (*context[:5],routes,context[6])
    controller = budget['controller'](context[1][6]['controller.py'])
    reader = module(READER)
    read_core,read_base,inputs = reader['load']()
    require([row['node'] for row in inputs] == list(ORDER) and set(ORDER).isdisjoint(FENCES),
            'fixed independent proof scope conflicts')
    rows, targets = {}, {}
    for parent, reference in ORIGINALS.items():
        packet = json.loads(pinned(reference))
        observed = packet['observed']
        expected = [row for row in inputs if row['parent']==parent]
        require(packet['rc']==0 and packet['parent']==parent and packet['stderr_SHA256']==EMPTY
                and packet['replay_allowed'] is False
                and packet['Slurm_mutations']==packet['physical_admissions']==observed['Slurm_mutations']==observed['physical_admissions']==0
                and observed['plans_only'] is observed['requires_fresh_independent_device_and_dispatch_checks'] is True
                and packet['selected_nodes']==[row['node'] for row in expected]
                and len(expected)==len(observed['targets']), 'original exact independent parent receipt differs')
        source = read_core['render'](read_base,expected,observed['read_id'])
        require(packet['source_SHA256']==digest(source.encode()),'actual independent parent read program differs')
        for old,proof in zip(expected,observed['targets'],strict=True):
            node,step,request = old['node'],old['previous'],old['request']
            require(proof['node']==node and proof['prior_numbered']==step and proof['prior_request']==request
                    and proof['parent']==parent and proof['scope']=='NATURAL' and proof['physical_clear'] is False
                    and proof['qualified_for_separate_fresh_physical'] is proof['original_natural_return'] is True
                    and proof['controller_numeric']==proof['queue_numeric']==[], 'actual C0 or original return not independently observed')
            gpu = proof['original_accounting']
            cpu, = proof['original_CPU']
            files = proof['files_SHA256']
            start,seconds = old['worker_start_epoch'],old['worker_seconds']
            require(len(gpu)==len(cpu)==9 and gpu[:4]==[step,f'training80-dcu-standby-{request[:12]}-{node}','COMPLETED','0:0']
                    and gpu[5:7]==[node,early['GPU_TRES']] and gpu[7]<gpu[8] and int(gpu[4])>=seconds
                    and cpu[:4]+cpu[5:7]+cpu[8:]==old['old_CPU'] and cpu[7]<=cpu[8]
                    and set(files)=={'cgroup','expected','submission','terminal','worker'}
                    and all(re.fullmatch(r'[a-f0-9]{64}', value or '') for value in files.values())
                    and files['worker']==old['worker_SHA256'] and files['expected']==old['expected_SHA256']
                    and files['submission']==old['submit_SHA256'] and len(set(proof['pci']))==8 and proof['pci']==old['pci']
                    and proof['worker_pid']==old['previous_worker_pid']>0 and proof['worker_start_epoch']==start
                    and proof['worker_seconds']==seconds and proof['returned_epoch']>=start+seconds,
                    'paired CPU, immutable worker, cgroup or original PCI evidence differs')
            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]
            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,
                    'old accepted scope route or actual controller budget differs')
            new = uuid5(UUID('c90b7f77-0127-48da-9f70-395fdb980bd3'), f'{node}/{step}/{request}').hex
            renewal = controller['node_seconds'](node,10800,new)
            require(new!=request and 10500<=renewal<=10800,'new deterministic worker request or budget differs')
            rows[node] = dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'],clearance=old['old_CPU'],worker_start_epoch=start,worker_seconds=seconds,
                planned_end_epoch=start+seconds,prepare_epoch=start+seconds-900,pci=proof['pci'],
                worker_SHA256=files['worker'],expected_SHA256=files['expected'],submit_SHA256=files['submission'],
                cgroup_SHA256=files['cgroup'],previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal)
            targets[node] = dict(parent=parent,original_GPU=step,original_CPU=cpu[0],original_request=request,
                new_request=new,actual_prior_seconds=seconds,new_worker_seconds=renewal,
                Slurm_time_cap_seconds=renewal+300,original_worker_files=files,
                queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR')
    require(set(rows)==set(ORDER) and len({v['new_request'] for v in rows.values()})==7,'fixed seven exclusive new requests differ')
    claim = dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
        launcher=WRAPPER.relative_to(ROOT).as_posix(),original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()},
        original_local_reader_SHA256=READER[1],
        source_pins={spec[0]:spec[1] for spec in (PREVIOUS,FOLLOWING,WARNING,LAST40)},
        controller_SHA256=budget['CONTROLLER'][1],targets=targets,parent_serial_order=list(ORDER),
        priority_scopes_preserved=sorted(FENCES),ready_GPU_count=0,GPU_per_node=8,max_invocations_per_node=1,
        max_concurrent_send=1,ledger_mutations=0,replay_allowed=False)
    return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)


def runtime(previous,warning,material,claim_sha,ledger_fence=None):
    first = material[1]
    clone = first['_fn']
    inherited = clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx, source, read, render, dispatch, observe = inherited(material,claim_sha)
    publish = dispatch.__globals__['publication']
    ordinary_fence = dispatch.__globals__['exact_no_other_claims']
    specialist_owner,_,_,_,specialist = warning[:5]
    r27,base = specialist[:2]
    def fence(ordinary,authority,row):
        node = row['node']
        if node in SPECIAL:
            specialist_owner['local_fences'](specialist,node)
            (ledger_fence or r27['require_no_slots'])(base['JOURNAL'],node,row['previous'],(0,1,2))
        else:
            ordinary_fence(ordinary,authority,row)
    def publication(actual,revision):
        publish(actual,revision)
        _,publisher = actual[3][-2]['modules']()
        for spec in (PREVIOUS,FOLLOWING,WARNING,LAST40,module(WARNING)['R34']):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pinned(spec),
                    'published exact guarded ownership source differs')
    return ctx,source,read,render,clone(dispatch,publication=publication,exact_no_other_claims=fence),observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--show-run-order',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=ORDER)
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args = parser.parse_args()
    require((args.check or args.show_run_order or args.node) and bool(args.publication)==bool(args.terminal_receipt),
            'one exact live node and fresh predecessor receipt required')
    previous,warning,material = load()
    budget,first,r34,_,n02,_,rows,_,claim = material
    raw = r34['pinned'](CLAIM)
    require(json.loads(raw)==claim,'seven authorized originals or exact compute scope changed')
    claim_sha = digest(raw)
    if args.show_run_order:
        result = dict(state='PREPARED_ONLY',ordered=[
           {key:rows[node][key] for key in ('node','parent','previous','new_request','renewal_seconds')} for node in ORDER],
           max_concurrent_send=1,replay_allowed=False)
    else:
        ctx,source,read,render,dispatch,observe = runtime(previous,warning,material,claim_sha)
        if args.check:
            cases = {}
            for node,row in rows.items():
                read_source = source(ctx,row,'1'*32)
                cgroup = json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],
                    worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
                rendered = render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)
                first['cgroup_fixture'](rendered['launch'],row,cgroup,require)
                seconds = budget['spawn_fixture'](rendered['launch'],row,ctx[1][6]['controller.py'])
                cases[node] = dict(request=row['new_request'],worker_seconds=seconds,
                                   read_SHA256=digest(read_source.encode()),role=claim['targets'][node]['queue_ownership'])
            n02['fixture_holds']()
            result = dict(state='PREPARED_ONLY',targets=cases,locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14,
                         HOST_spawn_positive=7,HOST_spawn_vetoes=14,SSH=0,Git=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            result = read(ctx,rows[args.node],args.observe_predecessor)
        elif args.observe_original:
            result = observe(ctx,rows[args.node])
        else:
            result = dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=claim_sha),sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()

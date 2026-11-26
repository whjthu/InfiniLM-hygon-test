"""Renew only the four reconciled R75 and six new 10:13 ordinary departures."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r85-ten-1013-guarded-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R84 = ('tools/run_training80_r84_seventeen_0930_guarded_successor_once.py',
       'fc7443510c0570c9be042219000dd82ec4914055eaad241b3916bb129ed40077')
R83 = ('tools/run_training80_r83_r75_guarded_successor_once.py',
       '2a29e7a8c5367e262e1a7cd945d28e06c562950ba66049e6b2b210bfa315e6cc')
R75 = ('tools/run_training80_r75_r57_remaining_twelve_guarded_successor_once.py',
       '6f2e8db0e1e90cce833536d6a1d4ea9aa12c3e9a6635897d3e308d47825737ca')
R77 = ('tools/run_training80_r77_r60_r61_two_guarded_successor_once.py',
       'eb75f3012f5dc3b6d927c4966e566cb884e654923767e0551a45129122d24074')
R78 = ('tools/run_training80_r78_r62_r64_r65_five_guarded_successor_once.py',
       'd7d97677933b177d0d8bd510d3a62c75212ce41c5a213abb9ec21267efdc2327')
FOUR_READER = ('local-only/read_training80_r83_r75_twelve_0907_exact_originals_once.py',
               'c10eedcee0ee1a24f07fb0394f771f416ba08cc2d96baf8589d6c49e4994b279')
FOUR = ('local-only/training80-r83-r75-twelve-0907-exact-originals-1194900-616799e5f55a4b68981ad4483d13e7b7.json',
        '1c67604e7f754a614423178096849a13decae8ffcbcf28cb1b474884889908f2')
SIX_READER = ('local-only/read_training80_r85_1013_six_new_ordinary_frontiers_once.py',
              'edd0cea62aca14bc7e421d72aa53b4163c1007f81432c31864cbc09418f1c542')
SIX = ('local-only/training80-r85-1013-six-new-ordinary-frontiers-ee545e593bbd4b928325fc8d0a088965.json',
       '6db6431eb33b33cd578f8f3230637d0167ea5ef1545b888f0a58242034f6e519')
FOUR_NODES = ('f12r3n07','f13r4n15','f12r3n18','f13r3n03')


def clone(function,**changes):
    return FunctionType(function.__code__,dict(function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def entry():
    if sha256((ROOT/R84[0]).read_bytes()).hexdigest()!=R84[1]:
        raise ValueError('frozen R84 guard differs')
    guard = runpy.run_path(str(ROOT/R84[0]))
    require,pin,module = (guard[k] for k in ('require','pin','module'))
    scope_paths = dict(STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    group_path = clone(guard['group_path'],**scope_paths)
    verified_dependencies = (R75,R77,R78,R83,R84)

    def load():
        source = module(FOUR_READER)
        read_core,read_base,prior_groups,_,_ = source['load']()
        old_four = prior_groups[2]
        first = json.loads(pin(FOUR))
        first_body = first['observed']
        require([r['node'] for r in old_four]==first['selected_nodes']==list(FOUR_NODES)
                and first['parent']=='1194900' and first['rc']==0 and first['stderr_SHA256']==guard['EMPTY']
                and first['replay_allowed'] is False and first_body['plans_only'] is True
                and first_body['requires_fresh_independent_device_and_dispatch_checks'] is True
                and first['Slurm_mutations']==first['physical_admissions']==first_body['Slurm_mutations']==first_body['physical_admissions']==0
                and first['source_SHA256']==sha256(read_core['render'](read_base,old_four,first_body['read_id']).encode()).hexdigest(),
                'four reconciled R75 receipt or original read source differs')
        reader,core,base,old_six,_,_ = module(SIX_READER)['load']()
        second = json.loads(pin(SIX))
        second_body = second['observed']
        second_id = second_body['read_id']
        require(second['rc']==0 and second['stderr_SHA256']==guard['EMPTY'] and second['replay_allowed'] is False
                and second['Slurm_mutations']==second['physical_admissions']==second_body['Slurm_mutations']==second_body['physical_admissions']==0
                and set(old_six)==set(second_body['parents'])
                and second['source_SHA256']==sha256(reader['render'](core,base,old_six,second_id).encode()).hexdigest(),
                'six new receipt or frozen aggregate source differs')
        pairs = list(zip(old_four,first_body['targets'],strict=True))
        for parent,rows in old_six.items():
            branch = second_body['parents'][parent]['response']
            require(branch['read_id']==second_id and branch['plans_only'] is True
                    and branch['requires_fresh_independent_device_and_dispatch_checks'] is True
                    and branch['physical_admissions']==branch['Slurm_mutations']==0,'six original parent read scope differs')
            pairs.extend(zip(rows,branch['targets'],strict=True))
        r76 = module(guard['R76'])
        load76, runtime76,_ = r76['entry']()
        previous,_,template = load76()
        budget,first_api,r34,early,n02,context,_,_,_ = template
        sources = (module(R75)['entry']()[0]()[2],template,
                   module(R77)['entry']()[0]()[2],module(R78)['entry']()[0]()[2])
        controller = budget['controller'](context[1][6]['controller.py'])
        rows,feed,targets,routes = {},{},{},{}
        for old,item in pairs:
            node,parent,step,request = (old[k] for k in ('node','parent','previous','request'))
            [origin] = [data for data in sources if node in data[6]]
            accepted = origin[6][node]
            require(accepted['new_request']==request and accepted['parent']==parent
                    and accepted['renewal_seconds']==old['worker_seconds'],'claimed old ordinary lineage differs')
            gpu,files,pid = guard['proof_row'](item,old)
            new = uuid5(UUID('cff655c8-4dba-4c5a-8c67-65fc9e2d3358'),f'{node}/{step}/{request}').hex
            seconds = controller['node_seconds'](node,10800,new)
            require(new!=request and 10500<=seconds<=10800 and node not in rows,'unique successor identity or duration differs')
            rows[node] = dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'],clearance=old['old_CPU'],worker_start_epoch=old['worker_start_epoch'],
                worker_seconds=old['worker_seconds'],planned_end_epoch=old['planned_end_epoch'],
                prepare_epoch=old['planned_end_epoch']-900,pci=old['pci'],worker_SHA256=files['worker'],
                expected_SHA256=files['expected'],submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],
                previous_worker_pid=pid,renewal_seconds=seconds)
            routes[node] = origin[5][5][node]
            feed[node] = old
            targets[node] = dict(parent=parent,original_GPU=step,original_CPU=old['old_CPU'][0],original_request=request,
                new_request=new,actual_prior_seconds=old['worker_seconds'],new_worker_seconds=seconds,
                Slurm_time_cap_seconds=seconds+300,original_worker_files=files,original_receipt_SHA256=FOUR[1] if node in FOUR_NODES else SIX[1],
                queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR')
        old_launchers = module(R83)['ORDER']
        earlier_r84 = json.loads(pin((guard['CLAIM'].relative_to(ROOT).as_posix(),
                                     '6ea23fd30d68e9ab970892bb65faf67ef057b5966463590258b6b73b5f733d80')))
        require(len(rows)==len({r['new_request'] for r in rows.values()})==10
                and not set(rows)&(set(old_launchers)|set(earlier_r84['targets'])|{'f13r1n12','f13r3n13','f13r2n19'}),
                'current previous dispatch or high-priority scope overlaps')
        context = (*context[:5],routes,context[6])
        order = sorted(rows,key=lambda n:rows[n]['planned_end_epoch'])
        claim = dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
            launcher=WRAPPER.relative_to(ROOT).as_posix(),original_four_receipt_SHA256=FOUR[1],original_six_receipt_SHA256=SIX[1],
            original_reader_SHA256=dict((FOUR_READER,SIX_READER)),source_pins=dict(verified_dependencies+(guard['R76'],)),
            targets=targets,parent_serial_order=order,ready_GPU_count=0,GPU_per_node=8,max_invocations_per_node=1,
            max_concurrent_send=1,max_nodes_per_fresh_parent_read=4,ledger_mutations=0,replay_allowed=False)
        capture = STEM+'-read'
        reader = dict(reader,STEM=capture,client=clone(reader['client'],STEM=capture))
        return reader,core,base,feed,previous,runtime76,(budget,first_api,r34,early,n02,context,rows,{},claim)

    inherited = clone(guard['runtime'],**scope_paths,group_path=group_path,R70=R75,R72=R77,R74=R78)

    def runtime(*args):
        ctx,source,read,render,send,observe,identity = inherited(*args)

        def another_pin(ctx,revision):
            # This runs inside the inherited send before reserving that node.
            _,publisher=ctx[3][-2]['modules']()
            for spec in (R83,R84):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published prior group launcher differs')

        def checked_send(ctx,row,revision,receipt):
            require(guard['REV'](revision),'exact published revision required')
            another_pin(ctx,revision)
            return send(ctx,row,revision,receipt)

        return ctx,source,read,render,checked_send,observe,identity

    def group_fixture(reader,core,base,feed,rows,identity):
        # Reuse the existing verifier for both parents with in-memory terminal fixtures.
        old_four = json.loads(pin(FOUR))['observed']
        old_six = json.loads(pin(SIX))
        body = old_six['observed']
        p = body['parents']['1194900']['response']
        updated = dict(p, targets=old_four['targets'][:2]+p['targets'])
        combined = dict(old_six, observed=dict(body,parents=dict(body['parents'], **{'1194900':dict(response=updated)})))
        fixture = guard['group_fixture']
        source = inspect.getsource(fixture)
        anchor = 'members=list(reversed(names))'
        require(source.count(anchor)==1,'original local roster fixture differs')
        source = source.replace(anchor, 'members=list(reversed(names)) if len(names)>1 else []', 1)
        namespace = dict(fixture.__globals__,pin=lambda spec:json.dumps(combined).encode())
        exec(compile(source,'<one-member roster negative fixture>','exec'),namespace)
        namespace['group_fixture'](reader,core,base,feed,rows,identity)

    main = clone(guard['main'],**scope_paths,load=load,runtime=runtime,group_fixture=group_fixture,
                 group_read=clone(guard['group_read'],group_path=group_path),
                 dispatch_group=clone(guard['dispatch_group'],group_path=group_path))
    return load,main


if __name__ == '__main__':
    _, main = entry()
    main()

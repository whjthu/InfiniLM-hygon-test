"""Renew the three independently closed R87 workers on parent 1103606."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r87-three-1103606-guarded-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
PARENT = '1103606'
NODES = ('f13r1n07','f13r2n11','f13r2n05')
R84 = ('tools/run_training80_r84_seventeen_0930_guarded_successor_once.py',
       'fc7443510c0570c9be042219000dd82ec4914055eaad241b3916bb129ed40077')
R80 = ('tools/run_training80_r80_r64_r66_five_guarded_successor_once.py',
       '3dbcc9574ddc81eb6a64d9a8056eab2c6e7374999b1dda13aba489a37f18b4ac')
READER = ('local-only/read_training80_r87_1055_seven_new_ordinary_frontiers_once.py',
          'eb76a7a743927bead41ba51352a596cab6f61343603c711186a24545a67f23db')
ACTUAL = ('local-only/training80-r87-1055-seven-new-ordinary-frontiers-65eaad9516594f43b31c7687a2707890.json',
          'a6354ecbcd9d5ab01b792ec695418e59556170f253abf06810075f0317b4177c')


def clone(function,**changes):
    return FunctionType(function.__code__,dict(function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def entry():
    if sha256((ROOT/R84[0]).read_bytes()).hexdigest()!=R84[1]:
        raise ValueError('frozen R84 guard differs')
    guard = runpy.run_path(str(ROOT/R84[0]))
    require,pin,module = (guard[k] for k in ('require','pin','module'))
    scope = dict(STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    group_path = clone(guard['group_path'],**scope)

    def load():
        reader,core,base,groups,_,_ = module(READER)['load']()
        packet = json.loads(pin(ACTUAL))
        observed = packet['observed']
        nonce = observed['read_id']
        parents = observed['parents']
        require(guard['ID'](nonce) and packet['rc']==0 and packet['stderr_SHA256']==guard['EMPTY']
                and packet['replay_allowed'] is False and set(parents)==set(groups)=={PARENT,'1194900'}
                and parents['1194900']=={'response':None,'error_class':'ValueError'}
                and packet['Slurm_mutations']==packet['physical_admissions']==observed['Slurm_mutations']==observed['physical_admissions']==0
                and packet['source_SHA256']==sha256(reader['render'](core,base,groups,nonce).encode()).hexdigest(),
                'actual partial R87 receipt or exact original source differs')
        branch = parents[PARENT]['response']
        require(tuple(row['node'] for row in groups[PARENT])==NODES
                and branch['read_id']==nonce and branch['Slurm_mutations']==branch['physical_admissions']==0
                and branch['plans_only'] is branch['requires_fresh_independent_device_and_dispatch_checks'] is True,
                'exact known parent scope differs')
        old_route = module(R80)['load']()
        _,_,_,_,_,_,ctx80,rows80,claim80 = old_route
        r76 = module(guard['R76'])
        load76,runtime76,_ = r76['entry']()
        previous,_,template = load76()
        budget,first,r34,early,n02,context,_,_,_ = template
        controller = budget['controller'](context[1][6]['controller.py'])
        rows,feed,routes,targets = {},{},{},{}
        for old,item in zip(groups[PARENT],branch['targets'],strict=True):
            node,parent,step,request = (old[k] for k in ('node','parent','previous','request'))
            incumbent = rows80[node]
            require(parent==PARENT==incumbent['parent']==ctx80[5][node]['parent']
                    and incumbent['new_request']==request and incumbent['renewal_seconds']==old['worker_seconds']
                    and claim80['targets'][node]['queue_ownership']=='ADOPTED_ORDINARY58_EXACT_ANCESTOR',
                    'R80 adopted predecessor or actual worker budget differs')
            gpu,files,pid = guard['proof_row'](item,old)
            new = uuid5(UUID('003006cb-a8ae-4475-8ef2-27d8d9679d97'),f'{node}/{step}/{request}').hex
            seconds = controller['node_seconds'](node,10800,new)
            require(new!=request and 10500<=seconds<10800,'new exact worker request or budget differs')
            rows[node] = dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'],clearance=old['old_CPU'],worker_start_epoch=old['worker_start_epoch'],
                worker_seconds=old['worker_seconds'],planned_end_epoch=old['planned_end_epoch'],
                prepare_epoch=old['planned_end_epoch']-900,pci=old['pci'],worker_SHA256=files['worker'],
                expected_SHA256=files['expected'],submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],
                previous_worker_pid=pid,renewal_seconds=seconds)
            feed[node] = old
            routes[node] = ctx80[5][node]
            targets[node] = dict(parent=parent,original_GPU=step,original_CPU=old['old_CPU'][0],original_request=request,
                new_request=new,actual_prior_seconds=old['worker_seconds'],new_worker_seconds=seconds,
                original_worker_files=files,queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',
                Slurm_time_cap_seconds=seconds+300)
        require(tuple(rows)==NODES and len({r['new_request'] for r in rows.values()})==3,'narrowed exact three differ')
        context=(*context[:5],routes,context[6])
        claim=dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
            launcher=WRAPPER.relative_to(ROOT).as_posix(),original_receipt_SHA256=ACTUAL[1],
            original_reader_SHA256=READER[1],source_pins=dict((R80,R84,guard['R76'])),targets=targets,
            parent_serial_order=list(NODES),excluded_unresolved_parent='1194900',ready_GPU_count=0,GPU_per_node=8,
            max_invocations_per_node=1,max_concurrent_send=1,max_nodes_per_fresh_parent_read=3,ledger_mutations=0,replay_allowed=False)
        capture=STEM+'-read'
        reader=dict(reader,STEM=capture,client=clone(reader['client'],STEM=capture))
        return reader,core,base,feed,previous,runtime76,(budget,first,r34,early,n02,context,rows,{},claim)

    runtime=clone(guard['runtime'],**scope,group_path=group_path,R70=R80,R72=R84)

    def group_fixture(reader,core,base,feed,rows,identity):
        receipt=json.loads(pin(ACTUAL))
        original=receipt['observed']
        bounded=dict(receipt,observed=dict(original,parents={PARENT:original['parents'][PARENT]}))
        clone(guard['group_fixture'],pin=lambda _:json.dumps(bounded).encode())(reader,core,base,feed,rows,identity)

    def parser_main():
        import inspect
        source=inspect.getsource(guard['main'])
        require(source.count('group_proof_vetoes=14')==1,'inherited test output differs')
        source=source.replace('group_proof_vetoes=14','group_proof_vetoes=7',1)
        namespace=dict(guard,**scope,load=load,runtime=runtime,group_fixture=group_fixture,
            group_read=clone(guard['group_read'],group_path=group_path),
            dispatch_group=clone(guard['dispatch_group'],group_path=group_path))
        exec(compile(source,'<three scoped singleton and batch modes>','exec'),namespace)
        namespace['main']()

    return load,parser_main


if __name__ == '__main__':
    _,main=entry()
    main()

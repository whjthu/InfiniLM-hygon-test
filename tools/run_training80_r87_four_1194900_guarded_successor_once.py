"""Renew the four R87 originals qualified by the independent bound parent read."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
WRAPPER=Path(__file__).resolve()
STEM='training80-r87-four-1194900-guarded-bf16-v1'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
PARENT='1194900'
NODES=('f14r3n04','f12r3n02','f13r3n09','f12r3n15')
R84=('tools/run_training80_r84_seventeen_0930_guarded_successor_once.py',
     'fc7443510c0570c9be042219000dd82ec4914055eaad241b3916bb129ed40077')
R80=('tools/run_training80_r80_r64_r66_five_guarded_successor_once.py',
     '3dbcc9574ddc81eb6a64d9a8056eab2c6e7374999b1dda13aba489a37f18b4ac')
R81=('tools/run_training80_r81_r67_r68_two_guarded_successor_once.py',
     '69901515b302e97f187a0641eefe4a1ec8127d8ec442d7ed2d710b22f388bc3e')
READER=('local-only/render_training80_r87_1194900_bound_read_once.py',
        '8bdad893fefe64e9172b52a8ff133cf790abb9267b9e6f9354fe155dcc67d074')
ORIGINAL_BASE='local-only/training80-r87-1194900-bound-read-authority-recheck-27dedbf4b0d840b18c41a524be8540c6'
ORIGINAL_PINS={
    'intent':'d4f348738770646d26092c7b3b55d0fd1a4fe4fcf5282364ca2556179a7daa27',
    'send':'e2f0c4bb97be5f3aac5b27140107cbcf08766e12b260392447f7896b8f851adc',
    'receipt':'29a75eb475d4a1afd6079c4b0b41ab90d3459b5bac43c8ba4c540be23f05ea78',
}


def clone(function,**changes):
    return FunctionType(function.__code__,dict(function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def entry():
    if sha256((ROOT/R84[0]).read_bytes()).hexdigest()!=R84[1]:
        raise ValueError('frozen R84 guard differs')
    guard=runpy.run_path(str(ROOT/R84[0]))
    require,pin,module=(guard[k] for k in ('require','pin','module'))
    scope=dict(STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    group_path=clone(guard['group_path'],**scope)

    def load():
        decoder=module(READER)
        diagnostic,reader,core,base,known,_,_=decoder['prepared']()
        intent,send,packet=(json.loads(pin((f'{ORIGINAL_BASE}/{name}.json',digest))) for name,digest in ORIGINAL_PINS.items())
        body=packet['response']
        req=body['request']
        source=decoder['render'](diagnostic,reader,core,base,known,req)
        require(packet['rc']==0 and packet['authority_verified'] is packet['remote_received'] is packet['payload_flushed'] is True
                and packet['remote_outcome']=='RESPONSE_OBSERVED' and intent['request']==send['request']==packet['request']==body['read_id']==req
                and send['payload_sha256']==packet['payload_sha256'] and intent['program_sha'] in {
                    sha256(source.encode()).hexdigest(),sha256(source.replace('\n','\r\n').encode()).hexdigest()}
                and body['physical_admissions']==body['Slurm_mutations']==0 and set(body['parents'])=={PARENT},
                'exact authenticated independent parent read differs')
        branch=body['parents'][PARENT]['response']
        require(tuple(row['node'] for row in known)==NODES
                and branch['read_id']==req and branch['Slurm_mutations']==branch['physical_admissions']==0
                and branch['plans_only'] is branch['requires_fresh_independent_device_and_dispatch_checks'] is True,
                'scope of qualified parent differs')
        _,_,_,_,_,_,ctx80,rows80,claim80=module(R80)['load']()
        _,_,_,_,_,_,ctx81,rows81,claim81=module(R81)['entry']()[0]()
        load76,runtime76,_=module(guard['R76'])['entry']()
        previous,_,template=load76()
        budget,first,r34,early,n02,context,_,_,_=template
        controller=budget['controller'](context[1][6]['controller.py'])
        rows,feed,routes,targets={},{},{},{}
        for old,item in zip(known,branch['targets'],strict=True):
            node,parent,step,request=(old[k] for k in ('node','parent','previous','request'))
            prev_ctx,prev_rows,prev_claim=(ctx80,rows80,claim80) if node in rows80 else (ctx81,rows81,claim81)
            incumbent,owner=prev_rows[node],prev_ctx[5][node]
            require(parent==PARENT==incumbent['parent']==owner['parent']
                    and incumbent['new_request']==request and incumbent['renewal_seconds']==old['worker_seconds']
                    and prev_claim['targets'][node]['queue_ownership']=='ADOPTED_ORDINARY58_EXACT_ANCESTOR',
                    'original confirmed ordinary claim differs')
            gpu,files,pid=guard['proof_row'](item,old)
            new=uuid5(UUID('781402eb-c698-4392-ba05-22ae983f0d48'),f'{node}/{step}/{request}').hex
            seconds=controller['node_seconds'](node,10800,new)
            require(new!=request and 10500<=seconds<10800,'new request or exact budget differs')
            rows[node]=dict(node=node,parent=parent,previous=step,request=request,new_request=new,pinned_GPU=gpu,
                pinned_terminal=files['terminal'],clearance=old['old_CPU'],worker_start_epoch=old['worker_start_epoch'],
                worker_seconds=old['worker_seconds'],planned_end_epoch=old['planned_end_epoch'],
                prepare_epoch=old['planned_end_epoch']-900,pci=old['pci'],worker_SHA256=files['worker'],
                expected_SHA256=files['expected'],submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],
                previous_worker_pid=pid,renewal_seconds=seconds)
            routes[node],feed[node]=owner,old
            targets[node]=dict(parent=parent,original_GPU=step,original_CPU=old['old_CPU'][0],original_request=request,
                new_request=new,actual_prior_seconds=old['worker_seconds'],new_worker_seconds=seconds,
                original_worker_files=files,queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',
                Slurm_time_cap_seconds=seconds+300)
        require(tuple(rows)==NODES and len({r['new_request'] for r in rows.values()})==4,'exact four new scopes differ')
        context=(*context[:5],routes,context[6])
        claim=dict(scope=STEM,state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
            launcher=WRAPPER.relative_to(ROOT).as_posix(),original_read_transport_SHA256=ORIGINAL_PINS,
            original_reader_SHA256=READER[1],source_pins=dict((R80,R81,R84,guard['R76'])),targets=targets,
            parent_serial_order=list(NODES),excluded_other_parent='1103606',ready_GPU_count=0,GPU_per_node=8,
            max_invocations_per_node=1,max_concurrent_send=1,max_nodes_per_fresh_parent_read=4,ledger_mutations=0,replay_allowed=False)
        capture=STEM+'-read'
        reader=dict(reader,STEM=capture,client=clone(reader['client'],STEM=capture))
        return reader,core,base,feed,previous,runtime76,(budget,first,r34,early,n02,context,rows,{},claim)

    runtime=clone(guard['runtime'],**scope,group_path=group_path,R70=R80,R72=R81,R74=R84)

    def group_fixture(reader,core,base,feed,rows,identity):
        p=json.loads(pin((f'{ORIGINAL_BASE}/receipt.json',ORIGINAL_PINS['receipt'])))
        fixture=dict(rc=p['rc'],stderr_SHA256=guard['EMPTY'],replay_allowed=False,
                     physical_admissions=0,Slurm_mutations=0,observed=p['response'])
        clone(guard['group_fixture'],pin=lambda _:json.dumps(fixture).encode())(reader,core,base,feed,rows,identity)

    def parser_main():
        body=inspect.getsource(guard['main'])
        require(body.count('group_proof_vetoes=14')==1,'inherited fixture output differs')
        body=body.replace('group_proof_vetoes=14','group_proof_vetoes=7',1)
        namespace=dict(guard,**scope,load=load,runtime=runtime,group_fixture=group_fixture,
            group_read=clone(guard['group_read'],group_path=group_path),
            dispatch_group=clone(guard['dispatch_group'],group_path=group_path))
        exec(compile(body,'<four scoped original successor checks>','exec'),namespace)
        namespace['main']()

    return load,parser_main


if __name__=='__main__':
    _,main=entry()
    main()

"""Renew the one independently closed f13r1n05 R78 worker."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r86-n05-1032-guarded-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
NODE = 'f13r1n05'
R84 = ('tools/run_training80_r84_seventeen_0930_guarded_successor_once.py',
       'fc7443510c0570c9be042219000dd82ec4914055eaad241b3916bb129ed40077')
R78 = ('tools/run_training80_r78_r62_r64_r65_five_guarded_successor_once.py',
       'd7d97677933b177d0d8bd510d3a62c75212ce41c5a213abb9ec21267efdc2327')
READER = ('local-only/read_training80_r86_1032_n05_exact_original_once.py',
          'c27e0abe5b857f5f8acaf8ed71fd1ffb3d8ee4acf02e379dbf7426100247db45')
ACTUAL = ('local-only/training80-r86-1032-n05-exact-original-9e9f02d1554f405890f93daaf2531765.json',
          '3e5400ee70f12700cfe1eeb003b7effdead62a194b21abb249fa8df53b2d9d75')


def clone(function,**changes):
    return FunctionType(function.__code__,dict(function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def load():
    if sha256((ROOT/R84[0]).read_bytes()).hexdigest()!=R84[1]:
        raise ValueError('frozen R84 guard differs')
    guard = runpy.run_path(str(ROOT/R84[0]))
    require,pin,module = (guard[k] for k in ('require','pin','module'))
    reader,core,base,groups,_,_ = module(READER)['load']()
    parent = '1103606'
    [original] = groups[parent]
    receipt = json.loads(pin(ACTUAL))
    body = receipt['observed']
    read_id = body['read_id']
    require(guard['ID'](read_id) and receipt['rc']==0 and receipt['stderr_SHA256']==guard['EMPTY']
            and receipt['replay_allowed'] is False and set(body['parents'])=={parent}
            and receipt['physical_admissions']==receipt['Slurm_mutations']==body['physical_admissions']==body['Slurm_mutations']==0
            and receipt['source_SHA256']==sha256(reader['render'](core,base,groups,read_id).encode()).hexdigest(),
            'original actual read provenance differs')
    report = body['parents'][parent]['response']
    [item] = report['targets']
    require(report['read_id']==read_id and report['Slurm_mutations']==report['physical_admissions']==0
            and report['plans_only'] is report['requires_fresh_independent_device_and_dispatch_checks'] is True,
            'original request is not an exact read-only qualification')
    gpu,files,pid = guard['proof_row'](item,original)
    prior = module(R78)
    original_load,previous_runtime,_ = prior['entry']()
    previous,_,template = original_load()
    budget,first,r34,early,n02,context,earlier,_,_ = template
    old = earlier[NODE]
    request = original['request']
    require(old['parent']==original['parent']==parent and old['new_request']==request
            and old['renewal_seconds']==original['worker_seconds'] and original['previous']=='1103606.3648',
            'real R78 standby ancestor differs')
    new = uuid5(UUID('8eff30da-70a6-447b-baad-23a622405eac'),f"{NODE}/{original['previous']}/{request}").hex
    controller = budget['controller'](context[1][6]['controller.py'])
    seconds = controller['node_seconds'](NODE,10800,new)
    require(new!=request and 10500<=seconds<10800,'new controller identity differs')
    row = dict(node=NODE,parent=parent,previous=original['previous'],request=request,new_request=new,pinned_GPU=gpu,
        pinned_terminal=files['terminal'],clearance=original['old_CPU'],worker_start_epoch=original['worker_start_epoch'],
        worker_seconds=original['worker_seconds'],planned_end_epoch=original['planned_end_epoch'],
        prepare_epoch=original['planned_end_epoch']-900,pci=original['pci'],worker_SHA256=files['worker'],
        expected_SHA256=files['expected'],submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],
        previous_worker_pid=pid,renewal_seconds=seconds)
    routes = {NODE:context[5][NODE]}
    context = (*context[:5],routes,context[6])
    claim = dict(scope=STEM,state='PREPARED_ONLY_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',launcher=WRAPPER.relative_to(ROOT).as_posix(),
        original_exact_read_SHA256=ACTUAL[1],original_local_reader_SHA256=READER[1],source_pins=dict((R84,R78)),
        targets={NODE:dict(parent=parent,original_GPU=row['previous'],original_CPU=row['clearance'][0],original_request=request,
            new_request=new,actual_prior_seconds=row['worker_seconds'],new_worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300,
            original_worker_files=files,queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR')},
        ready_GPU_count=0,GPU_per_node=8,max_invocations_per_node=1,max_concurrent_send=1,ledger_mutations=0,replay_allowed=False)
    material = budget,first,r34,early,n02,context,{NODE:row},{},claim
    return guard,previous,previous_runtime,material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--terminal-receipt',nargs=2)
    args = parser.parse_args()
    guard,previous,inherited,material = load()
    require = guard['require']
    budget,first,_,_,n02,_,rows,_,claim = material
    row = rows[NODE]
    raw = guard['raw'](CLAIM)
    require(json.loads(raw)==claim,'frozen singleton scope differs')
    factory = clone(inherited,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,source,read,render,send,observe = factory(previous,None,material,sha256(raw).hexdigest())
    inherited_publication = send.__globals__['publication']

    def publication(context,revision):
        inherited_publication(context,revision)
        _,publisher=context[3][-2]['modules']()
        for spec in (R84,R78):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==guard['pin'](spec),'prior owner bytes differ')

    require(bool(args.publication)==bool(args.terminal_receipt),'published revision and exact fresh receipt required together')
    if args.check:
        prior = source(ctx,row,'1'*32)
        render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)
        data = json.dumps(dict(node=NODE,step=row['previous'],worker_pid=row['previous_worker_pid'],
            worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
        digest = sha256(data).hexdigest()
        host = render(ctx,dict(row,cgroup_SHA256=digest),row['pinned_GPU'],(row['pinned_terminal'],digest),None)['launch']
        first['cgroup_fixture'](host,row,data,require)
        seconds = budget['spawn_fixture'](host,row,ctx[1][6]['controller.py'])
        n02['fixture_holds']()
        output = dict(state='PREPARATION_ONLY',node=NODE,request=row['new_request'],worker_seconds=seconds,
                      fresh_read_SHA256=sha256(prior.encode()).hexdigest(),SSH=0,Git=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        output = read(ctx,row,args.observe_predecessor)
    elif args.observe_original:
        output = observe(ctx,row)
    else:
        output = clone(send,publication=publication)(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(output,sort_keys=True))


if __name__ == '__main__':
    main()

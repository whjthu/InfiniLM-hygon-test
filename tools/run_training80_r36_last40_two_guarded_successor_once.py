"""Renew two exact naturally returned last40 workers under their original authority."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r36-last40-two-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r36_last40_two_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
NEXT=('tools/run_training80_r35_next_three_guarded_successor_once.py',
      'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
LAST40=('tools/run_training80_r25_last40_ordinary_guarded_refill_once.py',
        '575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e')
TERMINAL=('training80-r36-two-new-gap-original-terminal-77cc6b91c336476bba94ae2a6dcae19f/receipt.json',
          '4fee319ef9f6ace51f839a3cf92aa059765511978004d795c3ae929c8abd9e13')
CROSSWALK=('training80-1824-new-gap-incumbent-crosswalk-v1.json',
           '755adf0631fd7e8929b24ea21868f4f76493446abbc70caa44290087888f1c5b')
NODES=('f13r2n11','f14r3n05')


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok,why):
    if not ok:
        raise ValueError(why)


def load():
    source=ROOT/NEXT[0]
    raw=source.read_bytes()
    require(source.resolve(strict=True)==source and digest(raw)==NEXT[1],'frozen controller-budget guard differs')
    budget=runpy.run_path(str(source))
    first=budget['module'](budget['FIRST'])
    r34,early,n02=first['modules']()
    history=budget['module'](LAST40)
    config=history['bound_api']()['load']()
    frozen=budget['controller'](config[1][6]['controller.py'])
    read=budget['read_pinned']
    terminal=json.loads(read(LOCAL/TERMINAL[0],TERMINAL[1]))
    view=terminal['response']
    gap=json.loads(read(LOCAL/CROSSWALK[0],CROSSWALK[1]))['nodes']
    require(terminal['rc']==0 and terminal['remote_outcome']=='RESPONSE_OBSERVED'
            and terminal['authority_verified'] is terminal['remote_received'] is True
            and view['new_GPU']==0 and view['physical_clear'] is False and set(view['nodes'])==set(gap)==set(NODES),
            'exact terminal read scope differs')
    rows,targets={},{}
    for node in NODES:
        actual=view['nodes'][node]
        declared=gap[node]
        old=config[5][node]
        gpu,cpu,files=actual['gpu'],actual['cpu'],actual['worker_records']
        request=declared['last40_prior_request']
        parent=declared['parent']
        snap_name=f"{history['STEM']}-{node}-{request}-reply.json"
        reply_sha=declared.get('prior_reply_SHA256') or {
            'f13r2n11':'a9a98d6ad6e44ed3931e356f3d39d8214b0b4cfb48e10309c01d14f8ce39544a',
            'f14r3n05':'db97d96b3b48d9d9bf67e36b3bf2b2585151baf562dc24562bb2f6c389a8e8a5',
        }[node]
        prior=json.loads(read(LOCAL/snap_name,reply_sha))['transport']
        authorized=prior['response']
        require(prior['rc']==0 and prior['remote_outcome']=='RESPONSE_OBSERVED'
                and request==prior['request']==authorized['request']==old['new_request']
                and authorized['GPU_srun_attempted'] is True and authorized['parent']==old['parent']==parent
                and actual['controller_rows']==actual['queue_rows']==[] and actual['natural_return'] is True
                and actual['original_released_for_fresh_physical_preflight'] is True
                and len(gpu)==9 and gpu[:4]==[declared['numbered_predecessor'],f'training80-dcu-standby-{request[:12]}-{node}','COMPLETED','0:0']
                and gpu[5:7]==[node,early['GPU_TRES']] and gpu[7]<gpu[8]
                and int(gpu[4])>=actual['worker_seconds']==declared['launch_budget_seconds']==authorized['receipt']['worker_seconds']
                and len(cpu)==9 and cpu[:4]+cpu[5:7]+cpu[8:]==authorized['fresh_zero_GRES_accounting']==declared['old_CPU']
                and cpu[5:7]==[node,'cpu=1,mem=256M,node=1'] and cpu[7]<=cpu[8]
                and actual['worker_pid']>0 and actual['worker_start_ticks']>0
                and set(files)=={'cgroup','expected','submission','terminal','worker'}
                and all(re.fullmatch(r'[a-f0-9]{64}',v) for v in files.values())
                and actual['returned_epoch']>=actual['worker_start_epoch']+actual['worker_seconds'],
                'actual terminal or original authenticated last40 authority differs')
        new=uuid5(UUID('4f7a5b44-408e-437c-962d-b1d1fb44925c'),f'{node}/{gpu[0]}/{request}').hex
        seconds=frozen['node_seconds'](node,10800,new)
        require(len(old['pci'])==len(set(old['pci']))==8 and 10500<=seconds<=10800,'original PCI scope or next controller budget differs')
        end=actual['worker_start_epoch']+actual['worker_seconds']
        rows[node]=dict(node=node,parent=parent,previous=gpu[0],request=request,new_request=new,clearance=cpu[:4]+cpu[5:7]+cpu[8:],
                        pinned_GPU=gpu,pinned_terminal=files['terminal'],worker_start_epoch=actual['worker_start_epoch'],
                        worker_seconds=actual['worker_seconds'],worker_SHA256=files['worker'],expected_SHA256=files['expected'],
                        submit_SHA256=files['submission'],cgroup_SHA256=files['cgroup'],previous_worker_pid=actual['worker_pid'],
                        pci=old['pci'],planned_end_epoch=end,prepare_epoch=end-900,renewal_seconds=seconds)
        targets[node]=dict(parent=parent,prior_GPU=gpu[0],prior_CPU=cpu[0],prior_request=request,new_request=new,
                           sealed_original_files=files,old_launch_reply_SHA256=reply_sha,worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300)
    claim=dict(scope=STEM,state='PREPARED_ONLY_NEW_FRESH_PHYSICAL_REQUIRED',source_terminal_SHA256=TERMINAL[1],
               source_gap_SHA256=CROSSWALK[1],last40_ancestry_SHA256=LAST40[1],runtime_guard_SHA256=NEXT[1],
               standby_controller_SHA256=budget['CONTROLLER'][1],targets=targets,launcher=WRAPPER.relative_to(ROOT).as_posix(),
               GPU_per_node=8,max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False,
               unrelated_O01_and_Gemma_nodes_excluded=['f12r2n10','f13r1n12'])
    return budget,first,r34,early,n02,config,rows,claim,raw


def api(material,claim_sha):
    budget,first,r34,early,n02,config,rows,claim,raw=material
    fn=first['_fn']
    tools=dict(early)
    def exact_read(ctx,row,nonce):
        source=early['read_program'](ctx,row,nonce)
        return early['unique'](source,'    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,',
                             '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'
                             '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,')
    tools['read_program']=exact_read
    bind=fn(budget['api'],STEM=STEM,G34=LAST40,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,read_program,read,render,dispatch,observe=bind((first,r34,tools,n02,config,rows,claim),claim_sha)
    old_publish=dispatch.__globals__['publication']
    def publication(c,revision):
        old_publish(c,revision)
        _,publisher=c[3][-2]['modules']()
        require(publisher['git']('cat-file','blob',revision+':'+NEXT[0])==raw,'published budget guard differs')
    return ctx,read_program,read,render,fn(dispatch,publication=publication),observe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=NODES)
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one node and actual predecessor receipt required')
    material=load()
    budget,first,r34,_,_,_,rows,claim,_=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact two last40 successors differ')
    sha=digest(raw)
    ctx,reader,read,render,dispatch,observe=api(material,sha)
    if args.check:
        nodes={}
        for node,row in rows.items():
            r=reader(ctx,row,'5'*32)
            gpu=row['pinned_GPU']
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                   same_UID=True,exact_job_and_step_cgroup=True)).encode()
            program=render(ctx,row,gpu,(row['pinned_terminal'],digest(cgroup)),None)
            first['cgroup_fixture'](program['launch'],row,cgroup,require)
            seconds=budget['spawn_fixture'](program['launch'],row,ctx[1][6]['controller.py'])
            nodes[node]=dict(request=row['new_request'],worker_seconds=seconds,reader_SHA256=digest(r.encode()),
                             rendered_HOST_SHA256=digest(program['launch'].encode()))
        result=dict(state='PREPARED_ONLY',nodes=nodes,actual_HOST_spawn_positive=2,actual_HOST_spawn_vetoes=4,
                    locked_cgroup_and_CPU_positive=2,locked_cgroup_and_CPU_vetoes=4,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

"""Prepare five ordinary replacements, each requiring its own fresh terminal receipt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r80-r64-r66-five-guarded-bf16-v1'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R44 = ('tools/run_training80_r44_four_2053_guarded_successor_once.py',
       'eb321d52a97d27a7b5707b3c0fd2c8bf2e22c525cda2c902923fc25b4f4b91ed')
R64 = ('tools/run_training80_r64_r52_pair_0349_guarded_successor_once.py',
       '4cbdcad7720b5ad9f68356aee7c6c3cfef0486a4f696193fbc95936f2fa77ae6')
R66 = ('tools/run_training80_r66_r53_four_0405_guarded_successor_once.py',
       'f9e146044ccbf64277b0078cf80f3c337a599f5940676e2be8673112b1bdc14c')
FLEET = ('local-only/training80-post-renewal-scheduler-79eafefd25d64b34bc0b45a396b54af1/receipt.json',
         '369a33185fa34672d6876480080827b8412836f15e24aa4421b2abd4d37acbf7')
PROTECTED = {'f13r2n07','f13r2n08','f13r3n13','f13r2n19','f13r2n06','f13r3n04','f14r3n06','f13r1n05',
             'f12r1n18','f12r3n10','f12r2n10','f12r2n11','f13r1n12','f13r1n19'}
R64_FAMILY = 'training80-r64-r52-pair-0349-guarded-bf16-v1'
R66_FAMILY = 'training80-r66-r53-four-0405-guarded-bf16-v1'
FRONTIERS = {
    'f14r3n04': ('1194900.1564','1194900.1563',R64_FAMILY,'3c5491a2739e4e72a5092a5d955759db',
                 'a2c841bb0867faba7799a60c9584acea9d1ba941ec3401c873df1c2935851455',
                 '18fcbd2636913610c2e1d485f3ac3fd6053570f013ec28fe428dab2b9c00de69'),
    'f13r1n07': ('1103606.3590','1103606.3589',R66_FAMILY,'78363c2b924f451ba618a3ca8a4c88cb',
                 '9b267c0785847641b917164fbdddf1307ba9298872b2fd7ed070616041d7c5be',
                 'f00f6ce09aa169c105a7d3d7e5861c4fe3b16147f5628489554efcdee7c562e1'),
    'f13r2n11': ('1103606.3594','1103606.3593',R66_FAMILY,'ef29372d01014701b13627017aa9d993',
                 '2a9594cfe9430fe3e55d66a697f027b8e078c650db2bd5d8516be4ebc125a8d1',
                 '0c06cf81562a4b363d727bfbb5642110e35d695eaf0c417c9e82e6b2ff28f235'),
    'f13r2n05': ('1103606.3592','1103606.3591',R66_FAMILY,'45cf3d695bc242bcacd20a616a858c4e',
                 '939753287759772091303c942a382811a5c8d93bdea48d73245a545df389903d',
                 '4bb5208cd0d91577893f94f3d927ba3fbc2fc98398e78df2ee43c182b650766c'),
    'f12r3n02': ('1194900.1570','1194900.1569',R66_FAMILY,'a261222d8cc44475870eda7e54c97eeb',
                 '006263fa0f0e5f6bfb38bfe73257a737613f51f010576d612574e6365423c2ed',
                 'b7309e127ff46147fcfa6fe860d6fc75b0e9c62ae49a6a8bbd7bfbe0db7aeb6b'),
}
EMPTY = sha256(b'').hexdigest()
REVISION = re.compile(r'[0-9a-f]{40}').fullmatch


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def pin(spec):
    path = ROOT / spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path, 'noncanonical frozen evidence')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1], 'frozen original evidence differs')
    return raw


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def load():
    gateway=module(R44)
    budget,first,r34,early,n02,ctx,_,_=gateway['load']()
    _,_,old64=module(R64)['entry']()[0]()
    _,_,old66=module(R66)['entry']()[0]()
    owners={R64_FAMILY:old64,R66_FAMILY:old66}
    controller=budget['controller'](ctx[1][6]['controller.py'])
    fleet=json.loads(pin(FLEET))
    snap=fleet['response']['snapshot']
    require(fleet['rc']==0 and fleet['authority_verified'] is fleet['remote_received'] is True
            and fleet['remote_outcome']=='RESPONSE_OBSERVED' and snap['numbered_cards']==320
            and set(FRONTIERS).isdisjoint(PROTECTED), 'exact scheduler or exclusions differ')
    rows,targets,routes={},{},{}
    for node,(step,paired,family,obs,obs_sha,reply_sha) in FRONTIERS.items():
        parent=step.split('.')[0]
        material=owners[family]
        incumbent=material[-3][node]
        route=material[5][5][node]
        source=json.loads(pin((f'local-only/{family}-actual-batch-counters-{obs}/receipt.json',obs_sha)))
        [sample]=[item['receipt'] for item in source['receipt_by_batch'] if item['rc']==0
                  and node in (item.get('receipt') or {}).get('nodes',{})]
        worker=sample['nodes'][node]
        request,seconds,start=(worker[key] for key in ('request','worker_seconds','worker_start_epoch'))
        one,two,deltas=(worker[key] for key in ('first','second','count_deltas'))
        require(source['scope']==family and worker['previous']==step and worker['parent']==route['parent']==parent
                and node in snap['no_numbered'][parent] and request==incumbent['new_request']
                and seconds==incumbent['renewal_seconds'] and start+seconds==worker['planned_expiry_epoch']
                and material[-1]['targets'][node]['queue_ownership']=='ADOPTED_ORDINARY58_EXACT_ANCESTOR'
                and sample['stage']=='COMPLETE' and sample['GPU_steps_created']==0
                and len(one)==len(two)==len(deltas)==worker['growth_DCU']==8 and set(one)==set(two)==set(deltas)
                and all(two[p]['calls']-one[p]['calls']==deltas[p]>0 and two[p]['epoch']>one[p]['epoch'] for p in one),
                'historical exact worker, eight-card evidence or vacancy differs')
        launch=json.loads(pin((f'local-only/{family}-{node}-{request}-reply.json',reply_sha)))['transport']
        body=launch['response']
        cpu=body['fresh_zero_GRES_accounting']
        require(launch['rc']==0 and launch['authority_verified'] is launch['remote_received'] is True
                and launch['remote_outcome']=='RESPONSE_OBSERVED' and launch['request']==body['request']==request
                and body['GPU_srun_attempted'] is body['mutex_acquired'] is body['zero_GRES_call_attempted'] is True
                and body['probe_step']==paired==cpu[0] and body['parent']==parent and body['node']==node
                and body['probe_stderr_SHA256']==EMPTY and body['receipt']['worker_seconds']==seconds
                and cpu[:6]==[paired,'training80-r25-fresh-'+request[:12],'COMPLETED','0:0',node,'cpu=1,mem=256M,node=1'],
                'paired original CPU or exact launch differs')
        new=uuid5(UUID('57b51ef5-4c3a-4f47-94bd-6c6443160b6a'),f'{node}/{step}/{request}').hex
        renewal=controller['node_seconds'](node,10800,new)
        require(new!=request and 10500<=renewal<=10800, 'new exact request or worker budget differs')
        rows[node]=dict(node=node,parent=parent,previous=step,request=request,new_request=new,clearance=cpu,
                        cgroup_SHA256=None,worker_start_epoch=start,worker_seconds=seconds,planned_end_epoch=start+seconds,
                        prepare_epoch=start+seconds-900,pci=sorted(one),previous_worker_pid=worker['worker_PID'],
                        renewal_seconds=renewal,worker_SHA256=worker['worker_SHA256'],
                        expected_SHA256=worker['expected_SHA256'],submit_SHA256=worker['submit_SHA256'])
        routes[node]=route
        targets[node]=dict(parent=parent,original_GPU=step,original_CPU=paired,original_request=request,new_request=new,
                           original_counter_SHA256=obs_sha,original_launch_SHA256=reply_sha,worker_SHA256=worker['worker_SHA256'],
                           expected_SHA256=worker['expected_SHA256'],submission_SHA256=worker['submit_SHA256'],
                           new_worker_seconds=renewal,Slurm_time_cap_seconds=renewal+300,
                           queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR')
    require(len({row['new_request'] for row in rows.values()})==5, 'new request identity collision')
    require(set(rows).isdisjoint(ctx[0]['local_holds']()), 'current formal work owns an R80 target')
    ctx=(*ctx[:5],routes,ctx[6])
    claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),state='PREPARED_ONLY_ACTUAL_C0_AND_FRESH_DEVICE_REQUIRED',
               scheduler_snapshot_SHA256=FLEET[1],targets=targets,priority_scopes_preserved=sorted(PROTECTED),
               parent_serial_order=list(FRONTIERS),controller_SHA256=budget['CONTROLLER'][1],source_pins=dict((R44,R64,R66)),
               post_publication_predecessor_digest_required=True,ready_GPU_count=0,GPU_per_node=8,
               max_concurrent_send=1,max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False)
    return gateway,budget,first,r34,early,n02,ctx,rows,claim


def runtime(material,claim_sha,revision):
    gateway,budget,first,r34,early,n02,ctx,rows,claim=material
    clone=first['_fn']
    old_source=early['read_program']

    def published_reader(context,row,nonce):
        source=old_source(context,row,nonce)
        return f'# exact-published-R80-revision {revision}\n'+source

    reader=dict(early,read_program=published_reader)
    guarded=clone(budget['api'],STEM=STEM,G34=gateway['R29'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    context,source,read,render,dispatch,observe=guarded((first,r34,reader,n02,ctx,rows,claim),claim_sha)
    inherited=dispatch.__globals__['publication']

    def publication(actual,actual_revision):
        require(actual_revision==revision and REVISION(revision), 'exact publication differs from the read receipt scope')
        inherited(actual,actual_revision)
        _,publisher=actual[3][-2]['modules']()
        for spec in (R44,R64,R66):
            require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                    'published prior guarded owner differs')

    return context,source,read,render,clone(dispatch,publication=publication),observe,publication


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--show-run-order',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=FRONTIERS)
    parser.add_argument('--terminal-receipt',nargs=2)
    parser.add_argument('--published-revision')
    args=parser.parse_args()
    local=args.check or args.show_run_order
    require(bool(local or args.node) and bool(args.publication)==bool(args.terminal_receipt)
            and bool(args.observe_predecessor)==bool(args.published_revision), 'exact node, revision and receipts required')
    revision=args.publication or args.published_revision or '0'*40
    require(REVISION(revision), 'need exact 40-hex publication')
    material=load()
    _,budget,first,_,_,n02,_,rows,claim=material
    raw=CLAIM.read_bytes()
    require(json.loads(raw)==claim,'frozen five-node scope differs')
    sha=sha256(raw).hexdigest()
    if args.show_run_order:
        result=dict(state='PREPARED_ONLY',max_concurrent_send=1,replay_allowed=False,
                    ordered=[{key:row[key] for key in ('node','parent','previous','new_request','renewal_seconds')}
                             for row in rows.values()])
    else:
        ctx,source,read,render,dispatch,observe,publication=runtime(material,sha,revision)
        if args.check:
            other=runtime(material,sha,'1'*40)[1]
            fixtures={}
            for node,row in rows.items():
                reader=source(ctx,row,'a'*32)
                require(reader!=other(ctx,row,'a'*32),'reader bytes were not bound to publication')
                compile(reader,'<publication-scoped original reader fixture>','exec')
                account=[row['previous'],'training80-dcu-standby-'+row['request'][:12]+'-'+node,'COMPLETED','0:0',
                         str(row['worker_seconds']),node,'cpu=8,gres/dcu=8,mem=16G,node=1',
                         '2000-01-01T00:00:00','2000-01-01T03:00:00']
                cg=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],
                    worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
                launch=render(ctx,row,account,('a'*64,sha256(cg).hexdigest()),None)['launch']
                first['cgroup_fixture'](launch,row,cg,require)
                seconds=budget['spawn_fixture'](launch,row,ctx[1][6]['controller.py'])
                fixtures[node]=dict(request=row['new_request'],seconds=seconds,reader_SHA256=sha256(reader.encode()).hexdigest())
            n02['fixture_holds']()
            result=dict(state='PREPARED_ONLY',fixtures=fixtures,prior_cgroup_CPU_vetoes=10,
                        runtime_budget_vetoes=10,distinct_publication_reader_fixtures=5,Git=0,SSH=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            publication(ctx,revision)
            require(args.node not in ctx[0]['local_holds'](),'current formal work owns this target')
            result=read(ctx,rows[args.node],args.observe_predecessor)
        elif args.observe_original:
            result=observe(ctx,rows[args.node])
        else:
            result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

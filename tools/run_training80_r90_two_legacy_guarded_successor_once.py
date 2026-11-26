"""Guard the two R90 legacy continuations after their exact natural return."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
WRAPPER=Path(__file__).resolve()
STEM='training80-r90-two-legacy-guarded-bf16-v1'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
ORDER=('f13r1n08','f13r4n17')
V2=('tools/run_training80_legacy_three_fresh_owner_bf16_v2_once.py',
    '16eb4c737c78270bcd89866a38c8feaf38d44471e5a9289b3ee2c97255433fc6')
V1=('tools/run_training80_legacy_three_fresh_owner_bf16_once.py',
    'de55ffda601984e9222fc315535cdd957207f862a82ba8610d5f9cd8ee40ced6')
V2_CLAIM=('coordination/tasks/training80-legacy-three-fresh-owner-bf16-v2.json',
          'd338899e57ab281b36d31446a740e1f83db9edf5cfb6b24639b8fb563f0053d2')
R84=('tools/run_training80_r84_seventeen_0930_guarded_successor_once.py',
     'fc7443510c0570c9be042219000dd82ec4914055eaad241b3916bb129ed40077')
READER=('local-only/render_training80_r90_1150_legacy_r83_predecessors_once.py',
        'bb232ab62389f3021ff1fa705c00a20fa0a0119c9ca8f80e262ed58c2af98ce2')
ORIGINAL='local-only/training80-r90-two-original-legacy-natural-returns-c9d87ce8e1834d71bf0257de270f64a9'
ORIGINAL_PINS={
    'intent':'a92aaea022b158a39086c08e930324c58f87d3a652bfc8256022a6d7e0396dc2',
    'send':'544c81837daeed83cab37fe2942bab4beb563b84c5ff8676fcc781701cf4c02f',
    'receipt':'1aa8edbeba6452d867ed2293bcf481660a20e3d01d4eaf5c8308e05d387c4b60',
}
MARKERS={
    'f13r1n08':('e1c0da749bde3aaaa6b3096fa393df80a5cd4d028e41d14e7fcb83d7baa26ee8',
              'd733743f46e110ccddb7a90aee673f755d3003f3e90870417ade9c1555de02e7',
              '81f88bcf041e228a6e6fb4d25988addf11e7b60e6e3bad0095a0afcc4840aa00'),
    'f13r4n17':('d5127994423dd8af1db7e94d5ffc6b98b65c5c8a6dd8b224221274abc4ab5cb5',
              '690665b75a80f6b23ba2ec65d6ab0e0b7d7dcf4fab8bbd9c7b0fc6a9c8d51bd8',
              'd60cdbef89ce102b2a1a06306b50e6de9dbc7e418b027440d48ee61b5274bb77'),
}


def clone(function,**changes):
    return FunctionType(function.__code__,dict(function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def entry():
    require(sha256((ROOT/V1[0]).read_bytes()).hexdigest()==V1[1],'exact legacy guard differs')
    legacy=runpy.run_path(str(ROOT/V1[0]))
    pin,module=(legacy[k] for k in ('pin','module'))

    def load():
        version2=module(V2)
        base,_,budget,context,prior,prior_claim=version2['entry']()[0]()
        require(json.loads(pin(V2_CLAIM))==prior_claim,'published prior legacy owner differs')
        evidence=module(READER)
        roll,read_core,read_base,known,_=evidence['load']()
        proof_api=module(R84)
        intent,send,packet=(json.loads(pin((f'{ORIGINAL}/{member}.json',digest))) for member,digest in ORIGINAL_PINS.items())
        body=packet['response']
        read_id=body['request']
        source=evidence['render'](roll,read_core,read_base,[known[n] for n in ORDER],read_id)
        require(packet['rc']==0 and packet['authority_verified'] is packet['remote_received'] is packet['payload_flushed'] is True
                and packet['remote_outcome']=='RESPONSE_OBSERVED' and body['physical_admissions']==body['Slurm_mutations']==0
                and intent['request']==send['request']==packet['request']==body['read_id']==read_id
                and send['payload_sha256']==packet['payload_sha256'] and intent['program_sha'] in {
                    sha256(source.encode()).hexdigest(),sha256(source.replace('\n','\r\n').encode()).hexdigest()}
                and set(body['parents'])=={'1103606','1194900'},'authenticated two-node predecessor read differs')
        first=module(budget['FIRST'])
        r34=module(first['R34'])
        n02=module(first['N02'])
        controller=budget['controller'](context[1][6]['controller.py'])
        rows,targets={},{}
        for node in ORDER:
            old,owner=known[node],prior[node]
            parent,step,request=(old[k] for k in ('parent','previous','request'))
            branch=body['parents'][parent]['response']
            [item]=branch['targets']
            require(branch['read_id']==read_id and branch['Slurm_mutations']==branch['physical_admissions']==0
                    and branch['plans_only'] is branch['requires_fresh_independent_device_and_dispatch_checks'] is True
                    and owner['parent']==parent and owner['new_request']==request
                    and owner['renewal_seconds']==old['worker_seconds'],'prior legacy role, parent or budget differs')
            gpu,files,pid=proof_api['proof_row'](item,old)
            prefix=f"{version2['STEM']}-{node}-{request}"
            prior_markers={f'{prefix}-{kind}.json':digest for kind,digest in zip(('intent','send','reply'),MARKERS[node],strict=True)}
            start,sent,reply=[json.loads(pin((f'local-only/{name}',digest))) for name,digest in prior_markers.items()]
            wire=reply['transport']
            require({k:v for k,v in reply.items() if k!='transport'}==start
                    and start['request']==sent['request']==wire['request']==request
                    and start['node']==node and start['claim_SHA256']==V2_CLAIM[1] and start['replay_allowed'] is False
                    and start['GPU_requested']==8 and wire['payload_sha256']==sent['payload_sha256']
                    and wire['rc']==0 and wire['authority_verified'] is wire['remote_received'] is wire['payload_flushed'] is True
                    and wire['response']['GPU_srun_attempted'] is True,'closed legacy predecessor send chain differs')
            new=uuid5(UUID('021bdc75-a48c-4ba0-a57c-50d5e06e2ecc'),f'{node}/{step}/{request}').hex
            seconds=controller['node_seconds'](node,10800,new)
            require(new not in {read_id,request,owner['request']} and 10500<=seconds<10800,'separate new request or budget differs')
            rows[node]=dict(old,new_request=new,pinned_GPU=gpu,pinned_terminal=files['terminal'],
                            clearance=old['old_CPU'],prepare_epoch=old['planned_end_epoch']-900,
                            submit_SHA256=files['submission'],previous_worker_pid=pid,renewal_seconds=seconds,
                            old_owner=owner,prior_markers=prior_markers)
            targets[node]=dict(parent=parent,original_GPU=step,original_CPU=old['old_CPU'][0],original_request=request,
                               new_request=new,old_worker_files=files,old_legacy_prior_markers=prior_markers,
                               old_v1_request=owner['previous_v1']['request'],actual_prior_seconds=old['worker_seconds'],
                               new_worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300,
                               original_role='VERIFIED_LEGACY_FRESH_OWNER_BF16_CONTINUATION')
        require(len({r['new_request'] for r in rows.values()})==len(ORDER),'new requests overlap')
        sources=(V1,V2,V2_CLAIM,R84,budget['FIRST'],legacy['R35_BUDGET'],version2['R45'])
        claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),targets=targets,
                   state='PREPARED_ONLY_EACH_NODE_REQUIRES_FRESH_PHYSICAL_AND_PRIORITY',
                   original_read_transport_SHA256=ORIGINAL_PINS,original_reader_SHA256=READER[1],source_pins=dict(sources),
                   GPU_per_node=8,actual_physical_admissions=0,ledger_mutations=0,max_concurrent_send=1,
                   max_invocations_per_node=1,replay_allowed=False)
        return base,budget,first,r34,n02,context,rows,claim,sources

    def runtime(material,claim_sha):
        base,budget,first,r34,n02,context,rows,claim,sources=material
        owner_context=(*context[:5],{node:row['old_owner'] for node,row in rows.items()},context[6])
        api=clone(budget['api'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        ctx,source,read,inherited_render,send,observe=api((first,r34,base,n02,owner_context,rows,claim),claim_sha)
        version2,shadow=module(V2),module(sources[-1])
        terminal_identity=observe.__globals__['terminal_identity']

        def fence(ordinary,authority,row):
            old=row['old_owner']
            version2['no_v1_marker'](old['previous_v1']['scope'])
            actual=base['exact_no_other_claims'](ordinary,authority,old)
            expected={(v['parent'],v['ancestor'],v['claim'],v['wave'],v['group']) for v in old['ledger_rows']}
            require(len(actual)==2 and set(actual)==expected,'prior legacy booking has another writer')
            journal=LOCAL/'training80-1738-ordinary-renewal-dispatch'
            require(journal.is_dir() and not journal.is_symlink()
                    and not any(os.path.lexists(journal/f"{row['node']}-{step}-{slot}.json")
                                for step in (*old['journal_steps'],row['previous']) for slot in range(3)),
                    'old or current ordinary slot still reserved')
            seen={p.name for p in LOCAL.glob(f"training80-*-{row['node']}-*.json") if re.search(r'-(intent|send|reply)\.json$',p.name)}
            require(seen==set(row['prior_markers']),'a competing singleton requires reconciliation')
            for name,digest in row['prior_markers'].items():
                pin((f'local-only/{name}',digest))

        def render(ctx,row,account,pins,revision):
            require(account==row['pinned_GPU'] and pins==(row['pinned_terminal'],row['cgroup_SHA256']),
                    'fresh predecessor differs from authenticated terminal/cgroup bytes')
            program=inherited_render(ctx,row,account,pins,revision)
            old=row['old_owner']
            launch=base['unique'](program['launch'],'\ndef run_guarded():',
                    f"\nOLD_WRITER_MEMBERS={old['old_members']!r}\nOLD_ACCOUNTING={old['old_account']!r}\n"
                    f"OLD_PROBE_BOOT={old['boot']!r}\n"+legacy['LOCKED_WRITERS']+'\n\ndef run_guarded():')
            anchor="            prior_cgroup_path = original_dir / NODE / 'worker-cgroup.json'"
            launch=base['unique'](launch,anchor,'            old_writer_closure()\n'+anchor)
            launch=base['unique'](launch,'            probe_sample(sample)',
                    '            require(sample["boot_SHA256"]==OLD_PROBE_BOOT, "target boot changed since independent observation")\n'
                    '            probe_sample(sample)')
            program['launch']=shadow['guarded_shadow'](launch,old['previous_v1'],row['node'],base['unique'])
            compile(program['launch'],'<same-lock prior legacy/cgroup/CPU and one fresh send>','exec')
            return program

        old_publication=send.__globals__['publication']

        def publication(ctx,revision):
            old_publication(ctx,revision)
            _,publisher=ctx[3][-2]['modules']()
            for spec in sources:
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published historical owner or fresh guard source differs')

        return ctx,source,read,render,clone(send,render=render,publication=publication,exact_no_other_claims=fence),clone(observe,render=render),publication,fence,terminal_identity

    return load,runtime


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    modes=cli.add_mutually_exclusive_group(required=True)
    for mode in ('--check','--check-bookings','--show-run-order','--observe-original'):
        modes.add_argument(mode,action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    cli.add_argument('--node',choices=ORDER)
    cli.add_argument('--published-revision')
    cli.add_argument('--terminal-receipt',nargs=2)
    args=cli.parse_args()
    require((args.check or args.check_bookings or args.show_run_order or args.node)
            and bool(args.publication)==bool(args.terminal_receipt)
            and bool(args.observe_predecessor)==bool(args.published_revision),'exact node and fresh receipt/revision required')
    load,api=entry()
    material=load()
    _,budget,first,_,_,_,rows,claim,_=material
    raw=CLAIM.read_bytes()
    require(json.loads(raw)==claim,'frozen two-node claim differs')
    ctx,source,read,render,send,observe,publish,fence,_=api(material,sha256(raw).hexdigest())
    if args.check:
        checked={}
        for node,row in rows.items():
            reader=source(ctx,row,'1'*32)
            launch=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)['launch']
            require(launch.index('old_writer_closure()\n')<launch.index('prior_cgroup_path =')
                    <launch.index('outcome["zero_GRES_call_attempted"] = True')
                    <launch.index('target boot changed since independent observation')<launch.index('save(audit / "gpu-send-intent.json"'),
                    'old custody and new attempt ordering differs')
            fixture=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],
                                    worker_start_ticks=1,same_UID=True,exact_job_and_step_cgroup=True)).encode()
            fixture_row=dict(row,cgroup_SHA256=sha256(fixture).hexdigest())
            fixture_launch=render(ctx,fixture_row,row['pinned_GPU'],(row['pinned_terminal'],fixture_row['cgroup_SHA256']),None)['launch']
            first['cgroup_fixture'](fixture_launch,fixture_row,fixture,require)
            seconds=budget['spawn_fixture'](launch,row,ctx[1][6]['controller.py'])
            checked[node]=dict(request=row['new_request'],seconds=seconds,reader_SHA256=sha256(reader.encode()).hexdigest())
        response=dict(state='PREPARED_ONLY',targets=checked,Slurm=0,SCNet=0,GPU=0,Git=0)
    elif args.show_run_order:
        response=dict(order=list(ORDER),targets=claim['targets'],replay_allowed=False)
    elif args.check_bookings:
        for row in rows.values():
            fence(ctx[2],ctx[3],row)
        require(not set(ORDER)&ctx[0]['local_holds'](),'current project priority owns a target')
        response=dict(state='LOCAL_BOOKINGS_CLEAR',targets=list(ORDER),SCNet=0,GPU=0,Git=0)
    elif args.observe_predecessor:
        row=rows[args.node]
        publish(ctx,args.published_revision)
        fence(ctx[2],ctx[3],row)
        require(args.node not in ctx[0]['local_holds'](),'current project priority owns the node')
        response=read(ctx,row,args.observe_predecessor)
    elif args.observe_original:
        response=observe(ctx,rows[args.node])
    else:
        response=send(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(response,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

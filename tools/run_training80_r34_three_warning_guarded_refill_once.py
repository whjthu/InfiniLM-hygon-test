"""Resume three exact original warning no-sends through their existing distinct guards."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from types import FunctionType, SimpleNamespace
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r34-three-warning-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r34_three_warning_guarded_refill_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
PARENT='1103606'
EXTERN=PARENT+'.extern'
ORD='f13r4n05'
NODES=(ORD,'f13r2n03','f13r2n07')
ANCHORS={ORD:('1103606.3235','1103606.3322','26b8abf6af6a4da2b3916d091188e94f'),
         'f13r2n03':('1103606.3193','1103606.3302','f88b267bfa7d49c58866b4650abc849d'),
         'f13r2n07':('1103606.3203','1103606.3303','c2acb1b023c34240af5826d9db100064')}
ORDINARY='tools/run_training80_r25_remaining_ordinary_guarded_refill_once.py'
ORDINARY_SHA='020650b28165ca16ac94c8d937a466b3bb47a34ff3ce840fc2b402acae30dcc5'
SPECIAL='tools/run_training80_r27_six_specialist_guarded_refill_once.py'
SPECIAL_SHA='9f941399241fb83d968217e0e0e13df0683a0464bd169bd3c8daa6f787210a18'
PROOF=LOCAL/'training80-r34-three-prior-warning-steps-31d40937fbd14d9ebf4eadf7754987f0.json'
PROOF_SHA='624a1136a08f18c0e1c9a390628c9da6c6d74c645c6f0c15fd9e9797b06c28dc'
OLD_WARNING='819487f21be5e7eeffc86fdd45d5d28f64adbecc596ef82ba176c8a5bc4b6799'
OLD_DETAIL='65f3a77095fdcab49b34702824d24e5e44299f56107c36f5985aec601edb2c5b'
NAMESPACE=UUID('18aaa3b7-a192-4b58-a15f-f842cae6902b')
GPU_TRES='cpu=8,gres/dcu=8,mem=16G,node=1'
ZERO='cpu=1,mem=256M,node=1'
HOST_CHECK=r'''
def verify_old_no_send():
    out=RESULTS/OLD_SCOPE/'outcome.json'
    require(out.resolve(strict=True)==out and out.stat().st_uid==os.getuid(),'previous no-send owner or path differs')
    with out.open('rb') as before:
        sealed_outcome_bytes=before.read(8193)
    require(0<len(sealed_outcome_bytes)<=8192 and sha256(sealed_outcome_bytes).hexdigest()==OLD_DURABLE_SHA
            and json.loads(sealed_outcome_bytes)==OLD_BODY,'sealed original no-GPU-send outcome differs')
    require(not os.path.lexists(RESULTS/('training80-dcu-standby-'+OLD_REQUEST)), 'original GPU namespace exists')
    require(account(OLD_CPU[0])==OLD_CPU,'actual old CPU C0 or End changed')
'''


def digest(data):
    return sha256(data).hexdigest()


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,sha=None):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'noncanonical source or receipt path')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and (sha is None or digest(raw)==sha),'source or receipt bytes differ')
    return raw


def substitute(text,old,new):
    require(text.count(old)==1,'deployed guard insertion differs')
    return text.replace(old,new,1)


def copy_function(fn,env):
    copied=FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
    copied.__kwdefaults__=fn.__kwdefaults__
    return copied


def originals():
    evidence=json.loads(pin(PROOF,PROOF_SHA))
    source=evidence['observed']
    require(evidence['rc']==0 and source['slurm_operations']==0 and source['raw_exported'] is False and source['parent']==PARENT,
            'original read-only authority differs')
    targets={}
    for item in source['targets']:
        node=item['node']; gpu,zero,old=ANCHORS[node]
        [previous]=item['original_ordinary_rows']; [cpu]=item['prior_cpu_rows']
        old_scope=('training80-r25-remaining-ordinary-first6-guarded-bf16-v1' if node==ORD else 'training80-r27-six-specialist-guarded-bf16-v1')+f'-{node}-{old}'
        wire=json.loads(pin(LOCAL/f'{old_scope}-reply.json',item['prior_reply_SHA256']))['transport']; auth=wire['response']
        ctrl=item['matching_scontrol_step_ids']; queue=item['matching_squeue_step_ids']
        require(node not in targets and item['prior_request']==wire['request']==auth['request']==old
                and wire['remote_outcome']=='RESPONSE_OBSERVED' and wire['rc']==0
                and wire['authority_verified'] is wire['remote_received'] is True
                and auth['node']==node and auth['parent']==PARENT and auth['state']=='GPU_NOT_SENT'
                and auth['probe_rc']==0 and auth['probe_stderr_SHA256']==OLD_WARNING and auth['detail_SHA256']==OLD_DETAIL
                and auth['GPU_srun_attempted'] is False and auth['zero_GRES_call_attempted'] is True
                and item['durable_match'] is True and re.fullmatch(r'[a-f0-9]{64}',item['durable_SHA256'])
                and item['prior_gpu_namespace_exists'] is False and item['prior_gpu_label_rows']==[]
                and item['may_prepare_distinct_zero_GRES'] is False and ctrl==queue==[EXTERN]
                and previous[:1]==[gpu] and previous[2:6]==['COMPLETED','0:0',node,GPU_TRES]
                and cpu[:1]==[zero] and cpu[2:6]==['COMPLETED','0:0',node,ZERO]
                and all(re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',r[6]) for r in (previous,cpu)),
                'original no-send, terminal or parent-only context differs')
        label=('training80-r25-fresh-' if node==ORD else 'training80-g4-fresh-')+old[:12]
        require(cpu[1]==label,'original warning step was from a different launcher')
        targets[node]=dict(node=node,request=uuid5(NAMESPACE,f'{PROOF_SHA}/{node}/{old}').hex,old=old,old_scope=old_scope,auth=auth,
                           previous=previous,cpu=cpu,durable=item['durable_SHA256'],original_reply_SHA256=item['prior_reply_SHA256'],
                           before=dict(scontrol=ctrl,squeue=queue))
    requests={row['request'] for row in targets.values()}
    require(set(targets)==set(NODES) and len(requests)==3 and not requests&{r['old'] for r in targets.values()},'new exact scopes overlap')
    return targets


def expected_claim(selected):
    return dict(scope=STEM,state='PREPARED_ONLY_DISTINCT_FRESH_DEVICE_ADMISSION_REQUIRED',launcher=WRAPPER.relative_to(ROOT).as_posix(),
                original_probe_proof_SHA256=PROOF_SHA,old_warning_SHA256=OLD_WARNING,GPU_per_node=8,Slurm_cap_seconds=11100,
                max_invocations_per_node=1,replay_allowed=False,ledger_mutations=0,warning_policy='ALL_NONEMPTY_STDERR_BLOCKS_GPU',
                targets={n:dict(parent=PARENT,previous_GPU=r['previous'][0],previous_warning_CPU=r['cpu'][0],request=r['request'],
                          rejected_request=r['old'],original_reply_SHA256=r['original_reply_SHA256'],
                          original_parent_context=r['before'],original_numeric_target_steps=[]) for n,r in selected.items()})


def add_guard(program,target):
    bind=dict(OLD_REQUEST=target['old'],OLD_SCOPE=target['old_scope'],OLD_CPU=target['cpu'],OLD_DURABLE_SHA=target['durable'],
              OLD_BODY=target['auth'],ORIGINAL_PARENT_CONTEXT=target['before'])
    script=''.join(f'{k}={v!r}\n' for k,v in bind.items())+program['launch']
    script=substitute(script,'\ndef run_guarded():',HOST_CHECK+'\ndef run_guarded():')
    label='training80-dcu-standby-'+target['old'][:12]+'-'+target['node']
    script=substitute(script,'([old_label], [gpu_label])',f'([old_label], [gpu_label], [{label!r}])')
    script=substitute(script,'            check_previous(account(PREVIOUS))',
                      '            check_previous(account(PREVIOUS))\n            verify_old_no_send()')
    script=substitute(script,'    current = {}\n','    parent_context = {"scontrol": [], "squeue": []}\n    current = {}\n')
    for source,expression in (('scontrol','item["NodeList"]'),('squeue','expression')):
        marker='        if step in (PARENT + ".batch", PARENT + ".extern"):\n            continue\n'
        marker+=f'        require(STEP.fullmatch(step) and hosts({expression}) <= parent_nodes,'
        replacement=marker.replace('            continue\n',f'            parent_context[{source!r}].append(step)\n            continue\n')
        script=substitute(script,marker,replacement)
    script=substitute(script,'in_squeue=OBSERVER_STEP in seen_queue)',
                      "in_squeue=OBSERVER_STEP in seen_queue,parent_context=parent_context,numeric_target_steps={'scontrol':sorted(current),'squeue':sorted(seen_queue)})")
    script=substitute(script,'            scope()\n',
                      "            outcome['original_observed_parent_context'] = ORIGINAL_PARENT_CONTEXT\n"
                      "            outcome['current_scheduler_before'] = scope()\n")
    script=substitute(script,'            scope(own=step)\n',"            outcome['current_scheduler_after_probe'] = scope(own=step)\n")
    require(script.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<script.index('            verify_old_no_send()')
            <script.index('outcome["zero_GRES_call_attempted"] = True')
            <script.index('require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,'),
            'existing locked empty-stderr guard order changed')
    compile(script,'<exact R34 strict successor>', 'exec')
    return dict(program,launch=script)


def ordinary(selected,claim_sha):
    pin(ROOT/ORDINARY,ORDINARY_SHA)
    main=runpy.run_path(str(ROOT/ORDINARY))
    old=main['bound_api']();context=old['load']()
    env=dict(old,__file__=str(Path(__file__).resolve()),STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,SPEC={ORD:selected[ORD]})
    for k,fn in old.items():
        if isinstance(fn,FunctionType) and fn.__globals__ is old:
            env[k]=copy_function(fn,env)
    prior_render=env['render']
    spec=context[-2][ORD]; target=selected[ORD]
    require((spec['previous'],spec['new_request'])==(target['previous'][0],target['old']) and target['previous'][1]
            ==f"training80-dcu-standby-{spec['request'][:12]}-{ORD}",'ordinary original worker differs')
    context=(*context[:-2],{ORD:dict(spec,new_request=target['request'])},claim_sha)

    def render(ctx,row,acct,old_terminal,rev):
        require(acct[:4]+acct[5:7]+acct[8:]==target['previous'],'fresh ordinary accounting End differs from original')
        return add_guard(prior_render(ctx,row,acct,old_terminal,rev),target)

    def publication(ctx,rev):
        verify_own_publication(ctx[3],rev)
        old['publication'](ctx,rev)

    env.update(render=render,publication=publication)
    return env,context,context[-2][ORD]


def verify_own_publication(plan,revision):
    require(Path(__file__).resolve()==WRAPPER and re.fullmatch(r'[a-f0-9]{40}',revision),'exact installed root publication required')
    _,publisher=plan[-2]['modules']()
    for path,sha in ((WRAPPER,None),(CLAIM,None),(ROOT/ORDINARY,ORDINARY_SHA),(ROOT/SPECIAL,SPECIAL_SHA)):
        ref=revision+':'+path.relative_to(ROOT).as_posix()
        require(publisher['git']('cat-file','blob',ref)==pin(path,sha),'published three-node guard bytes differ')


def specialist(node,selected):
    pin(ROOT/SPECIAL,SPECIAL_SHA)
    old=runpy.run_path(str(ROOT/SPECIAL))
    base,config,claim,_,prior,clearance,legacy=old['load']()
    target=selected[node]
    very_old=prior[node]['accounting']
    require(very_old[:4]+very_old[5:7]+very_old[8:]==target['previous'] and claim['targets'][node]['request']==target['old'],
            'protected R27 worker and earlier physical receipt differ')
    claim=dict(claim,targets={node:dict(claim['targets'][node],request=target['request'])})
    bound=copy_function(old['bound'],dict(old['bound'].__globals__,STEM=STEM))
    render=copy_function(old['render'],dict(old['render'].__globals__,bound=bound))
    return old,base,config,claim,prior,clearance,legacy,bound,render


def extern_fixture(program):
    tree=ast.parse(program)
    [scope]=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scope']
    test=compile(ast.Module(body=[scope],type_ignores=[]),'<actual numeric gate and extern projection>','exec')
    node=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='NODE')
    parent_nodes=set([node]+[f'f99r1n{i:02d}' for i in range(1,40)])
    def fields(line):return dict(x.split('=',1) for x in line.split())
    for conflict in (None,'scontrol','squeue'):
        def query(*args):
            if args[:3]==('scontrol','show','job'):
                return [f'JobId={PARENT} JobState=RUNNING UserId=project(154459) NumNodes=40 AllocTRES=gres/dcu=320 NodeList=all']
            if args[:3]==('scontrol','show','hostnames'):
                return sorted(parent_nodes)
            if args[:3]==('scontrol','show','step'):
                return [f'StepId={EXTERN} NodeList=all']+([f'StepId={PARENT}.9876 NodeList={node}'] if conflict=='scontrol' else [])
            require(args[0]=='squeue','unexpected fixture command')
            return [f'{EXTERN}|all']+([f'{PARENT}.9876|{node}'] if conflict=='squeue' else [])
        old='observer'
        env=dict(PARENT=PARENT,NODE=node,STEP=re.compile(PARENT+r'\.[0-9]+'),HOSTNAME=re.compile(r'f[0-9]{2}r[0-9]+n[0-9]{2}'),
                 ZERO=ZERO,LABEL='probe',os=SimpleNamespace(getuid=lambda:154459),fields=fields,query=query,require=require,
                 OBSERVER_STEP=old,OBSERVER_LABEL='old label',TERMINAL={'COMPLETED'},ALLOW_OBSERVER_OVERLAP=False,
                 account=lambda _:[old,'old label','COMPLETED','0:0','all','cpu=40,mem=10G,node=40','actual End'])
        exec(test,env)
        try: evidence=env['scope']()
        except ValueError:
            require(conflict is not None,'plain parent extern unexpectedly vetoed')
        else:
            require(conflict is None and evidence['parent_context']==dict(scontrol=[EXTERN],squeue=[EXTERN])
                    and evidence['numeric_target_steps']==dict(scontrol=[],squeue=[]),'numeric hold passed or parent extern was lost')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=NODES)
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args=parser.parse_args()
    require(args.check or args.node,'one exact node required')
    require(not args.observe_predecessor or args.node==ORD,'the two specialist gates recheck their exact original terminal under the lock')
    require(bool(args.terminal_receipt)==bool(args.publication and args.node==ORD),'new terminal receipt required for ordinary node only')
    selected=originals()
    raw=pin(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==expected_claim(selected),'exact three-node claim differs')
    result=[]
    for node in ([args.node] if args.node else list(NODES)):
        if node==ORD:
            api,ctx,row=ordinary(selected,digest(raw))
            if args.check:
                [source_slot]=[selected[node]['previous']]
                detailed=source_slot[:4]+[str(row['worker_seconds'])]+source_slot[4:6]+['2026-11-25T05:00:00',source_slot[6]]
                sample=api['render'](ctx,row,detailed,'a'*64,None)
                extern_fixture(sample['launch'])
                api['host_fixture'](sample['launch'],row,detailed,ctx[1][6])
                value=dict(node=node,request=row['new_request'],compile_SHA256=digest(sample['launch'].encode()),after_lock_extern_fixtures=3)
            elif args.observe_predecessor:
                value=api['observe_predecessor'](ctx,row,args.observe_predecessor)
            elif args.observe_original:
                value=api['observe_original'](ctx,row)
            else:
                value=api['dispatch'](ctx,row,args.publication,args.terminal_receipt)
        else:
            old,base,conf,claim,prior,clearance,legacy,bound,render=specialist(node,selected)
            spec=claim['targets'][node]
            program=add_guard(render(base,conf,claim,prior,clearance,node,args.publication),selected[node])
            if args.observe_original:
                value=bound(base,'observe',node,spec)(node,program,conf)
            else:
                old['require_no_slots'](base['JOURNAL'],node,legacy[node],(1,2))
                old['require_no_slots'](base['JOURNAL'],node,spec['original_GPU'],(0,1,2))
                if args.check:
                    extern_fixture(program['launch'])
                    old['offline_fixture'](program['launch'],node,prior[node]['accounting'])
                    old['offline_staging_fixture'](program['launch'],conf[6])
                    value=dict(node=node,request=spec['request'],compile_SHA256=digest(program['launch'].encode()),after_lock_extern_fixtures=3)
                else:
                    anchor=runpy.run_path(str(ROOT/ORDINARY))['bound_api']()['load']()[3]
                    verify_own_publication(anchor,args.publication)
                    base['verify_publication'](args.publication,conf)
                    ranks=base['live_authority'](conf)
                    old['require_no_slots'](base['JOURNAL'],node,legacy[node],(1,2))
                    old['require_no_slots'](base['JOURNAL'],node,spec['original_GPU'],(0,1,2))
                    value=bound(base,'dispatch',node,spec)(node,program,args.publication,digest(raw),conf,ranks)
        result.append(dict(value,original_parent_context=selected[node]['before']))
    print(json.dumps(dict(claim_SHA256=digest(raw),reports=result,replay_allowed=False),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

"""Renew only two scopes after exact, fresh reconciliation of their no-send."""

import argparse
import ast
from datetime import datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r36-two-warning-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r36_two_warning_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
CPU_LEDGER=('training80-unique-warning-cpu-accounting-24a2a4e8b9a249c4bb84c00bd5656015/receipt.json',
            '0b3090cc941ce2924e5b9b0769536300881aad2f8f770a155a8f24888e4d8ba4')
TERMINAL=('training80-r34-four-new-terminal-owned-workers-b6b49f6339ac4f4a9f0b0ed49b4d78b7.json',
          '102ea6aee031e2ca96d2a8f492e0896746a3afcb121a9530b78e6fd9c0d4278e')
R34=('tools/run_training80_r34_four_terminal_guarded_refill_once.py',
     'd3f72317a56fc31363a14d2b9c017fa612c9781238e1ca3124fd3cd983e5b4e8')
BUDGET=('tools/run_training80_r35_next_three_guarded_successor_once.py',
        'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
SPECS={
 'f13r1n11':('1103606.3299','1103606.3380','c5502f8045f052098543a5d6ef5ce1d1','34ff1d222d356adbbc74d0a22f174542168c12dcc58298869e3fd9e84cd2ab4e'),
 'f13r2n10':('1103606.3305','1103606.3381','c599ef3ad3815a3f978c5be443d4064f','a82e5667abfbd3f4fe7a84604ec082c153f9deb2df1c19f73d7f5ae59d8ad458'),
}
OLD_WARNING='819487f21be5e7eeffc86fdd45d5d28f64adbecc596ef82ba176c8a5bc4b6799'
OLD_DETAIL='65f3a77095fdcab49b34702824d24e5e44299f56107c36f5985aec601edb2c5b'
PIN=r'[a-f0-9]{64}'
PARENT='1103606'


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok,message):
    if not ok:
        raise ValueError(message)


def pin(path,expected=None):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'exact source or receipt path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and (expected is None or digest(raw)==expected),'pinned source or receipt differs')
    return raw


def module(spec):
    relative,expected=spec
    path=ROOT/relative
    pin(path,expected)
    return runpy.run_path(str(path))


def utc_from_beijing(row):
    return row[:7]+[(datetime.fromisoformat(value)-timedelta(hours=8)).isoformat() for value in row[7:]]


def seven(row):
    return row[:4]+row[5:7]+row[8:]


def load():
    previous,budget=module(R34),module(BUDGET)
    first=budget['module'](budget['FIRST'])
    old_read=first['modules']()[1]
    r27,base,config,old_claim,prev,_,legacy_priors,legacy,o01,protected=previous['load']()
    raw=config[6]['controller.py']
    require(raw==pin(ROOT/budget['CONTROLLER'][0],budget['CONTROLLER'][1]),'HOST controller differs')
    frozen=budget['controller'](raw)
    old_ledger=json.loads(pin(LOCAL/CPU_LEDGER[0],CPU_LEDGER[1]))
    require(old_ledger['rc']==0 and old_ledger['remote_outcome']=='RESPONSE_OBSERVED'
            and old_ledger['response']['mutations']==0,'original accounting-only read differs')
    account=old_ledger['response']['nodes']
    observed=json.loads(pin(LOCAL/TERMINAL[0],TERMINAL[1]))['observed']['targets']
    old_workers={x['node']:x for x in observed}
    rows,targets,clear={}, {}, {}
    for node,(step,zero,reject,reply_sha) in SPECS.items():
        prior=old_workers[node]
        launch=old_claim['targets'][node]
        before=prev[node]
        old_scope=f"{previous['STEM']}-{node}-{reject}"
        wire=json.loads(pin(LOCAL/f'{old_scope}-reply.json',reply_sha))['transport']
        body=wire['response']
        [gpu_local]=account[node]['predecessor'];[cpu_local]=account[node]['CPU']
        old_gpu,old_cpu=utc_from_beijing(gpu_local),utc_from_beijing(cpu_local)
        require(old_gpu==before['accounting'] and old_gpu[0]==step and old_gpu[2:4]==['COMPLETED','0:0']
                and old_cpu[:4]==[zero,'training80-g4-fresh-'+reject[:12],'COMPLETED','0:0']
                and old_cpu[5:7]==[node,'cpu=1,mem=256M,node=1'] and old_cpu[7]<=old_cpu[8]
                and account[node]['known_CPU_C0'] is True and account[node]['GPU']==[]
                and wire['rc']==0 and wire['remote_outcome']=='RESPONSE_OBSERVED' and wire['authority_verified'] is wire['remote_received'] is True
                and reject==wire['request']==body['request']==launch['request']
                and body['state']=='GPU_NOT_SENT' and body['GPU_srun_attempted'] is False and body['mutex_acquired'] is body['zero_GRES_call_attempted'] is True
                and body['probe_rc']==0 and body['probe_stderr_SHA256']==OLD_WARNING and body['detail_SHA256']==OLD_DETAIL
                and body['node']==node and body['parent']==PARENT
                and prior['worker_returned'] is True and prior['worker_return_UTC']>=prior['worker_start_UTC']+prior['worker_seconds'],
                'authentic original no-send, actual CPU C0, or worker natural return differs')
        request=uuid5(UUID('3fb5f308-0e91-4e96-896e-4bed1ccd8802'),f'{step}/{reject}/{CPU_LEDGER[1]}').hex
        seconds=frozen['node_seconds'](node,10800,request)
        require(10500<=seconds<=10800,'actual frozen node budget differs')
        rows[node]=dict(node=node,parent=PARENT,previous=step,request=launch['original_request'],new_request=request,
                        rejected_request=reject,rejected_scope=old_scope,rejected_outcome=body,old_GPU=old_gpu,old_CPU=old_cpu,
                        worker_SHA256=prior['worker_SHA256'],expected_SHA256=prior['expected_SHA256'],previous_worker_pid=prior['worker_PID'],
                        cgroup_SHA256=prior['worker_cgroup_SHA256'],terminal_SHA256=prior['terminal_SHA256'],
                        pci=sorted(prior['original_8_PCI']),worker_start_epoch=prior['worker_start_UTC'],worker_seconds=prior['worker_seconds'],
                        planned_end_epoch=prior['worker_start_UTC']+prior['worker_seconds'],prepare_epoch=prior['worker_start_UTC']+prior['worker_seconds']-900,
                        renewal_seconds=seconds)
        targets[node]=dict(launch,request=request,original_zero_GRES=zero)
        clear[node]=old_cpu
    def seconds(node,initial,request):
        r=rows[node]
        require(initial==10800 and request==r['new_request'],'actual worker controller input differs')
        return r['renewal_seconds']
    conf=(*config[:7],seconds)
    claim=dict(scope=STEM,state='PREPARED_ONLY_FRESH_DURABLE_RECONCILIATION_AND_PHYSICAL_REQUIRED',parent=PARENT,
               launcher=WRAPPER.relative_to(ROOT).as_posix(),original_terminal_SHA256=TERMINAL[1],original_accounting_SHA256=CPU_LEDGER[1],
               no_send_warning_SHA256=OLD_WARNING,controller_SHA256=budget['CONTROLLER'][1],replay_allowed=False,ledger_mutations=0,
               max_invocations_per_node=1,GPU_per_node=8,exclude_O01='f12r2n11',
               targets={n:dict(new_request=r['new_request'],rejected_request=r['rejected_request'],prior_GPU=r['previous'],
                               prior_CPU=r['old_CPU'][0],worker_seconds=r['renewal_seconds'],
                               Slurm_time_cap_seconds=r['renewal_seconds']+300,old_reply_SHA256=SPECS[n][3]) for n,r in rows.items()})
    r34_material=(r27,base,conf,dict(targets=targets),prev,clear,legacy_priors,legacy,o01,protected)
    context=(base,conf,{},None,old_read['toolbox'](old_read['R25']),rows,None)
    return previous,budget,first,old_read,r34_material,context,rows,claim


def read_path(row,nonce):
    require(re.fullmatch(r'[a-f0-9]{32}',nonce) and int(nonce,16)>0 and nonce not in (row['new_request'],row['rejected_request']),
            'a fresh nonzero read nonce is required')
    return LOCAL/f"{STEM}-{row['node']}-read-{nonce}.json"


def read_source(material,row,nonce):
    old=material[3]
    signature=old['read_program'](material[5],row,nonce)
    extra=r'''
def qualify(x):
    result=group(x)
    cpu_expected=x['old_CPU']
    cpu=[r for line in query('sacct','-nP','-j',cpu_expected[0],'-o',fmt)
         if len(r:=line.split('|'))==9 and r[0]==cpu_expected[0]]
    rejected=x['rejected_request']
    scope=base/x['rejected_scope']
    outcome,pin=read(scope/'outcome.json')
    target_label='training80-dcu-standby-'+rejected[:12]+'-'+x['node']
    history=query('sacct','-nP','-j',x['parent'],'-o','JobIDRaw,JobName%100')
    old_gpu_rows=[line.split('|')[0] for line in history if line.split('|')[1:2]==[target_label]]
    if not history:
        raise ValueError('cannot establish prior exact job name accounting')
    namespace=os.path.lexists(base/('training80-dcu-standby-'+rejected))
    marker=os.path.lexists(scope/'gpu-send-intent.json')
    result['original_terminal_and_targets_empty']=bool(result['original_terminal_and_targets_empty']
         and result['accounting']==x['old_GPU'] and result['terminal_SHA256']==x['terminal_SHA256']
         and cpu==[cpu_expected] and outcome==x['rejected_outcome'] and not namespace and not marker and not old_gpu_rows)
    result.update(old_CPU=cpu,durable_no_send_SHA256=pin,old_GPU_namespace_exists=namespace,
                  old_GPU_intent_exists=marker,old_GPU_label_rows=old_gpu_rows)
    return result
'''
    signature=old['unique'](signature,'print(json.dumps(dict(read_id=',extra+'\nprint(json.dumps(dict(read_id=')
    signature=old['unique'](signature,'targets=[group(item) for item in items]','targets=[qualify(item) for item in items]')
    compile(signature,'<exact original zero-GRES + no-send read-only>', 'exec')
    return signature


def captured(material,row,nonce,proof,fresh):
    old=material[3]
    accounting,terminal=old['terminal_identity'](row,nonce,proof)
    [value]=proof['observed']['targets']
    seal=value['durable_no_send_SHA256']
    require(proof['rc']==0 and proof['replay_allowed'] is False and proof['stderr_SHA256']==digest(b'')
            and proof['source_SHA256']==digest(read_source(material,row,nonce).encode()) and 0<=proof['RTT']<=45
            and accounting==row['old_GPU'] and terminal==row['terminal_SHA256'] and value['worker_cgroup_SHA256']==row['cgroup_SHA256']
            and value['old_CPU']==[row['old_CPU']] and value['old_GPU_namespace_exists'] is value['old_GPU_intent_exists'] is False
            and value['old_GPU_label_rows']==[] and re.fullmatch(PIN,seal or ''),'fresh immutable old accounting and sealed no-send not proved')
    if fresh:
        now=time.monotonic()
        require(0<=now-proof['received_monotonic']<=45 and abs(time.time()-now-proof['anchor'])<2,'local receipt monotonic window ended')
    return seal


def rendered(material,row,seal,revision):
    previous,budget,first=material[:3]
    fn=first['_fn']
    guard,bound=fn(previous['rendered'],STEM=STEM)(material[4],row['node'],revision)
    rep=material[3]['unique']
    script=guard['launch']
    old_label=f"training80-dcu-standby-{row['rejected_request'][:12]}-{row['node']}"
    script=rep(script,'([old_label], [gpu_label])',f'([old_label], [gpu_label], [{old_label!r}])')
    anchor='            check_previous(account(PREVIOUS))'
    check=(
        f"            no_send_scope=RESULTS/{row['rejected_scope']!r}\n"
        "            no_send_path=no_send_scope/'outcome.json'\n"
        "            require(no_send_path.resolve(strict=True)==no_send_path and no_send_path.stat().st_uid==os.getuid(), 'prior no-send owner or path changed')\n"
        "            with no_send_path.open('rb') as old_no_send_handle:\n"
        "                old_no_send_bytes=old_no_send_handle.read(8193)\n"
        f"            require(0<len(old_no_send_bytes)<=8192 and sha256(old_no_send_bytes).hexdigest()=={seal!r}\n"
        f"                    and json.loads(old_no_send_bytes)=={row['rejected_outcome']!r}, 'immutable original no-send changed')\n"
        f"            require(not os.path.lexists(RESULTS/{('training80-dcu-standby-'+row['rejected_request'])!r})\n"
        "                    and not os.path.lexists(no_send_scope/'gpu-send-intent.json'), 'original GPU namespace or send intent appeared')\n"
        f"            require(account({row['old_CPU'][0]!r})=={seven(row['old_CPU'])!r}, 'previous warning probe not exact C0')\n")
    script=rep(script,anchor,anchor+'\n'+check.rstrip('\n'))
    require(script.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<script.index('no_send_scope=RESULTS')
            <script.index('outcome["zero_GRES_call_attempted"] = True'),'no-send gate is outside actual lock or fresh check')
    guard['launch']=budget['host_budget_guard'](script,rep)
    return guard,bound


def observe_predecessor(material,row,nonce):
    old,first=material[3],material[2]
    call=first['_fn'](old['observe_predecessor'],
                      read_program=lambda ctx,item,identifier:read_source(material,item,identifier),read_path=read_path,
                      runpy=SimpleNamespace(run_path=first['predecessor_reader_helper']))
    return call(material[5],row,nonce)


def dispatch(material,row,revision,receipt):
    previous,budget,first,_,view=material[:5]
    fn=first['_fn']
    node=row['node'];request=row['new_request']
    fn(previous['publication'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)(view,revision)
    git=['git','--no-optional-locks','-c','safe.directory='+ROOT.as_posix()]
    for file,sha in (R34,BUDGET,budget['CONTROLLER']):
        published=subprocess.check_output([*git,'cat-file','blob',revision+':'+file],
                                          cwd=ROOT,stderr=subprocess.PIPE,timeout=8)
        require(published==pin(ROOT/file,sha),'published strict predecessor or worker budget source differs')
    previous['local_fences'](view,node)
    nonce,sha=receipt
    proof=json.loads(pin(read_path(row,nonce),sha))
    guard,bound=rendered(material,row,captured(material,row,nonce,proof,True),revision)
    base,config=view[1:3]
    spec=view[3]['targets'][node]
    locked=bound(base,'observe',node,spec)
    intent,token,outcome=locked.__globals__['paths'](node,request)
    require(not any(os.path.lexists(p) for p in (intent,token,outcome)),'exact successor already attempted; reconcile original')
    source=config[0]
    helper=runpy.run_path(str(source['HELPER']))
    sender=runpy.run_path(str(source['TRANSPORT']))
    argv=helper['ssh'](ROOT.parent)
    require(0<=time.monotonic()-proof['received_monotonic']<=45,'physical reconciliation aged before exact send reservation')
    saved=dict(node=node,request=request,publication=revision,claim_SHA256=digest(pin(CLAIM)),term_read=nonce,term_SHA256=sha,
               program_SHA256=digest(guard['launch'].encode()),GPU_requested=8,replay_allowed=False)
    config[1]['save'](intent,saved)
    response=sender['send'](argv,request=request,program=guard['launch'],data=b'',send_token=token,authority_seconds=18,outcome_seconds=82)
    config[1]['save'](outcome,dict(saved,transport=response))
    return dict(receipt=outcome.name,SHA256=digest(pin(outcome)),state=(response.get('response') or {}).get('state'),replay_allowed=False)


def observe(material,row):
    previous,_,first,_,view=material[:5]
    base,config=view[1:3]
    node,request=row['node'],row['new_request']
    bound=first['_fn'](view[0]['bound'],STEM=STEM)
    observer=bound(base,'observe',node,view[3]['targets'][node])
    intent,token,_=observer.__globals__['paths'](node,request)
    require(token.is_file(),'only prior exact sent request may be observed')
    proof=json.loads(pin(intent));nonce=proof['term_read']
    read=json.loads(pin(read_path(row,nonce),proof['term_SHA256']))
    guard,_=rendered(material,row,captured(material,row,nonce,read,False),proof['publication'])
    require(digest(guard['launch'].encode())==proof['program_SHA256'],'reserved exact program differs')
    return observer(node,guard,config)


def no_send_fixture(launch,row,raw):
    tree=ast.parse(launch)
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_guarded')
    scope=next(n for n in ast.walk(guard) if isinstance(n,ast.With)
               and any(isinstance(item.optional_vars,ast.Name) and item.optional_vars.id=='lock' for item in n.items))
    statements=[ast.unparse(line) for line in scope.body]
    lo=next(i for i,line in enumerate(statements) if line.startswith('no_send_scope ='))
    hi=next(i for i,line in enumerate(statements) if 'previous warning probe not exact C0' in line)
    exact=compile(ast.Module(body=scope.body[lo:hi+1],type_ignores=[]),'<actual locked original no-send check>', 'exec')
    for scenario in ('match','drift','prior-GPU','prior-intent','CPU'):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            scope=root/row['rejected_scope']
            scope.mkdir()
            outcome=scope/'outcome.json'
            outcome.write_bytes(raw+b' ' if scenario=='drift' else raw)
            if scenario=='prior-GPU':
                (root/('training80-dcu-standby-'+row['rejected_request'])).mkdir()
            if scenario=='prior-intent':
                (scope/'gpu-send-intent.json').touch()
            owner=SimpleNamespace(path=os.path,getuid=lambda:outcome.stat().st_uid)
            try:
                exec(exact,dict(RESULTS=root,os=owner,require=require,sha256=sha256,json=json,
                                account=lambda step:seven(row['old_CPU']) if scenario!='CPU' else None))
            except ValueError as error:
                expected={'drift':'immutable original no-send changed','prior-GPU':'original GPU namespace or send intent appeared',
                          'prior-intent':'original GPU namespace or send intent appeared','CPU':'previous warning probe not exact C0'}
                require(expected.get(scenario)==str(error),'wrong independently tested no-send veto')
            else:
                require(scenario=='match','old no-send guard accepted incorrect proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=SPECS)
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and fresh predecessor receipt required')
    material=load()
    raw=pin(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==material[-1],'exact two-scope claim changed')
    rows=material[-2]
    if args.check:
        nodes={}
        for node,row in rows.items():
            read=read_source(material,row,'4'*32)
            original=json.dumps(row['rejected_outcome'],sort_keys=True).encode()
            program,_=rendered(material,row,digest(original),None)
            no_send_fixture(program['launch'],row,original)
            seconds=material[1]['spawn_fixture'](program['launch'],row,material[4][2][6]['controller.py'])
            material[4][0]['offline_staging_fixture'](program['launch'],material[4][2][6])
            nodes[node]=dict(request=row['new_request'],worker_seconds=seconds,compiled_read_SHA256=digest(read.encode()),
                             compiled_HOST_SHA256=digest(program['launch'].encode()))
        result=dict(nodes=nodes,state='PREPARED_ONLY',actual_HOST_spawn_positive=2,actual_HOST_spawn_vetoes=4,
                    staging_positive=2,staging_vetoes=4,locked_no_send_positive=2,locked_no_send_vetoes=8,
                    SSH=0,Git=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=observe_predecessor(material,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(material,rows[args.node])
    else:
        result=dispatch(material,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=digest(raw)),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

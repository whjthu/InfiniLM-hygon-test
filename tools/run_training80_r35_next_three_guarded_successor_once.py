"""Guarded rollover for three incumbents with the actual controller worker budget."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r35-next-three-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r35_next_three_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
FIRST = ('tools/run_training80_r35_first_four_guarded_successor_once.py',
         '50518c96ab1a752c0a9272482074b5d67def57bb2945a7ab7a98b21e6154f609')
G34 = ('tools/run_training80_r25_next_ordinary_guarded_refill_once.py',
       'aa27c6421012b179b3587f08b02064649250880d06ecce28f2f1937e625d2af4')
CONTROLLER = ('code/bw1000/fleet-dcu-standby-v2/controller.py',
              'be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba')
SPECS = {
    'f14r3n06': ('1194900.1329', '1194900.1328', '5f3069f26ddd7d994c5ca53b43ed4f142de8797cef9fec185e9fb0785008141b'),
    'f13r1n05': ('1103606.3315', '1103606.3314', '8e9af53e985ecda2750620f23ecd18164e693a388157eff8c9a64b467984364a'),
    'f12r3n02': ('1194900.1327', '1194900.1326', 'b4868f20143bea76b2b5e4dd4ed96794482c8260ead6cd4bb2255b639c100258'),
}


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pinned(path, expected):
    require(path.resolve(strict=True)==path, 'exact pinned path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==expected, 'exact pinned bytes differ')
    return raw


def module(spec):
    file,sha=spec
    p=ROOT/file
    read_pinned(p,sha)
    return runpy.run_path(str(p))


def controller(raw):
    require(digest(raw)==CONTROLLER[1], 'frozen controller bytes differ')
    namespace={'__name__':'frozen_next_three_standby_v2'}
    # The local Windows budget calculation and stubbed spawn do not use fcntl.
    # The HOST keeps the original unmodified bytes including that import.
    parsed=ast.parse(raw)
    linux=[item for item in parsed.body if isinstance(item,ast.Import) and ast.unparse(item)=='import fcntl']
    [import_linux]=linux
    parsed.body.remove(import_linux)
    exec(compile(parsed,'<actual pinned standby controller>', 'exec'),namespace)
    return namespace


def load():
    first=module(FIRST)
    ancestor=module(G34)
    r34,early,n02=first['modules']()
    ctx=ancestor['bound_api']()['load']()
    raw_controller=read_pinned(ROOT/CONTROLLER[0],CONTROLLER[1])
    require(ctx[1][6]['controller.py']==raw_controller,'rendered controller source differs')
    frozen=controller(raw_controller)
    cal_path,cal_sha=first['CAL']
    cal=json.loads(read_pinned(LOCAL/cal_path,cal_sha))
    timed=cal['scheduled_preparation_90min'][4:7]
    require(list(SPECS)==[row['node'] for row in timed],'current exact three-node sequence changed')
    rows,targets={},{}
    for evidence in timed:
        node=evidence['node']
        step,cpu,reply_sha=SPECS[node]
        request=evidence['request']
        raw=read_pinned(LOCAL/evidence['counter_receipt'],evidence['counter_receipt_SHA256'])
        counter=json.loads(raw)
        worker=counter['result']['nodes'][node]
        one,two=worker['first'],worker['second']
        require(counter['rc']==0 and counter['replay_allowed'] is False and counter['result']['GPU_steps_created']==0
                and worker['previous']==evidence['step']==step and request==worker['request']==ctx[5][node]['new_request']
                and worker['status']=='RUNNING_WITH_ORIGINAL_BUDGET' and worker['pair_state']=='PASS'
                and len(one)==len(two)==worker['growth_DCU']==8 and set(one)==set(two)
                and all(two[p]['calls']>one[p]['calls'] and two[p]['epoch']>one[p]['epoch'] for p in one)
                and worker['worker_start_epoch']+worker['worker_seconds']==evidence['expected_budget_end_epoch']
                and evidence['expected_budget_end_epoch']-900==evidence['prepare_epoch'],
                'actual exact incumbent worker, counter or timing differs')
        name=f"{ancestor['STEM']}-{node}-{request}-reply.json"
        launch=json.loads(read_pinned(LOCAL/name,reply_sha))['transport']
        result=launch['response']
        clearance=result['fresh_zero_GRES_accounting']
        require(launch['rc']==0 and launch['remote_outcome']=='RESPONSE_OBSERVED'
                and result['GPU_srun_attempted'] is result['mutex_acquired'] is result['zero_GRES_call_attempted'] is True
                and result['probe_stderr_SHA256']==digest(b'') and result['probe_step']==cpu==clearance[0]
                and launch['request']==result['request']==request and clearance[2:6]==['COMPLETED','0:0',node,'cpu=1,mem=256M,node=1']
                and result['receipt']['worker_seconds']==worker['worker_seconds']
                and result['parent']==worker['parent']==step.split('.')[0]==cpu.split('.')[0],
                'pinned old authorization, CPU completed identity or actual worker budget differs')
        new=uuid5(UUID('cce7a795-51eb-4fe6-a05c-b563d1d69491'),f'{node}/{step}/{request}').hex
        seconds=frozen['node_seconds'](node,10800,new)
        require(10500<=seconds<=10800, 'frozen actual successor budget outside range')
        rows[node]=dict(node=node,parent=worker['parent'],previous=step,request=request,new_request=new,clearance=clearance,
                        cgroup_SHA256=None,worker_start_epoch=worker['worker_start_epoch'],worker_seconds=worker['worker_seconds'],
                        worker_SHA256=worker['worker_SHA256'],expected_SHA256=worker['expected_SHA256'],submit_SHA256=worker['submit_SHA256'],
                        previous_worker_pid=worker['worker_PID'],pci=sorted(one),planned_end_epoch=evidence['expected_budget_end_epoch'],
                        prepare_epoch=evidence['prepare_epoch'],renewal_seconds=seconds)
        targets[node]=dict(original_GPU=step,original_zero_GRES=cpu,prior_request=request,new_request=new,parent=worker['parent'],
                           prepare_epoch=evidence['prepare_epoch'],planned_worker_budget_end=evidence['expected_budget_end_epoch'],
                           worker_SHA256=worker['worker_SHA256'],expected_SHA256=worker['expected_SHA256'],
                           submit_SHA256=worker['submit_SHA256'],original_launcher_reply_SHA256=reply_sha,
                           counter_receipt_SHA256=evidence['counter_receipt_SHA256'],worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300)
    claim=dict(scope=STEM,state='PREPARED_ONLY_ACTUAL_C0_AND_FRESH_DEVICE_REQUIRED',
               source_calendar_SHA256=cal_sha,old_G3G4_scope_SHA256=G34[1],transport_guard_SHA256=FIRST[1],controller_SHA256=CONTROLLER[1],
               requested_controller_input_seconds=10800,worker_budget_source='PINNED_CONTROLLER_NODE_SECONDS',
               parent_ids=['1103606','1194900'],GPU_per_node=8,max_invocations_per_node=1,targets=targets,
               launcher=WRAPPER.relative_to(ROOT).as_posix(),ledger_mutations=0,replay_allowed=False)
    return first,r34,early,n02,ctx,rows,claim


def host_budget_guard(source, replace):
    intent='            save(audit / "gpu-send-intent.json",'
    pre=("            require(controller['node_seconds'](NODE, 10800, REQUEST) == WORK_SECONDS, "
         "'frozen controller derived worker budget differs before srun')\n")
    emitted='            outcome.update(state="SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING", receipt=record)'
    after=("            require(record['worker_seconds'] == WORK_SECONDS, "
           "'actual submitted worker budget differs; reconcile exact original')\n")
    source=replace(source,intent,pre+intent)
    source=replace(source,emitted,after+emitted)
    compile(source,'<exact standby controller launch budget>', 'exec')
    return source


def api(material,claim_sha):
    first,r34,early,n02,context,rows,claim=material
    fn=first['_fn']
    early=dict(early)
    original_render=early['render']

    def budget_render(ctx,row,acct,terminal,revision):
        out=fn(original_render,STEM=STEM)(ctx,row,acct,terminal,revision)
        out['launch']=host_budget_guard(out['launch'],early['unique'])
        return out

    # Preserve the exact existing reused reader (with only SSH -v removed) and
    # the G3/G4 adopted ancestry, while giving every successor a new intent path.
    early['render']=budget_render
    clone=fn(first['entry'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,read_program,read,render,dispatch,observe=clone((r34,early,n02,context,rows,claim),claim_sha)
    inherited_publish=dispatch.__globals__['publication']

    def publish(c,revision):
        inherited_publish(c,revision)
        _,publisher=c[3][-2]['modules']()
        for path,sha in (FIRST,G34,CONTROLLER):
            relative=ROOT/path
            require(publisher['git']('cat-file','blob',revision+':'+path)==read_pinned(relative,sha),
                    'published transport, ancestry or worker budget source differs')

    return ctx,read_program,read,render,fn(dispatch,publication=publish),observe


def spawn_fixture(launch,row,raw_controller):
    tree=ast.parse(launch)
    seconds=next(ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign) and ast.unparse(a.targets[0])=='WORK_SECONDS')
    guard=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run_guarded')
    locked=next(node for node in ast.walk(guard) if isinstance(node,ast.With)
                and any(isinstance(item.optional_vars,ast.Name) and item.optional_vars.id=='lock' for item in node.items))
    body=[ast.unparse(line) for line in locked.body]
    start=next(i for i,line in enumerate(body) if 'frozen controller derived worker budget differs before srun' in line)
    end=next(i for i,line in enumerate(body) if 'actual submitted worker budget differs' in line)
    chunk=compile(ast.Module(body=locked.body[start:end+1],type_ignores=[]),'<actual HOST spawn path fixture>', 'exec')
    node,parent,request=row['node'],row['parent'],row['new_request']
    label=f'training80-dcu-standby-{request[:12]}-{node}'
    require(seconds==row['renewal_seconds'], 'rendered WORK_SECONDS differs from exact claim')
    for mode in ('match','wrong-prefix','wrong-receipt'):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            frozen=controller(raw_controller)
            popen, intents=[],[]
            def stub(argv,**ignored):
                popen.append(argv)
                return SimpleNamespace(pid=100)
            frozen['subprocess']=SimpleNamespace(Popen=stub, DEVNULL=-1)
            frozen['Path']=lambda _:SimpleNamespace(read_text=lambda:'100 (stub) '+'0 '*19+'1')
            spawn=frozen['spawn']
            if mode=='wrong-receipt':
                def changed(*args):
                    launched=spawn(*args)
                    return dict(launched,worker_seconds=launched['worker_seconds']+1)
                frozen['spawn']=changed
            env=dict(controller=frozen,NODE=node,PARENT=parent,REQUEST=request,root=root,audit=root,
                     PCI=row['pci'],PUBLICATION=None,require=require,outcome={},gpu_label=label,
                     WORK_SECONDS=seconds+1 if mode=='wrong-prefix' else seconds,save=lambda path,value:intents.append(value))
            try:
                exec(chunk,env)
            except ValueError as e:
                if mode=='wrong-prefix':
                    require('before srun' in str(e) and not popen and not intents and not (root/node).exists(),
                            'wrong booked budget was not rejected before srun')
                elif mode=='wrong-receipt':
                    require('reconcile exact original' in str(e) and len(popen)==len(intents)==1
                            and (root/'submission-receipt.json').is_file(),'actual send receipt drift was not preserved for reconciliation')
                else:
                    raise
            else:
                require(mode=='match' and len(popen)==len(intents)==1,'unexpected HOST spawn fixture outcome')
                for name,key in (('expected.json','seconds'),('submit.json','worker_seconds')):
                    require(json.loads((root/node/name).read_bytes())[key]==seconds,'controller output differs from exact planned seconds')
                require(json.loads((root/'submission-receipt.json').read_bytes())['submit_records'][0]['worker_seconds']==seconds,
                        'same planned seconds missing from durable HOST receipt')
    return seconds


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
    require((args.check or args.node) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and actual old receipt required')
    material=load()
    first,r34,early,_,_,rows,claim=material
    raw=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==claim,'exact three successors changed')
    sha=digest(raw)
    ctx,read_program,read,render,dispatch,observe=api(material,sha)
    if args.check:
        output={}
        for node,row in rows.items():
            source=read_program(ctx,row,'2'*32)
            gpu=[row['previous'],f"training80-dcu-standby-{row['request'][:12]}-{node}",'COMPLETED','0:0',str(row['worker_seconds']),
                 node,early['GPU_TRES'],'2000-01-01T00:00:00','2000-01-01T03:00:00']
            cgroup=json.dumps(dict(node=node,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                   same_UID=True,exact_job_and_step_cgroup=True)).encode()
            guard=render(ctx,row,gpu,('a'*64,digest(cgroup)),None)
            first['cgroup_fixture'](guard['launch'],row,cgroup,require)
            actual_seconds=spawn_fixture(guard['launch'],row,ctx[1][6]['controller.py'])
            output[node]=dict(request=row['new_request'],worker_seconds=actual_seconds,preparation_epoch=row['prepare_epoch'],
                              reader_fixture_SHA256=digest(source.encode()),host_fixture_SHA256=digest(guard['launch'].encode()))
        result=dict(nodes=output,state='PREPARED_ONLY',actual_HOST_spawn_positive=3,actual_HOST_spawn_vetoes=6,
                    locked_cgroup_and_CPU_positive=3,locked_cgroup_and_CPU_vetoes=6,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,rows[args.node],args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,rows[args.node])
    else:
        result=dispatch(ctx,rows[args.node],args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

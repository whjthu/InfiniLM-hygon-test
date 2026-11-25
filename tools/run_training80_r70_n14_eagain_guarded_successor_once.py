"""Renew f13r4n14 once, after independently closing its failed R68 request."""

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
STEM = 'training80-r70-n14-eagain-guarded-bf16-v1'
WRAPPER = ROOT/'tools/run_training80_r70_n14_eagain_guarded_successor_once.py'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
NODE = 'f13r4n14'
R68 = ('tools/run_training80_r68_r54_pair_0427_guarded_successor_once.py',
       '4de30666274739909b78070cc2e07480878f06bf5024977a86d60cea86848329')
R69 = ('local-only/read_training80_r69_n14_r68_eagain_receipt_once.py',
       'a998c6244f762ab6011dbcc9ca0b4c62e20354f6b84109a86d8972b8855b671a')
RECONCILE = ('local-only/training80-r69-n14-r68-eagain-reconciliation-1194900-291cee593c40437bbf296a691939b23c.json',
             '399f42eebb49ed451fe956394ecd7c08bc22002d64db0b5039d71d4b0489bb1a')
R54_REPLY = ('local-only/training80-r54-three-0113-guarded-bf16-v1-f13r4n14-43f57ef2d20f57b1a0c1efd099dd89d2-reply.json',
             '2b2d0533288d2cefba5e3a57492715a32c8e9faf2e0683a47c208795cbd98f2c')
MARKERS = ('fresh-probe-intent.json','fresh-probe.json','gpu-send-intent.json')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    p=ROOT/spec[0]
    require(p.resolve(strict=True)==p,'exact local path differs')
    with p.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'exact local bytes differ')
    return raw


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def load():
    r68=module(R68)
    prior,_,material=r68['entry']()[0]()
    budget,first,r34,early,n02,context,old_rows,shadow,old_claim=material
    old=old_rows[NODE]
    observer=module(R69)
    refused,_=observer['load']()
    packet=json.loads(pin(RECONCILE))
    observed=packet['observed']
    proof,=observed['targets']
    require(packet['rc']==0 and packet['stderr_SHA256']==sha256(b'').hexdigest() and packet['replay_allowed'] is False
            and packet['physical_admissions']==packet['Slurm_mutations']==observed['physical_admissions']==observed['Slurm_mutations']==0
            and packet['selected_nodes']==[NODE] and proof['node']==NODE and packet['parent']==proof['parent']==old['parent']
            and packet['source_SHA256']==sha256(observer['render'](observed['read_id'],refused).encode()).hexdigest()
            and proof['prior_numbered']==old['previous'] and proof['prior_request']==old['new_request']==refused['request']
            and proof['controller_numeric']==proof['queue_numeric']==[]
            and proof['original_no_send_verified'] is proof['intent_matches'] is proof['outcome_matches'] is True
            and proof['original_phase']=='AFTER_MUTEX_BEFORE_ZERO_GRES'
            and proof['physical_clear'] is proof['zero_GRES_call_attempted'] is proof['GPU_srun_attempted'] is proof['gpu_namespace_exists'] is False
            and proof['marker_exists']==dict.fromkeys(MARKERS,False) and proof['exact_cpu_and_gpu_accounting']==dict(GPU=[],zero_GRES=[]),
            'independent exact no-send and original custody differs')
    [account]=proof['original_accounting']
    gpu=old['pinned_GPU']
    require(account[:6]==gpu[:4]+gpu[5:7] and account[6] not in ('','Unknown'),
            'original GPU accounting was not independently repeated')
    old_launch=json.loads(pin(R54_REPLY))['transport']
    reply=old_launch['response']
    require(old_launch['rc']==0 and old_launch['authority_verified'] is old_launch['remote_received'] is True
            and old_launch['remote_outcome']=='RESPONSE_OBSERVED' and old_launch['request']==reply['request']==old['request']
            and reply['node']==NODE and reply['parent']==old['parent'] and reply['probe_step']==old['clearance'][0]
            and reply['fresh_zero_GRES_accounting']==old['clearance']
            and reply['state']=='SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING' and reply['GPU_srun_attempted'] is True,
            'original R54 boot-source launch differs')
    require(all(len(k)==64 and all(c in '0123456789abcdef' for c in k)
                for k in (proof['intent_SHA256'],proof['outcome_SHA256'],reply['probe_stdout_SHA256'])),
            'exact evidence digest differs')
    controller=budget['controller'](context[1][6]['controller.py'])
    new=uuid5(UUID('0f5b2ce8-378e-47d3-8c14-fd324517f18c'),f"{NODE}/{old['previous']}/{old['new_request']}").hex
    seconds=controller['node_seconds'](NODE,10800,new)
    require(new not in {old['request'],old['new_request']} and 10500<=seconds<=10800,'new immutable request or worker budget differs')
    original_scope='training80-r54-three-0113-guarded-bf16-v1-'+NODE+'-'+old['request']
    custody=dict(refused=refused,scope=observer['SCOPE'],intent_SHA256=proof['intent_SHA256'],outcome_SHA256=proof['outcome_SHA256'],
                 boot_source_scope=original_scope,boot_source_stdout_SHA256=reply['probe_stdout_SHA256'])
    target=dict(old_claim['targets'][NODE],new_request=new,new_worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300,
                closed_previous_request=old['new_request'],closed_previous_outcome_SHA256=proof['outcome_SHA256'],
                old_probe_stdout_SHA256=reply['probe_stdout_SHA256'])
    row=dict(old,new_request=new,renewal_seconds=seconds)
    scoped=(*context[:5],{NODE:context[5][NODE]},context[6])
    claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),state='PREPARED_ONLY_EXACT_FRESH_ADMISSION_REQUIRED',
               original_receipt_SHA256=RECONCILE[1],original_natural_scope_SHA256=R68[1],original_no_send_reader_SHA256=R69[1],
               original_R54_reply_SHA256=R54_REPLY[1],targets={NODE:target},priority_scopes_preserved=sorted(set(r68['FENCES'])|{'f12r3n15'}),
               GPU_per_node=8,ready_GPU_count=0,max_concurrent_send=1,max_invocations_per_node=1,ledger_mutations=0,replay_allowed=False)
    return prior,(budget,first,r34,early,n02,scoped,{NODE:row},shadow,claim),custody


def insert_custody(program,proof,row):
    refused=proof['refused']
    old_request=refused['request']
    old_cpu='training80-r25-fresh-'+old_request[:12]
    old_gpu='training80-dcu-standby-'+old_request[:12]+'-'+NODE
    expected_intent={k:refused[k] for k in ('request','node','parent','GPU_requested','replay_allowed',
                                           'zero_GRES_call_attempted','GPU_srun_attempted')}
    expected_intent['state']='PRECHECK'

    def replace(before,after):
        nonlocal program
        require(program.count(before)==1,'known guarded program boundary differs')
        program=program.replace(before,after,1)

    extra=(
        "def closed_artifact(path):\n"
        "    require(path.resolve(strict=True)==path and path.is_file() and path.stat().st_uid==os.getuid(), 'closed artifact owner or path differs')\n"
        "    with path.open('rb') as file:\n"
        "        raw=file.read(16385)\n"
        "    require(0<len(raw)<=16384,'closed artifact bound exceeded')\n"
        "    return json.loads(raw),sha256(raw).hexdigest()\n\n\n"
    )
    replace('def run_guarded():',extra+'def run_guarded():')
    anchor='            outcome["mutex_acquired"] = True\n'
    guard=(
        "            failure_stage='ORIGINAL_NO_SEND_ARTIFACTS'\n"
        f"            old_audit=RESULTS/{proof['scope']!r}\n"
        "            require(old_audit.resolve(strict=True)==old_audit and old_audit.stat().st_uid==os.getuid(), 'closed previous scope owner differs')\n"
        "            old_intent,old_intent_sha=closed_artifact(old_audit/'intent.json')\n"
        "            old_outcome,old_outcome_sha=closed_artifact(old_audit/'outcome.json')\n"
        f"            require(old_intent=={expected_intent!r} and old_intent_sha=={proof['intent_SHA256']!r}\n"
        f"                    and old_outcome=={refused!r} and old_outcome_sha=={proof['outcome_SHA256']!r}, 'original pre-probe no-send outcome changed')\n"
        f"            require(not any(os.path.lexists(old_audit/name) for name in {MARKERS!r})\n"
        f"                    and not os.path.lexists(RESULTS/{('training80-dcu-standby-'+old_request)!r}), 'prior attempt now has a send marker or GPU namespace')\n"
        "            failure_stage='ORIGINAL_R54_BOOT_CUSTODY'\n"
        f"            prior_sample_file,unused_pin=closed_artifact(RESULTS/{proof['boot_source_scope']!r}/'fresh-probe.json')\n"
        "            require(set(prior_sample_file)=={'sample','payload_SHA256'}, 'original boot sample shape changed')\n"
        "            prior_sample=prior_sample_file['sample']\n"
        "            prior_probe_stdout=(json.dumps(prior_sample,sort_keys=True)+'\\n').encode()\n"
        f"            require(prior_sample_file['payload_SHA256']==sha256(prior_probe_stdout).hexdigest()=={proof['boot_source_stdout_SHA256']!r}\n"
        f"                    and prior_sample['request']=={row['request']!r} and prior_sample['step']=={row['clearance'][0]!r}\n"
        "                    and prior_sample['node']==NODE and prior_sample['parent']==PARENT and prior_sample['observer_uid']==os.getuid()\n"
        "                    and prior_sample['own_probe_cgroup'] is prior_sample['global_driver_KFD_queried'] is True\n"
        "                    and prior_sample['first']['PCIs']==prior_sample['second']['PCIs']==PCI\n"
        "                    and re.fullmatch(r'[a-f0-9]{64}',prior_sample['boot_SHA256']),'original R54 bound boot or 8PCI differs')\n"
        "            failure_stage='FIXED_LABEL_ACCOUNTING'\n"
    )
    replace(anchor,anchor+guard)
    replace('in ([old_label], [gpu_label])',f'in ([old_label], [gpu_label], [{old_cpu!r}], [{old_gpu!r}])')
    replace('    save(audit / "intent.json", outcome)\n    try:', '    save(audit / "intent.json", outcome)\n    failure_stage="PRECHECK"\n    try:')
    replace('        with (RESULTS / ".training80-dcu-standby.lock").open("rb") as lock:',
            '        failure_stage="GLOBAL_MUTEX"\n        with (RESULTS / ".training80-dcu-standby.lock").open("rb") as lock:')
    replace('            original_records = {}', '            failure_stage="ORIGINAL_WORKER_FILES"\n            original_records = {}')
    old='            check_previous(account(PREVIOUS))\n            scope()\n            no_foreign_allocation()'
    replace(old,'            failure_stage="ORIGINAL_ACCOUNTING"\n            check_previous(account(PREVIOUS))\n'
                '            failure_stage="TARGET_SCHEDULER_SCOPE"\n            scope()\n'
                '            failure_stage="TARGET_ALLOCATION"\n            no_foreign_allocation()\n'
                '            failure_stage="FRESH_ZERO_GRES"')
    replace('            probe_sample(sample)',
            '            failure_stage="FRESH_PHYSICAL_ACCEPTANCE"\n            probe_sample(sample)\n'
            '            require(sample["boot_SHA256"]==prior_sample["boot_SHA256"], "original and fresh node boot differ")')
    replace('            for _ in range(5):', '            failure_stage="FRESH_ZERO_GRES_ACCOUNTING"\n            for _ in range(5):')
    replace('            scope(own=step)\n            no_foreign_allocation()',
            '            failure_stage="POST_PROBE_SCHEDULER_SCOPE"\n            scope(own=step)\n'
            '            failure_stage="POST_PROBE_ALLOCATION"\n            no_foreign_allocation()')
    replace('            root.mkdir(mode=0o700)', '            failure_stage="GPU_STAGING"\n            root.mkdir(mode=0o700)')
    replace('            record = controller["spawn"]', '            failure_stage="GPU_SRUN"\n            record = controller["spawn"]')
    replace('    reply = run(argv)\n    require(reply.returncode == 0, "scheduler read failed")',
            '    try:\n        reply = run(argv)\n    except BlockingIOError as error:\n'
            '        error.standby_read_command = argv[0]\n        raise\n'
            '    require(reply.returncode == 0, "scheduler read failed")')
    replace('    save(audit / "outcome.json", outcome)',
            '        if isinstance(error,BlockingIOError) and error.errno==11:\n'
            '            command = getattr(error,"standby_read_command",None)\n'
            '            outcome["EAGAIN_stage"] = failure_stage\n'
            '            outcome["EAGAIN_read_command"] = command if command in {"sacct","scontrol","squeue"} else None\n'
            '    save(audit / "outcome.json", outcome)')
    compile(program,'<original no-send, same boot and bounded stage attribution>', 'exec')
    return program


def runtime(previous,material,custody,claim_sha):
    clone=material[1]['_fn']
    inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,source,read,render,dispatch,observe=inherited(material,claim_sha)
    old_publication=dispatch.__globals__['publication']
    def publication(actual,revision):
        old_publication(actual,revision)
        _,publisher=actual[3][-2]['modules']()
        require(publisher['git']('cat-file','blob',revision+':'+R68[0])==pin(R68),'original production guard differs')
    def guarded_render(ctx,row,account,pins,revision):
        result=render(ctx,row,account,pins,revision)
        result['launch']=insert_custody(result['launch'],custody,row)
        return result
    return ctx,source,read,guarded_render,clone(dispatch,publication=publication,render=guarded_render),clone(observe,render=guarded_render)


def fixture(program,row,custody):
    tree=ast.parse(program)
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_guarded')
    locked=next(n for n in ast.walk(guard) if isinstance(n,ast.With)
                and any(isinstance(i.optional_vars,ast.Name) and i.optional_vars.id=='lock' for i in n.items))
    lines=[ast.unparse(n) for n in locked.body]
    start=next(i for i,line in enumerate(lines) if "ORIGINAL_NO_SEND_ARTIFACTS" in line)
    end=next(i for i,line in enumerate(lines) if "FIXED_LABEL_ACCOUNTING" in line)
    accepted=next((n for n in locked.body if 'original and fresh node boot differ' in ast.unparse(n)),None)
    require(accepted is not None and end>start,'same-boot gate absent')
    gate=ast.Module(body=locked.body[start:end],type_ignores=[])
    sample=dict(request=row['request'],step=row['clearance'][0],node=NODE,parent=row['parent'],own_probe_cgroup=True,
                global_driver_KFD_queried=True,observer_uid=1,boot_SHA256='a'*64,first={'PCIs':row['pci']},second={'PCIs':row['pci']})
    sample_pin=sha256((json.dumps(sample,sort_keys=True)+'\n').encode()).hexdigest()
    for n in ast.walk(gate):
        if isinstance(n,ast.Constant) and n.value==custody['boot_source_stdout_SHA256']:
            n.value=sample_pin
    first=compile(gate,'<locked old artifacts fixture>','exec')
    same=compile(ast.Module(body=[accepted],type_ignores=[]),'<fresh same boot fixture>','exec')
    read_rpc=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query')
    def pressure(_argv):
        raise BlockingIOError(11,'Resource temporarily unavailable')
    attribution=dict(run=pressure,require=require)
    exec(compile(ast.Module(body=[read_rpc],type_ignores=[]),'<read RPC attribution fixture>','exec'),attribution)
    try:
        attribution['query']('scontrol','show','step',row['parent'])
    except BlockingIOError as error:
        require(error.errno==11 and error.standby_read_command=='scontrol','original EAGAIN is not correctly attributed')
    else:
        raise ValueError('EAGAIN did not fail closed')
    refused=custody['refused']
    intent={k:refused[k] for k in ('request','node','parent','GPU_requested','replay_allowed',
                                  'zero_GRES_call_attempted','GPU_srun_attempted')}
    intent['state']='PRECHECK'
    with TemporaryDirectory() as folder:
        root=Path(folder).resolve()
        original=root/custody['scope']
        original.mkdir()
        for mode in ('clean','corrupt-outcome','old-marker','old-namespace','corrupt-boot-digest','boot-changed'):
            # Local metadata is simulated. Only the predicates from the actual rendered code execute.
            def read(path):
                if path.name=='intent.json':
                    return intent,custody['intent_SHA256']
                if path.name=='outcome.json':
                    outcome=dict(refused)
                    if mode=='corrupt-outcome':
                        outcome['mutex_acquired']=False
                    return outcome,custody['outcome_SHA256']
                return dict(sample=sample,payload_SHA256='f'*64 if mode=='corrupt-boot-digest' else sample_pin),'a'*64
            class OwnerPath:
                def __init__(self,path): self.path=path
                def __truediv__(self,name): return OwnerPath(self.path/name)
                def __fspath__(self): return str(self.path)
                @property
                def name(self): return self.path.name
                def resolve(self,strict=True): return self
                def stat(self): return SimpleNamespace(st_uid=1)
            def lexists(path):
                return ((mode=='old-marker' and path.name=='fresh-probe-intent.json')
                        or (mode=='old-namespace' and path.name=='training80-dcu-standby-'+refused['request']))
            bindings=dict(RESULTS=OwnerPath(root),NODE=NODE,PARENT=row['parent'],PCI=row['pci'],require=require,
                          closed_artifact=read,json=json,sha256=sha256,re=__import__('re'),
                          os=SimpleNamespace(path=SimpleNamespace(lexists=lexists),getuid=lambda:1),
                          sample={'boot_SHA256':'b'*64 if mode=='boot-changed' else 'a'*64})
            try:
                exec(first,bindings)
                exec(same,bindings)
            except ValueError:
                require(mode!='clean','original expected closure unexpectedly refused')
            else:
                require(mode=='clean','bad old ownership or boot fixture accepted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--show-run-order',action='store_true')
    modes.add_argument('--observe-predecessor')
    modes.add_argument('--publication')
    modes.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=(NODE,))
    parser.add_argument('--terminal-receipt',nargs=2,metavar=('READ_ID','SHA256'))
    args=parser.parse_args()
    local=args.check or args.show_run_order
    require((local or args.node==NODE) and bool(args.publication)==bool(args.terminal_receipt),'one exact node and fresh predecessor are required')
    previous,material,custody=load()
    raw=CLAIM.read_bytes()
    row=material[6][NODE]
    require(json.loads(raw)==material[-1],'exact pending claim differs')
    claim_sha=sha256(raw).hexdigest()
    if args.show_run_order:
        result=dict(state='PREPARED_ONLY',node=NODE,previous=row['previous'],request=row['new_request'],worker_seconds=row['renewal_seconds'])
    else:
        ctx,source,read,render,dispatch,observe=runtime(previous,material,custody,claim_sha)
        if args.check:
            budget,first,_,_,n02=material[:5]
            prior=json.dumps(dict(node=NODE,step=row['previous'],worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                                  same_UID=True,exact_job_and_step_cgroup=True)).encode()
            host=render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],sha256(prior).hexdigest()),None)
            first['cgroup_fixture'](host['launch'],row,prior,require)
            seconds=budget['spawn_fixture'](host['launch'],row,ctx[1][6]['controller.py'])
            fixture(host['launch'],row,custody)
            n02['fixture_holds']()
            result=dict(state='PREPARED_ONLY',cgroup_CPU_positive=1,cgroup_CPU_vetoes=2,HOST_spawn_positive=1,HOST_spawn_vetoes=2,
                        custody_boot_positive=1,custody_boot_vetoes=5,EAGAIN_read_fixture=1,worker_seconds=seconds,request=row['new_request'],
                        read_SHA256=sha256(source(ctx,row,'1'*32).encode()).hexdigest(),SSH=0,Git=0,Slurm=0,GPU=0)
        elif args.observe_predecessor:
            result=read(ctx,row,args.observe_predecessor)
        elif args.observe_original:
            result=observe(ctx,row)
        else:
            result=dispatch(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=claim_sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

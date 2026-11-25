"""Renew three proven ordinary originals, preserving exact prior CPU-only outcomes."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r54-three-0113-guarded-bf16-v1'
WRAPPER = ROOT/'tools/run_training80_r54_three_0113_guarded_successor_once.py'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R49 = ('tools/run_training80_r49_four_2334_guarded_successor_once.py',
       'bd9585250012781d8aff74fd44a6054f674f51eeaa99ec9480f21f0d5ecef505')
R53 = ('tools/run_training80_r53_four_0053_guarded_successor_once.py',
       '09755b53157c3522e0e3a1a77369c678092f25333ec18deeccc4e19c3a97b93f')
READER = ('local-only/read_training80_r54_0113_three_older_ordinary_once.py',
          '73a5da9fa289d326d4acbea006dd73f6589547e6aed35890b8fc427929e379bc')
ORIGINALS = {
    'known-no-send': ('local-only/training80-r54-0113-three-older-ordinary-1194900-d564c13b749742b9ad8f50097469d455.json',
                      'b1dde59118fdd4b451f3ca2590c457b297e061afd7c89def6ba3568c549749fe'),
    'warning-originals': ('local-only/training80-r54-0113-three-older-ordinary-1194900-356333a655c149e2ae9cf42cc60b0773.json',
                          'd0cd7479af4f6bd56349ca8451eea316b31f1683ae63e059464ad927c2fa86d7'),
}
ORDER = ('f12r3n15','f13r4n14','f13r3n09')
UNSENT = 'f12r3n15'
FENCES = {'f13r1n12','f13r3n13','f13r1n19','f13r1n02','f12r2n10','f12r2n11','f13r1n10','f13r1n17',
          'f13r2n03','f13r2n07','f13r1n11','f13r2n10','f12r3n02'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'pinned path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'pinned bytes differ')
    return raw


def warning_proof(old,proof,parent):
    request=old['rejected_request']
    [cpu]=proof['rejected_CPU']
    outcome=old['rejected_outcome']
    digest=proof['durable_no_send_SHA256']
    require(proof['rejected_request']==request and proof['original_CPU_match'] is True
            and proof['durable_no_send_matches'] is True and re.fullmatch(r'[a-f0-9]{64}',digest or '')
            and proof['rejected_GPU_label_rows']==[]
            and proof['rejected_GPU_namespace_exists'] is proof['rejected_GPU_intent_exists'] is False
            and len(cpu)==9 and re.fullmatch(parent+r'\.[1-9][0-9]*',cpu[0])
            and cpu[1:4]==['training80-r25-fresh-'+request[:12],'COMPLETED','0:0']
            and cpu[5:7]==[old['node'],'cpu=1,mem=256M,node=1'] and cpu[7]<=cpu[8] and cpu[8]!='Unknown'
            and outcome['GPU_srun_attempted'] is False and outcome['zero_GRES_call_attempted'] is True,
            'actual old warning CPU C0, durable no GPU and remote namespace proofs differ')
    return dict(request=request,scope=old['rejected_scope'],outcome=outcome,outcome_SHA256=digest,CPU=cpu[:4]+cpu[5:7]+cpu[8:])


def with_locked_warning(program,proof,node,unique):
    gpu='training80-dcu-standby-'+proof['request'][:12]+'-'+node
    root='training80-dcu-standby-'+proof['request']
    cpu=proof['CPU']
    guard=(
        '            old_warning_rows = [r.split("|") for r in query("sacct", "-n", "-P", "-j", PARENT, "-o", "JobIDRaw,JobName%110")]\n'
        '            require(any(len(r)==2 and r[0]==PARENT for r in old_warning_rows), "old warning parent accounting missing")\n'
        f'            require(not any(len(r)==2 and r[1]=={gpu!r} for r in old_warning_rows), "old warning GPU label became visible")\n'
        f'            old_warning_scope=RESULTS/{proof["scope"]!r}\n'
        '            old_warning_path=old_warning_scope/"outcome.json"\n'
        '            require(old_warning_path.resolve(strict=True)==old_warning_path and old_warning_path.stat().st_uid==os.getuid(), "old warning outcome owner or path changed")\n'
        '            with old_warning_path.open("rb") as warning_stream:\n'
        '                old_warning_bytes=warning_stream.read(8193)\n'
        f'            require(0<len(old_warning_bytes)<=8192 and sha256(old_warning_bytes).hexdigest()=={proof["outcome_SHA256"]!r}\n'
        f'                    and json.loads(old_warning_bytes)=={proof["outcome"]!r}, "old warning no-GPU outcome differs")\n'
        f'            require(not os.path.lexists(RESULTS/{root!r}) and not os.path.lexists(old_warning_scope/"gpu-send-intent.json"), "old warning GPU namespace or marker appeared")\n'
        f'            require(account({cpu[0]!r})=={cpu!r}, "old warning CPU did not remain exact C0")\n'
    )
    anchor='            check_previous(account(PREVIOUS))'
    source=unique(program,anchor,guard+anchor)
    require('and not observation.stderr and' in source,'existing warning-free fresh physical check is absent')
    compile(source,'<locked exact warning predecessor closure>', 'exec')
    return source


def warning_fixture(launch,proof,row):
    tree=ast.parse(launch)
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_guarded')
    locked=next(n for n in ast.walk(worker) if isinstance(n,ast.With)
                and any(isinstance(i.optional_vars,ast.Name) and i.optional_vars.id=='lock' for i in n.items))
    lines=[ast.unparse(n) for n in locked.body]
    lo=next(i for i,s in enumerate(lines) if s.startswith('old_warning_rows ='))
    hi=next(i for i,s in enumerate(lines) if 'old warning CPU did not remain exact C0' in s)
    require(hi<next(i for i,s in enumerate(lines) if s.startswith('check_previous(')), 'old warning verification is after fresh probe')
    # Match the actual durable bytes without retaining a copy of the remote
    # file. The SHA predicate receives the canonical test fixture hash.
    block=ast.Module(body=locked.body[lo:hi+1],type_ignores=[])
    raw=json.dumps(proof['outcome'],sort_keys=True).encode()
    test_pin=sha256(raw).hexdigest()
    for item in ast.walk(block):
        if isinstance(item,ast.Constant) and item.value==proof['outcome_SHA256']:
            item.value=test_pin
    executable=compile(block,'<actual warning guard fixture>', 'exec')
    for scenario in ('clean','history','parent','bytes','GPU-intent','GPU-namespace','CPU'):
        with TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            old=root/proof['scope']
            old.mkdir()
            path=old/'outcome.json'
            path.write_bytes(raw+(b' ' if scenario=='bytes' else b''))
            if scenario=='GPU-intent':
                (old/'gpu-send-intent.json').touch()
            if scenario=='GPU-namespace':
                (root/('training80-dcu-standby-'+proof['request'])).mkdir()
            scheduler=[] if scenario=='parent' else [row['parent']+'|parent']
            if scenario=='history':
                scheduler.append(row['parent']+'.1|training80-dcu-standby-'+proof['request'][:12]+'-'+row['node'])
            env=dict(PARENT=row['parent'],RESULTS=root,query=lambda *args:scheduler,require=require,json=json,sha256=sha256,
                     os=SimpleNamespace(path=os.path,getuid=lambda:path.stat().st_uid),
                     account=lambda step:proof['CPU'] if scenario!='CPU' and step==proof['CPU'][0] else None)
            try:
                exec(executable,env)
            except ValueError:
                require(scenario!='clean','valid old CPU warning proof was rejected')
            else:
                require(scenario=='clean','changed warning history was accepted')


def entry():
    pin(R49)
    older=runpy.run_path(str(ROOT/R49[0]))
    r48=older['R48']
    pin(r48)
    p48=runpy.run_path(str(ROOT/r48[0]))
    r47=p48['R47']
    pin(r47)
    p47=runpy.run_path(str(ROOT/r47[0]))
    r46,r39=p47['R46'],p47['R39']
    raw=pin(r46).decode()
    trusted=runpy.run_path(str(ROOT/r46[0]))
    tree=ast.parse(raw)
    env=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
             ORDER=ORDER,FENCES=FENCES,R49=R49,R53=R53,R39=r39,R46=r46,UNSENT=UNSENT,
             warning_proof=warning_proof,warning_fixture=warning_fixture,
             prior_token_absent=older['prior_token_absent'],__file__=__file__)

    def source(name):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(raw,fn)

    def adapt(program,changes):
        for before,after in changes:
            require(program.count(before)==1,'known exact old scope adapter differs')
            program=program.replace(before,after,1)
        return program

    load=source('load')
    begin=load.index('    _, load_following, _')
    end=load.index('    controller = budget',begin)
    load=load[:begin]+(
        "    trial = module(R46)['load']()[2]\n"
        "    run39 = module(R39)['entry']()[0]()\n"
        "    routes = dict(trial[5][5],**{UNSENT:run39[5][5][UNSENT]})\n"
        "    previous_attempts = dict(trial[-3],**{UNSENT:run39[-3][UNSENT]})\n"
        "    require(set(ORDER)<=set(routes)&set(previous_attempts), 'existing exact old owner routes differ')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[end:]
    handling=(
        "            if node==UNSENT:\n"
        "                prior=old['prior_transport_not_sent']\n"
        "                require(prior and previous_attempts[node]['new_request']==request\n"
        "                        and previous_attempts[node]['renewal_seconds']==seconds\n"
        "                        and proof.get('previous_unsent_attempt')==dict(request=prior['request'],\n"
        "                       prior_reply_SHA256=prior['reply_SHA256'],prior_intent_SHA256=prior['intent_SHA256'],\n"
        "                       gpu_label_ids=[],cpu_label_ids=[],gpu_namespace_exists=False,control_namespace_exists=False,\n"
        "                       history_and_namespaces_empty=True),'original positive no-send closure differs')\n"
        "                prior_token_absent(prior)\n"
        "                shadow[node]=prior\n"
        "                denied=None\n"
        "            else:\n"
        "                prior=None\n"
        "                require(previous_attempts[node]['new_request']==old['rejected_request'] and old['mode']=='WARNED',\n"
        "                       'exact signed old warning route differs')\n"
        "                denied=warning_proof(old,proof,parent)\n"
        "            require(files['cgroup']==old['cgroup_SHA256'] and routes[node]['parent']==parent,\n"
        "                    'before-capture cgroup or original owner differs')")
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,read_core,read_base,by_node = reader['load']()\n    inputs = [by_node[n] for n in ORDER]"),
        ("    rows, targets = {}, {}","    rows, targets, shadow = {}, {}, {}"),
        ("    for parent, reference in ORIGINALS.items():","    for group, reference in ORIGINALS.items():\n        parent='1194900'"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = [by_node[n] for n in reader['GROUPS'][group]]"),
        ("source = read_core['render'](read_base,expected,observed['read_id'])",
         "source = reader['render'](read_core,read_base,expected,observed['read_id'])"),
        ("and proof['scope']=='NATURAL'","and proof['scope']==old['mode']"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,\n"
         "                    'old accepted scope route or actual controller budget differs')",handling),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","67d9fa1c-ccf7-4f55-8b34-52062f507239"),
        ("require(new!=request and 10500<=renewal<=10800",
         "require(new not in {request,(prior or denied or {}).get('request')} and 10500<=renewal<=10800"),
        ("previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal)","previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal,prior_warning=denied)"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR')",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',closed_prior_transport=prior,\n"
         "                prior_warning_CPU=denied['CPU'][0] if denied else None,prior_warning_outcome_SHA256=denied['outcome_SHA256'] if denied else None)"),
        ("==7,'fixed seven exclusive new requests differ'","==3 and set(shadow)=={UNSENT},'three exclusive candidate requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}","original_group_receipt_SHA256={g:s[1] for g,s in ORIGINALS.items()}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R49,R53,R39,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,shadow,claim)"),
    ))
    exec(compile(load,'<three natural and exact CPU warning closures>','exec'),env)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        common=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,original_render,dispatch,observe=common(material,claim_sha)
        inherited=dispatch.__globals__['publication']
        ledger=dispatch.__globals__['exact_no_other_claims']
        def publication(actual,revision):
            inherited(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R49,R53,r46,r39):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published exact predecessor route or verified warning guard differs')
        def role(ordinary,authority,row):
            ledger(ordinary,authority,row)
            if row['node'] in material[7]:
                older['prior_token_absent'](material[7][row['node']])
        def render(ctx,row,account,pins,revision):
            payload=original_render(ctx,row,account,pins,revision)
            if row['prior_warning']:
                payload['launch']=with_locked_warning(payload['launch'],row['prior_warning'],row['node'],material[3]['unique'])
            return payload
        return (context,read_source,read,render,
                clone(dispatch,render=render,publication=publication,exact_no_other_claims=role),
                clone(observe,render=render))

    env['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','three authorized originals'),
        ('                cases[node] = dict(request=','''                if row['prior_warning']:
                    warning_fixture(rendered['launch'],row['prior_warning'],row)
                if node in material[7]:
                    previous['shadow_fixture'](rendered['launch'],row,material[7][node])
                cases[node] = dict(request='''),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14,SSH=0',
         'HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows),warning_positive=2,warning_vetoes=12,older_unsent_positive=1,older_unsent_vetoes=5,SSH=0'),
    ))
    exec(compile(main,'<three guarded single sends and offline gate fixtures>','exec'),env)
    return env['load'],runtime,env['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

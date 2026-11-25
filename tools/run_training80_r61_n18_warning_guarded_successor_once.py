"""Renew f12r1n18 after exact closure of the prior CPU-only warning."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r61-n18-warning-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r61_n18_warning_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R57 = ('tools/run_training80_r57_sixteen_0236_guarded_successor_once.py',
       '141bf5485ffed9762c5c59e9647850d44d1440d033a5ebeeced77ff7c4787041')
R56 = ('tools/run_training80_r56_four_0215_guarded_successor_once.py',
       'a7ac9f59a288b6d475eb6a8a314305cf2b9ac07f55ef39fe9afb9998399c9716')
R54 = ('tools/run_training80_r54_three_0113_guarded_successor_once.py',
       '3e72de0c7e52ebdd0eba65e199e934a0fab37a4d6d447c815c0dd275e0f9c941')
R51 = ('tools/run_training80_r51_three_due_guarded_successor_once.py',
       'd9aaff9b5d717722201b7713e5126567a09ce9d441309e71bbf3a2ad01a231f0')
R58 = ('tools/run_training80_r58_four_0253_guarded_successor_once.py',
       '58e108e857cf7f6a7f67fa333597ef194e23d40b9cfcfe2e99a19c5b3bea2831')
R59 = ('tools/run_training80_r59_six_0305_guarded_successor_once.py',
       '4285660a8199d97e4c4d3d6fc756a82fc7292312b8c35a461984c187747d3558')
READER = ('local-only/read_training80_r60_n18_warning_once.py',
          'f76918d4fff7eddf28cb284a4f7b52a0c0859909ac22e71ce228b751e7f59ae1')
ORIGINALS = {
    '1': ('local-only/training80-r60-n18-warning-1194900-c1d038c2a33d494eaa260aaf96a662da.json',
          'a48a00a0ca58fe1d3ab354d1e14ab5729d6261540fa4b341995809804ef256a8'),
}
ORDER = ('f12r1n18',)
FENCES = {'f13r1n02','f13r1n08','f13r1n12','f13r1n19','f12r2n10','f12r2n11','f13r3n13','f12r3n10','f13r2n08'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact frozen path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'exact frozen bytes differ')
    return raw


def entry():
    pin(R47)
    prior=runpy.run_path(str(ROOT/R47[0]))
    pin(R54)
    r54=runpy.run_path(str(ROOT/R54[0]))
    pin(R51)
    r51=runpy.run_path(str(ROOT/R51[0]))
    pin(R56)
    r56=runpy.run_path(str(ROOT/R56[0]))
    pin(R57)
    r57=runpy.run_path(str(ROOT/R57[0]))
    pin(R58)
    r58=runpy.run_path(str(ROOT/R58[0]))
    pin(R59)
    r59=runpy.run_path(str(ROOT/R59[0]))
    blocked = set(r54['ORDER']) | set(r51['ORDER']) | set(r57['ORDER']) | set(r58['ORDER']) | set(r59['ORDER'])
    require(set(ORDER).isdisjoint(blocked) and set(ORDER)<=set(r56['ORDER']),
            'recent exact successor scope overlaps')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R54=R54,R51=R51,R56=R56,R57=R57,R58=R58,R59=R59,R46=r46,warning_guard=r54,__file__=__file__)

    def source(name):
        fn=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name==name)
        return ast.get_source_segment(text,fn)

    def adapt(fn,changes):
        for old,new in changes:
            require(fn.count(old)==1,'inherited exact worker-release source boundary differs')
            fn=fn.replace(old,new,1)
        return fn

    load=source('load')
    lo=load.index('    _, load_following, _')
    hi=load.index('    controller = budget',lo)
    load=load[:lo]+(
        "    _, _, originals47 = module(R47)['entry']()[0]()\n"
        "    _, _, prior_warning = module(R56)['entry']()[0]()\n"
        "    routes, old_workers = originals47[5][5], originals47[-3]\n"
        "    require(set(ORDER)<=set(routes)&set(old_workers)&set(prior_warning[-3]), 'existing accepted owner route differs')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,warning_reader,read_core,read_base,exact,_ = reader['load']()\n    groups = {'1':[exact]}\n    inputs = [exact]"),
        ("    for parent, reference in ORIGINALS.items():","    for group, reference in ORIGINALS.items():\n        parent=groups[group][0]['parent']"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = groups[group]"),
        ("        source = read_core['render'](read_base,expected,observed['read_id'])",
         "        source = warning_reader['render'](read_core,read_base,expected,observed['read_id'])"),
        ("and proof['scope']=='NATURAL'","and proof['scope']=='WARNED'"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,\n"
         "                    'old accepted scope route or actual controller budget differs')",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent "
         "and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds "
         "and prior_warning[-3][node]['new_request']==old['rejected_request'] "
         "and files['cgroup']==old['cgroup_SHA256'], 'original or closed warning owner differs')\n"
         "            denied=warning_guard['warning_proof'](old,proof,parent)"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","b97a110e-1397-4cff-8f8b-d5684b204d9c"),
        ("require(new!=request and", "require(new not in {request,denied['request']} and"),
        ("previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal)",
         "previous_worker_pid=proof['worker_pid'],renewal_seconds=renewal,prior_warning=denied)"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',prior_warning_CPU=denied['CPU'][0],"
         "prior_warning_request=denied['request'],prior_warning_outcome_SHA256=denied['outcome_SHA256']"),
        ("==7,'fixed seven exclusive new requests differ'","==1,'fixed exclusive new request differs'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={r['node']:s[1] for g,s in ORIGINALS.items() for r in groups[g]}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R54,R51,R56,R57,R58,R59,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<one exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R54,R51,R56,R57,R58,R59,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        def guarded_render(ctx,row,account,pins,revision):
            packet=render(ctx,row,account,pins,revision)
            packet['launch']=r54['with_locked_warning'](packet['launch'],row['prior_warning'],row['node'],material[3]['unique'])
            return packet
        return context,read_source,read,guarded_render,clone(dispatch,publication=verify,render=guarded_render),clone(observe,render=guarded_render)

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','one authorized original'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('                cases[node] = dict(request=',
         "                warning_guard['warning_fixture'](rendered['launch'],row['prior_warning'],row)\n"
         "                cases[node] = dict(request="),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14',
         'HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows),locked_warning_positive=1,locked_warning_vetoes=6'),
    ))
    exec(compile(main,'<one guarded singleton CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

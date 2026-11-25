"""Renew six ordinary workers with independently verified natural returns."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r59-six-0305-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r59_six_0305_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R48 = ('tools/run_training80_r48_thirteen_2317_guarded_successor_once.py',
       '25fa7c92430b764838f23fedf94544ddcf4d1bfbe7c11c25f0917d085be702d2')
R56 = ('tools/run_training80_r56_four_0215_guarded_successor_once.py',
       'a7ac9f59a288b6d475eb6a8a314305cf2b9ac07f55ef39fe9afb9998399c9716')
R54 = ('tools/run_training80_r54_three_0113_guarded_successor_once.py',
       '3e72de0c7e52ebdd0eba65e199e934a0fab37a4d6d447c815c0dd275e0f9c941')
R51 = ('tools/run_training80_r51_three_due_guarded_successor_once.py',
       'd9aaff9b5d717722201b7713e5126567a09ce9d441309e71bbf3a2ad01a231f0')
R57 = ('tools/run_training80_r57_sixteen_0236_guarded_successor_once.py',
       '141bf5485ffed9762c5c59e9647850d44d1440d033a5ebeeced77ff7c4787041')
R58 = ('tools/run_training80_r58_four_0253_guarded_successor_once.py',
       '58e108e857cf7f6a7f67fa333597ef194e23d40b9cfcfe2e99a19c5b3bea2831')
READER = ('local-only/read_training80_r59_six_0305_exact_originals_once.py',
          'f0f984131c1f970dafa9324a6fed2da142415653961e7a5b30f8c61f2652980b')
ORIGINALS = {
    '1': ('local-only/training80-r59-six-0305-exact-originals-1194900-d880307c6c87457da1d56f88f96960a7.json',
          '5f4208a3db8dcb0deda58e064443ea1b2c0eb56f04d46ed570c58a11a888f024'),
    '2': ('local-only/training80-r59-six-0305-exact-originals-1103606-bd3595664b90496492d2c02cad82ebc6.json',
          'a397269623413acef0282d673bc535ffbbee1ad59a8ae6d63d0ae4befbaed468'),
    '3': ('local-only/training80-r59-six-0305-exact-originals-1194900-9bcd411eaf304e709ed6b1269b71538d.json',
          '202258f207e0906443b73d5a67ac496976190646dd8e46d43c5362ee1324a21d'),
}
ORDER = ('f12r3n09','f12r3n19','f13r2n09','f12r3n06','f13r2n20','f12r3n16')
FENCES = {'f13r1n02','f13r1n08','f13r1n12','f13r1n19','f12r2n10','f12r2n11','f13r3n13'}


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
    consumed = set(r54['ORDER']) | set(r51['ORDER']) | set(r56['ORDER']) | set(r57['ORDER']) | set(r58['ORDER'])
    require(set(ORDER).isdisjoint(consumed),
            'recent exact successor scope overlaps')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R48=R48,R54=R54,R51=R51,R56=R56,R57=R57,R58=R58,R46=r46,__file__=__file__)

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
        "    _, _, originals48 = module(R48)['entry']()[0]()\n"
        "    routes47, routes48 = originals47[5][5], originals48[5][5]\n"
        "    require(set(routes47).isdisjoint(routes48), 'old natural owner scopes overlap')\n"
        "    routes = dict(routes47, **routes48)\n"
        "    old_workers = dict(originals47[-3], **originals48[-3])\n"
        "    require(set(ORDER)<=set(routes)&set(old_workers), 'existing accepted owner route differs')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,read_core,read_base,groups,_ = reader['load']()\n    inputs = [r for group in groups.values() for r in group]"),
        ("    for parent, reference in ORIGINALS.items():","    for group, reference in ORIGINALS.items():\n        parent=groups[group][0]['parent']"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = groups[group]"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","3e377f5c-4590-49a3-9aa8-cb2f4c45436b"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==6,'fixed six exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={r['node']:s[1] for g,s in ORIGINALS.items() for r in groups[g]}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R48,R54,R51,R56,R57,R58,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<six exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R48,R54,R51,R56,R57,R58,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        return context,read_source,read,render,clone(dispatch,publication=verify),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','six authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<six guarded singleton CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

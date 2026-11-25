"""Renew sixteen ordinary workers with independently verified natural returns."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r57-sixteen-0236-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r57_sixteen_0236_guarded_successor_once.py'
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
READER = ('local-only/read_training80_r57_sixteen_0236_exact_originals_once.py',
          'd14e346a6c6d65b265918d156d0f9b349fba01a2fdcc2e21bfe90d5fa7a84ed6')
ORIGINALS = {
    '1': ('local-only/training80-r57-sixteen-0236-exact-originals-1103606-d9203c4b795f4f8fbb97894796dd8208.json',
          'bd872500a2897f4b458ed343acd585b1d5559ab45727d5a4742e5bb4599d874e'),
    '2': ('local-only/training80-r57-sixteen-0236-exact-originals-1194900-350efc50b890425488ce005d06b22b9a.json',
          '802cc7b8902bfc19e6942e1638fbcd55ba08fa7d53964a91b9aad736a9b25c22'),
    '3': ('local-only/training80-r57-sixteen-0236-exact-originals-1103606-1467527e83cc4d80a2b6ed2856726c69.json',
          '38297466823da4fcf715e49af3cd0d03168e6af528ed4da9b1c75e04c9a411f8'),
    '4': ('local-only/training80-r57-sixteen-0236-exact-originals-1194900-e2b06a64f5274efcba779e953641e0cb.json',
          'a5f9f48a441ca515cfd4f2cea59a2c89a2e781774c45e7c28cf5fbecbe25bc43'),
    '5': ('local-only/training80-r57-sixteen-0236-exact-originals-1194900-e637b9f21b574501a0b4f068f88d1bfa.json',
          '1360728a0f47f0d93790e23ac46d5f954192ba2acfc7698c6f90c912835f171b'),
}
ORDER = ('f13r2n04','f13r3n14','f12r3n12','f13r3n03','f12r1n17','f13r3n10',
         'f13r4n08','f13r4n07','f13r1n04','f12r3n14','f12r3n07','f13r4n15','f12r3n18',
         'f12r3n08','f12r1n16','f12r3n13')
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
    require(set(ORDER).isdisjoint(set(r54['ORDER']) | set(r51['ORDER']) | set(r56['ORDER'])),
            'recent exact successor scope overlaps')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R48=R48,R54=R54,R51=R51,R56=R56,R46=r46,__file__=__file__)

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
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","f47845c9-432a-4d3b-83b0-f12612a31113"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==16,'fixed sixteen exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={r['node']:s[1] for g,s in ORIGINALS.items() for r in groups[g]}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R48,R54,R51,R56,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<sixteen exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R48,R54,R51,R56,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        return context,read_source,read,render,clone(dispatch,publication=verify),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','sixteen authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<sixteen guarded singleton CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

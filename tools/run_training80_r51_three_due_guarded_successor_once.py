"""Renew only the three independently returned workers in their exact owner scopes."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r51-three-due-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r51_three_due_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R48 = ('tools/run_training80_r48_thirteen_2317_guarded_successor_once.py',
       '25fa7c92430b764838f23fedf94544ddcf4d1bfbe7c11c25f0917d085be702d2')
READER = ('local-only/read_training80_r51_three_due_originals_once.py',
          '85e0b8b15258f8d58bb6dd7212cc9a78a038ca88bfe0b39756444811d810d943')
ORIGINALS = {
    'f13r2n06': ('local-only/training80-r51-three-due-originals-1103606-94241b71cde9497db18e14d745dea325.json',
                '9cac91737369be17ab18a2f0b7e3cf06d01970605b22fd3356764ab5132e04aa'),
    'f12r3n10': ('local-only/training80-r51-three-due-originals-1194900-a796268e0ec44721a0173ad32553da04.json',
                '9fb3fc071ae54a76b69810ea519e3caee5cacc25aad7d22951614259a76c4402'),
    'f13r2n08': ('local-only/training80-r51-three-due-originals-1103606-15553eaa83e94b87a46a55a1b2f7fc24.json',
                '89a2ce6e39d9a8335e7ac5a17b195f45eaa0ca645450e11e8a04a9e8f7b84bd0'),
}
ORDER = ('f13r2n06','f12r3n10','f13r2n08')
FENCES = {'f13r1n02','f13r1n12','f13r1n19','f12r2n10','f12r2n11','f13r3n13',
          'f12r3n15','f12r3n19','f12r3n16','f13r2n20','f13r2n09','f12r3n09','f12r3n06'}


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
    pin(R48)
    previous=runpy.run_path(str(ROOT/R48[0]))
    r47=previous['R47']
    pin(r47)
    prior=runpy.run_path(str(ROOT/r47[0]))
    r46,r39=prior['R46'],prior['R39']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R48=R48,R39=r39,__file__=__file__)

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
        "    r39 = module(R39)['entry']()[0]()\n"
        "    only_r39 = 'f13r2n06'\n"
        "    old_workers = dict({only_r39:r39[-3][only_r39]}, **r44['load']()[-2])\n"
        "    routes = dict(context[5],**{only_r39:r39[5][5][only_r39]})\n"
        "    require(set(ORDER)<=set(routes)&set(old_workers), 'existing accepted owner route differs')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,read_core,read_base,by_node = reader['load']()\n    inputs = [by_node[n] for n in ORDER]"),
        ("    for parent, reference in ORIGINALS.items():","    for selected, reference in ORIGINALS.items():\n        parent=by_node[selected]['parent']"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = [by_node[selected]]"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","1b2b69b1-a99f-44ff-a7cc-10f69b1a6fe3"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==3,'fixed three exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={n:s[1] for n,s in ORIGINALS.items()}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R48,R39)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<three exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R48,r46,r39):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        return context,read_source,read,render,clone(dispatch,publication=verify),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','three authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<three guarded singleton CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

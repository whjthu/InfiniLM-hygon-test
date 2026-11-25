"""Renew four ordinary workers with independently verified natural returns."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r56-four-0215-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r56_four_0215_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R54 = ('tools/run_training80_r54_three_0113_guarded_successor_once.py',
       '3e72de0c7e52ebdd0eba65e199e934a0fab37a4d6d447c815c0dd275e0f9c941')
R51 = ('tools/run_training80_r51_three_due_guarded_successor_once.py',
       'd9aaff9b5d717722201b7713e5126567a09ce9d441309e71bbf3a2ad01a231f0')
READER = ('local-only/read_training80_r56_four_after_0215_originals_once.py',
          'f9fee1b9954850c05acfa316760ef01ae23f1f66a3a50e90b46360d0cffa8a5d')
ORIGINALS = {
    ('f13r1n13','f13r1n03'): ('local-only/training80-r56-four-after-0215-originals-1103606-61ad0c2f46154421b554dea6c59b3dc8.json',
                             'df8207fa81551eb6b2c62fe0017d538679f5aeb9c52fde44a9281565b7249487'),
    ('f12r1n18',): ('local-only/training80-r56-four-after-0215-originals-1194900-9ae3ff52371a4043b075ddcf04e9830f.json',
                    '61a5d8ad5cecd0b7fae397cf628bc7cee4ea212d5ea15226d54bbb7c84d5cc07'),
    ('f14r3n03',): ('local-only/training80-r56-four-after-0215-originals-1194900-63a52023d9e544f3b0a56d73cf54fa4e.json',
                    '82e303bc2afe5274de93c79ecc557478dce51375e565395f7bb529fb8c000c68'),
}
ORDER = ('f13r1n13','f13r1n03','f12r1n18','f14r3n03')
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
    require(set(ORDER).isdisjoint(r54['ORDER']) and set(ORDER).isdisjoint(r51['ORDER']),
            'recent exact successor scope overlaps')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R54=R54,R51=R51,R46=r46,__file__=__file__)

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
        "    _, _, previous_material = module(R47)['entry']()[0]()\n"
        "    routes = previous_material[5][5]\n"
        "    old_workers = previous_material[-3]\n"
        "    require(set(ORDER)<=set(routes)&set(old_workers), 'existing accepted owner route differs')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,_,read_core,read_base,by_node = reader['load']()\n    inputs = [by_node[n] for n in ORDER]"),
        ("    for parent, reference in ORIGINALS.items():","    for selected, reference in ORIGINALS.items():\n        parent=by_node[selected[0]]['parent']"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = [by_node[n] for n in selected]"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","6a6d9051-d48b-45da-b7ee-def2685bf115"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==4,'fixed four exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={n:s[1] for group,s in ORIGINALS.items() for n in group}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R54,R51,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<four exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R54,R51,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        return context,read_source,read,render,clone(dispatch,publication=verify),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','four authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<four guarded singleton CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

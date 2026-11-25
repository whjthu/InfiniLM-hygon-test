"""Renew only the six independently returned R45 workers through their adopted owner."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r52-six-0036-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r52_six_0036_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R51 = ('tools/run_training80_r51_three_due_guarded_successor_once.py',
       'd9aaff9b5d717722201b7713e5126567a09ce9d441309e71bbf3a2ad01a231f0')
READER = ('local-only/read_training80_r52_six_0036_originals_once.py',
          '436a0544322d3edd28d13d45553e27c4e1604d651460054d90975ac1b78aaa36')
ORIGINALS = {
    '1194900-A': ('local-only/training80-r52-six-0036-originals-1194900-ca2282c71eaf4bdabbbafc954cd43617.json',
                   'eea721b64ece469d19dc4c579721d84f362d08bed2541fe6c64158ebad08d122'),
    '1103606-A': ('local-only/training80-r52-six-0036-originals-1103606-5fdc15c9586e482a86acca43281acc6b.json',
                   'ff20b2ee03ebfcb4e5f3342f76515a8eb1dade36fae863b0ebd7dda7bf9105c2'),
    '1194900-B': ('local-only/training80-r52-six-0036-originals-1194900-80a0bf43891f4209a3bd2242acc1b38d.json',
                   'ab13e286bfba21e987a0303fc28bbc427924a9a21fa8407fcc21df7ad1f1460f'),
}
ORDER = ('f13r3n04','f14r3n06','f14r3n04','f12r3n02','f13r1n05','f13r2n19')
FENCES = {
    'f13r3n13','f13r1n12','f13r1n19','f13r1n02','f12r2n10','f12r2n11',
    'f13r2n03','f13r2n07','f13r3n09','f13r4n14','f12r3n15','f13r2n06','f12r3n10','f13r2n08',
}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'frozen program path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'frozen program bytes differ')
    return raw


def entry():
    pin(R51)
    r51=runpy.run_path(str(ROOT/R51[0]))
    r48=r51['R48']
    pin(r48)
    p48=runpy.run_path(str(ROOT/r48[0]))
    r47=p48['R47']
    pin(r47)
    p47=runpy.run_path(str(ROOT/r47[0]))
    r46=p47['R46']
    raw=pin(r46).decode()
    frozen=runpy.run_path(str(ROOT/r46[0]))
    scope=dict(frozen,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R51=R51,R48=r48,__file__=__file__)
    tree=ast.parse(raw)

    def source(name):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(raw,fn)

    def adapt(fn,changes):
        for old,new in changes:
            require(fn.count(old)==1,'inherited verified worker-release boundary differs')
            fn=fn.replace(old,new,1)
        return fn

    load=source('load')
    lo=load.index('    _, load_following, _')
    hi=load.index('    controller = budget',lo)
    load=load[:lo]+(
        "    accepted = previous['load']()\n"
        "    context,old_workers = accepted[5],accepted[-3]\n"
        "    require(set(ORDER)==set(context[5])==set(old_workers) and all(\n"
        "        {'ledger_old','ledger_request','published_group','parent'}<=set(context[5][n]) for n in ORDER),\n"
        "        'original six adopted58 owner routes differ')\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()",
         "    _,read_core,read_base,by_node = reader['load']()\n    inputs = [by_node[n] for n in ORDER]"),
        ("    for parent, reference in ORIGINALS.items():",
         "    for group_id, reference in ORIGINALS.items():\n        parent, group_nodes = reader['GROUPS'][group_id]"),
        ("        expected = [row for row in inputs if row['parent']==parent]",
         "        expected = [by_node[n] for n in group_nodes]"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","d596bfd3-0b19-4453-97b5-08889206d965"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==6,'fixed six exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_group_receipt_SHA256={group:s[1] for group,s in ORIGINALS.items()}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R51,R48)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<six exact returned R45 workers in three captured groups>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R51,r48,r46,frozen['PREVIOUS']):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published verified owner or exact predecessor guard differs')
        return context,read_source,read,render,clone(dispatch,publication=verify),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','six authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<six guarded one-shot entry and local fixtures>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

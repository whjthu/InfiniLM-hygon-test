"""Renew the thirteen independently returned workers through the proven ordinary guard."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r48-thirteen-2317-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r48_thirteen_2317_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R40 = ('tools/run_training80_r40_n14_proved_unsent_guarded_successor_once.py',
       '62124ebfb577f9127ab7141296a0baf6d6f9a7a6795ea2097f50a76f6093be0a')
READER = ('local-only/read_training80_r48_2317_thirteen_originals_once.py',
          'd65378af71257cb17cc344a7e9e34357a307d3d55b928a89d9b4ba11182a2105')
ORIGINALS = {
    '1103606': ('local-only/training80-r48-2317-thirteen-originals-1103606-f536e194c83a4f929863d9644be21765.json',
                'dade22ee98f4a5d0997993b343792caa7a1b3cb9a609c3e09221609c01644609'),
    '1194900': ('local-only/training80-r48-2317-thirteen-originals-1194900-c44cf84249e4418f9c47bf1d0f94ffc4.json',
                '138b6622d25befa515004f4c8844b94676ff9516ee12d966b777f66ba64caf10'),
}
ORDER = (
    'f13r1n20','f13r4n07','f13r4n08','f13r1n04',
    'f12r1n16','f12r3n09','f12r3n07','f12r3n08','f12r3n14','f12r3n06','f13r4n15','f12r3n13','f12r3n18',
)
PROTECTED = {
    'f13r1n19','f13r1n12','f12r2n10','f12r2n11','f13r3n13',
    'f13r3n09','f13r4n14','f12r3n17','f12r3n20','f13r4n16','f13r4n17','f13r1n08',
}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'pinned source path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'pinned source bytes differ')
    return raw


def entry():
    pin(R47)
    previous=runpy.run_path(str(ROOT/R47[0]))
    r46=previous['R46']
    raw=pin(r46)
    frozen=runpy.run_path(str(ROOT/r46[0]))
    scope=dict(frozen,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=PROTECTED | set(previous['ORDER']),R47=R47,R40=R40,__file__=__file__)
    text=raw.decode()
    tree=ast.parse(text)

    def source(name):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(text,fn)

    def adapt(fn,changes):
        for old,new in changes:
            require(fn.count(old)==1,'pinned guard adapter boundary differs')
            fn=fn.replace(old,new,1)
        return fn

    load=source('load')
    first=load.index('    _, load_following, _')
    last=load.index('    controller = budget',first)
    load=load[:first]+(
        "    original_owner = module(R40)['load']()\n"
        "    original_routes = original_owner[6][5]\n"
        "    require(set(ORDER)<=set(original_routes) and all(\n"
        "        {'ledger_old','ledger_request','published_group','parent'}<=set(original_routes[n]) for n in ORDER),\n"
        "        'not the existing ordinary58 owner lineage for all thirteen')\n"
        "    context = (*context[:5],{n:original_routes[n] for n in ORDER},context[6])\n"
    )+load[last:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    _,read_core,read_base,inputs = reader['load']()"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(original_routes[node]['parent']==parent and old['known_launcher']['worker_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","d59555ad-d9e1-4f6e-8d87-29da0d9480d6"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==13,'fixed thirteen exclusive new requests differ'"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R40)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<thirteen exact returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        common=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=common(material,claim_sha)
        existing=dispatch.__globals__['publication']
        def publication(actual,revision):
            existing(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R40,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published terminal verifier, accepted owner route or passed guard differs')
        return context,read_source,read,render,clone(dispatch,publication=publication),observe

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','thirteen authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<thirteen same gate one-shot CLI and local fixtures>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

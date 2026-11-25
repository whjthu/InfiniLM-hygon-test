"""Renew only 16 independently returned R39 workers, on their original parent jobs."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r47-sixteen-2208-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r47_sixteen_2208_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R46 = ('tools/run_training80_r46_seven_2137_guarded_successor_once.py',
       '437210faa80d64d792e6b6f4dabb33e54329631af128b342764b85c3fd48f1e4')
R39 = ('tools/run_training80_r39_last40_twenty_three_guarded_successor_once.py',
       '2fa7e7ec704fb2665bf8ea9cb43a07decafe1fd828ee43b21da399bb3efc05d5')
READER = ('local-only/read_training80_r47_2208_sixteen_originals_once.py',
          '8edf396edb83c3f02a4f6ec4733cc6262571cada7492a3c2ba7c38f3b499775d')
ORIGINALS = {
    '1103606': ('local-only/training80-r47-2208-sixteen-originals-1103606-6f06d27379034c32a630101ad7db3bd1.json',
                '1bf237967995c5683660960e94f97dc7e39740c48b7263637f9613fb09a4aca3'),
    '1194900': ('local-only/training80-r47-2208-sixteen-originals-1194900-33bf5541f40d4a92b8f48a8450a7f99a.json',
                '74d6e50bc706d4115ea93443b20cb6d9f151e2d1bf3651a60192335ac0894a4d'),
}
ORDER = (
    'f13r2n04','f13r1n13','f13r2n09','f13r3n14','f13r1n03',
    'f12r1n18','f12r3n15','f12r3n04','f13r3n03','f14r3n03','f12r3n19',
    'f12r3n12','f12r1n17','f12r3n16','f13r2n20','f13r3n10',
)
FENCES = {
    'f13r1n19','f13r1n12','f12r2n10','f12r2n11','f13r3n13',
    'f13r3n09','f13r4n14','f12r3n17','f12r3n20','f13r4n16','f13r4n17','f13r1n08',
}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def exact_bytes(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'frozen owner source path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'frozen owner source bytes differ')
    return raw


def entry():
    raw=exact_bytes(R46)
    frozen=runpy.run_path(str(ROOT/R46[0]))
    source=raw.decode()
    tree=ast.parse(source)
    namespace=dict(frozen,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
                   ORDER=ORDER,FENCES=FENCES,R46=R46,R39=R39,__file__=__file__)

    def extract(name):
        definition=next(fn for fn in tree.body if isinstance(fn,ast.FunctionDef) and fn.name==name)
        return ast.get_source_segment(source,definition)

    def replace_once(function,changes):
        for old,new in changes:
            require(function.count(old)==1,'immutable guard adapter source boundary changed')
            function=function.replace(old,new,1)
        return function

    load=extract('load')
    lo=load.index('    _, load_following, _')
    hi=load.index('    controller = budget',lo)
    load=load[:lo]+(
        "    r39 = module(R39)['entry']()[0]()\n"
        "    prior_routes, old_workers = r39[5][5], r39[-3]\n"
        "    require(set(ORDER)<=set(prior_routes)&set(old_workers), 'not the already accepted R39 owned workers')\n"
        "    context = (*context[:5],{n:prior_routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=replace_once(load,(
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","de59ed26-9ca9-4fc4-a395-c099f63b0b69"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==16,'fixed sixteen exclusive new requests differ'"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R46,R39)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<sixteen exact original terminal evidence bindings>','exec'),namespace)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        base=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,source,read,render,dispatch,observe=base(material,claim_sha)
        inherited=dispatch.__globals__['publication']
        def publication(actual,revision):
            inherited(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R46,R39):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==exact_bytes(spec),
                        'published exact terminal verifier or actual R39 owner source differs')
        return context,source,read,render,clone(dispatch,publication=publication),observe

    namespace['runtime']=runtime
    main=replace_once(extract('main'),(
        ('seven authorized originals','sixteen authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<sixteen same gate local fixtures and one-node CLI>','exec'),namespace)
    return namespace['load'],runtime,namespace['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

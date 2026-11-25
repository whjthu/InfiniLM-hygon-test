"""Renew the two R60 and R61 workers after independently verified natural returns."""

import ast
import json
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r77-r60-r61-two-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r77_r60_r61_two_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R60 = ('tools/run_training80_r60_n10_0316_guarded_successor_once.py',
       '2df4f04d417cfcebee212d3f2ca9449509bec0e41f758b7511c5d76b6759fd79')
R61 = ('tools/run_training80_r61_n18_warning_guarded_successor_once.py',
       '67ee360fc0c2ab5c2e8e4fc418cf9914d4354c950fb1a28e570f0ed02554bbdf')
R75 = ('tools/run_training80_r75_r57_remaining_twelve_guarded_successor_once.py',
       '6f2e8db0e1e90cce833536d6a1d4ea9aa12c3e9a6635897d3e308d47825737ca')
R76 = ('tools/run_training80_r76_r58_r59_ten_guarded_successor_once.py',
       '45d1cf61027eb9d2317ce7b4c7cf58a7a2a86014fe1dad3888e5d4896f7e14dd')
READER = ('local-only/read_training80_r77_r60_r61_two_exact_originals_once.py',
          '5453f2af35257d1b71dea77751a8e54372dc6fe055f8bbf805b90a742c0cb660')
ORIGINALS = {
    '1194900': ('local-only/training80-r77-r60-r61-two-exact-originals-1194900-8bbedd1bd68549cd8ddeb4b44779168e.json',
                '710fd727c53ab59663bb10144c8a20a0377194afe8cfcf94c35f472db6879d85'),
}
FLEET = ('local-only/training80-post-renewal-scheduler-7b05183f0628413f93ff23b5873d8810/receipt.json',
         '8aebb7bdb8d8643eb8cb3fbba271638f42922d7031ca68db4e51d229e5829cc1')
ORDER = ('f12r3n10', 'f12r1n18')
FENCES = {'f13r1n02','f13r1n08','f13r1n12','f13r1n14','f13r1n19','f12r2n10','f12r2n11','f13r3n13','f13r2n07'}



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
    pin(R75)
    pin(R76)
    previous75=runpy.run_path(str(ROOT/R75[0]))
    previous76=runpy.run_path(str(ROOT/R76[0]))
    blocked=set(previous75['ORDER']) | set(previous76['ORDER'])
    require(set(ORDER).isdisjoint(blocked | FENCES), 'recent exact successor scope overlaps')
    fleet=json.loads(pin(FLEET))
    require(fleet['rc']==0 and fleet['authority_verified'] is fleet['remote_received'] is True
            and fleet['response']['snapshot']['numbered_cards']==344
            and set(ORDER)<=set(fleet['response']['snapshot']['no_numbered']['1194900']),
            'exact 06:46 scheduler receipt differs')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R60=R60,R61=R61,R75=R75,R76=R76,FLEET=FLEET,R46=r46,__file__=__file__)

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
        "    _, _, originals60 = module(R60)['entry']()[0]()\n"
        "    _, _, originals61 = module(R61)['entry']()[0]()\n"
        "    routes = dict(originals60[5][5], **originals61[5][5])\n"
        "    old_workers = dict(originals60[-3], **originals61[-3])\n"
        "    require(set(ORDER)==set(routes)==set(old_workers), 'existing exact accepted owner routes differ')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()",
         "    _,read_core,read_base,_,inputs,_,holds = reader['load']()\n"
         "    require(set(ORDER).isdisjoint(holds()), 'current business reservation changed')"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","792cc3c5-975b-422a-871c-bf2770e2a1e8"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==2,'fixed exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={n:ORIGINALS[rows[n]['parent']][1] for n in ORDER}, scheduler_snapshot_SHA256=FLEET[1]"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R60,R61,R75,R76,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<two exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R60,R61,R75,R76,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        def pinned_render(ctx,row,account,pins,revision):
            require(account==row['pinned_GPU'] and pins==(row['pinned_terminal'],row['cgroup_SHA256']),
                    'fresh predecessor differs from originally observed terminal or cgroup bytes')
            return render(ctx,row,account,pins,revision)
        return context,read_source,read,pinned_render,clone(dispatch,publication=verify,render=pinned_render),clone(observe,render=pinned_render)

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','two authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('                rendered = render(ctx,row,row[\'pinned_GPU\'],(row[\'pinned_terminal\'],digest(cgroup)),None)',
         "                render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)\n"
         "                fixture_row=dict(row,cgroup_SHA256=digest(cgroup))\n"
         "                rendered=render(ctx,fixture_row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)"),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<two guarded singletons CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

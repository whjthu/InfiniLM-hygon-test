"""Renew five ordinary R62, R64 and R65 workers after their independently verified natural returns."""

import ast
import json
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r78-r62-r64-r65-five-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r78_r62_r64_r65_five_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R47 = ('tools/run_training80_r47_sixteen_2208_guarded_successor_once.py',
       'b8026c8f7051a8f68e3d15e9c56e2e52a0b970284cb536f399b0634734c7c1d3')
R62 = ('tools/run_training80_r62_r2_pair_0333_guarded_successor_once.py',
       '2637c9eef4288129e270b973b0b9246308d7603d2002e6367fe1d849f9bcff32')
R64 = ('tools/run_training80_r64_r52_pair_0349_guarded_successor_once.py',
       '4cbdcad7720b5ad9f68356aee7c6c3cfef0486a4f696193fbc95936f2fa77ae6')
R65 = ('tools/run_training80_r65_r52_three_0354_guarded_successor_once.py',
       '94c0392cdc42a11b59c4a960ed2437a0c99d798a9307e6a20ff682fd9b847d68')
R75 = ('tools/run_training80_r75_r57_remaining_twelve_guarded_successor_once.py',
       '6f2e8db0e1e90cce833536d6a1d4ea9aa12c3e9a6635897d3e308d47825737ca')
R76 = ('tools/run_training80_r76_r58_r59_ten_guarded_successor_once.py',
       '45d1cf61027eb9d2317ce7b4c7cf58a7a2a86014fe1dad3888e5d4896f7e14dd')
R77 = ('tools/run_training80_r77_r60_r61_two_guarded_successor_once.py',
       'eb75f3012f5dc3b6d927c4966e566cb884e654923767e0551a45129122d24074')
READER = ('local-only/read_training80_r78_r62_r64_r65_five_exact_originals_once.py',
          'd1a6b6fed4a3723eef3a0d9370066192de2b488ac0f97c1e5524ca65646a1017')
ORIGINALS = {
    '1103606': ('local-only/training80-r78-r62-r64-r65-five-exact-originals-1103606-ef9c309a6d0b4945b7b0f43f23df0894.json',
                '6adf6856c68a52834697ccd3ca46f0d2b9bf4122f8026110555f6d0965a08af7'),
    '1194900': ('local-only/training80-r78-r62-r64-r65-five-exact-originals-1194900-e2f110eaf8364ca5b050b156a899cba0.json',
                'afe8f140f96982279a92498e9d043e4db7c9e8487a3eb517c73269ef6be2a94a'),
}
FLEET = ('local-only/training80-post-renewal-scheduler-6a1f7dd9288e438fbf21098255ee54e6/receipt.json',
         'd3681f0dfc2ea617b55ed4068565e1780fb3f425bdbca65fd0eda3f5677e2d19')
ORDER = ('f13r2n06', 'f13r3n04', 'f14r3n06', 'f13r1n05', 'f13r2n19')
FENCES = {'f13r1n02','f13r1n08','f13r1n12','f13r1n14','f13r1n19','f12r2n10','f12r2n11',
          'f13r3n13','f13r2n07','f13r2n08','f14r3n04'}




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
    pin(R77)
    previous75=runpy.run_path(str(ROOT/R75[0]))
    previous76=runpy.run_path(str(ROOT/R76[0]))
    previous77=runpy.run_path(str(ROOT/R77[0]))
    blocked=set(previous75['ORDER']) | set(previous76['ORDER']) | set(previous77['ORDER'])
    require(set(ORDER).isdisjoint(blocked | FENCES), 'recent exact successor scope overlaps')
    fleet=json.loads(pin(FLEET))
    require(fleet['rc']==0 and fleet['authority_verified'] is fleet['remote_received'] is True
            and fleet['response']['snapshot']['numbered_cards']==320
            and set(ORDER)<=set().union(*fleet['response']['snapshot']['no_numbered'].values()),
            'exact 07:03 scheduler receipt differs')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R62=R62,R64=R64,R65=R65,R75=R75,R76=R76,R77=R77,FLEET=FLEET,R46=r46,__file__=__file__)

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
        "    _, _, originals62 = module(R62)['entry']()[0]()\n"
        "    _, _, originals64 = module(R64)['entry']()[0]()\n"
        "    _, _, originals65 = module(R65)['entry']()[0]()\n"
        "    routes = dict(originals62[5][5], **originals64[5][5], **originals65[5][5])\n"
        "    old_workers = dict(originals62[-3], **originals64[-3], **originals65[-3])\n"
        "    require(set(ORDER)<=set(routes)&set(old_workers), 'existing exact accepted owner routes differ')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()",
         "    _,read_core,read_base,_,groups,_,holds = reader['load']()\n"
         "    inputs_by_node={row['node']:row for group in groups.values() for row in group}\n"
         "    require(set(ORDER)==set(inputs_by_node), 'exact assigned read scope differs')\n"
         "    inputs=[inputs_by_node[node] for node in ORDER]\n"
         "    require(set(ORDER).isdisjoint(holds()), 'current business reservation changed')"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","dec04a48-18f4-41a3-8a95-1b39aed49b61"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==5,'fixed exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={n:ORIGINALS[rows[n]['parent']][1] for n in ORDER}, scheduler_snapshot_SHA256=FLEET[1]"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R62,R64,R65,R75,R76,R77,R46)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,{},claim)"),
    ))
    exec(compile(load,'<five exact independently returned worker receipts>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        inherited=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,read_source,read,render,dispatch,observe=inherited(material,claim_sha)
        publication=dispatch.__globals__['publication']
        def verify(actual,revision):
            publication(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R47,R62,R64,R65,R75,R76,R77,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        def pinned_render(ctx,row,account,pins,revision):
            require(account==row['pinned_GPU'] and pins==(row['pinned_terminal'],row['cgroup_SHA256']),
                    'fresh predecessor differs from originally observed terminal or cgroup bytes')
            return render(ctx,row,account,pins,revision)
        return context,read_source,read,pinned_render,clone(dispatch,publication=verify,render=pinned_render),clone(observe,render=pinned_render)

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','five authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('                rendered = render(ctx,row,row[\'pinned_GPU\'],(row[\'pinned_terminal\'],digest(cgroup)),None)',
         "                render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)\n"
         "                fixture_row=dict(row,cgroup_SHA256=digest(cgroup))\n"
         "                rendered=render(ctx,fixture_row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)"),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<five guarded singletons CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()

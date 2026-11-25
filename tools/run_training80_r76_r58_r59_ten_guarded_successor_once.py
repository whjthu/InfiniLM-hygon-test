"""Renew exactly ten R58 and R59 predecessors from the three verified reads."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r76-r58-r59-ten-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r76_r58_r59_ten_guarded_successor_once.py'
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
R72 = ('tools/run_training80_r72_r56_ordinary_three_guarded_successor_once.py',
       'de62a5f6f6fe8c9de560e1b87d9f020fca6c2456aa2900958a23de97d3cb053c')
R64 = ('tools/run_training80_r64_r52_pair_0349_guarded_successor_once.py',
       '4cbdcad7720b5ad9f68356aee7c6c3cfef0486a4f696193fbc95936f2fa77ae6')
R65 = ('tools/run_training80_r65_r52_three_0354_guarded_successor_once.py',
       '94c0392cdc42a11b59c4a960ed2437a0c99d798a9307e6a20ff682fd9b847d68')
R74 = ('tools/run_training80_r74_r57_first_four_guarded_successor_once.py',
       '2897293fbd122b40e29a15ab8a0afa01cf543d473390243049b4930f6c7328a6')
R75 = ('tools/run_training80_r75_r57_remaining_twelve_guarded_successor_once.py',
       '6f2e8db0e1e90cce833536d6a1d4ea9aa12c3e9a6635897d3e308d47825737ca')
READER = ('local-only/read_training80_r76_r58_r59_ten_exact_originals_once.py',
          '5479833ac8cda634b86f026bdbf60036f6c443ee260b8a6ea46f07f38f52998d')
ORIGINALS = {
    (1,('f13r4n05','f13r1n20','f13r2n09')): ('local-only/training80-r76-r58-r59-ten-exact-originals-1103606-7190ca4c33b94f81bc90d8762c7d0f8b.json',
        '0a8f81eef453ca825de73fafe324b946087f63cf7fba39d12b7f80b7a1a43d59'),
    (2,('f14r3n05','f12r3n04','f13r2n20','f12r3n16')): ('local-only/training80-r76-r58-r59-ten-exact-originals-1194900-c2836d19b95446899db3c2465e58e81a.json',
        '96138d26a806fb0950a9f456778d08e641f00ff7e996c87ff52b7aa360fdd214'),
    (3,('f12r3n09','f12r3n06','f12r3n19')): ('local-only/training80-r76-r58-r59-ten-exact-originals-1194900-1f1738625306420d8a1c3f50ef4210e8.json',
        'c01f4c55f930c29a1ed4222e2667bce07625711b29e799fff0dd1164dc46cce9'),
}
ORDER = ('f14r3n05','f12r3n04','f13r4n05','f13r1n20','f13r2n20','f12r3n16','f12r3n09','f12r3n06','f12r3n19','f13r2n09')
FENCES = {'f13r1n02','f13r1n08','f13r1n12','f13r1n19','f12r2n10','f12r2n11','f13r3n13','f12r3n10','f12r1n18',
          'f13r2n08','f13r2n06','f13r3n04','f14r3n04','f13r1n05','f13r2n19','f14r3n06',
          'f13r1n13','f13r1n03','f14r3n03'}


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
    pin(R72)
    r72=runpy.run_path(str(ROOT/R72[0]))
    pin(R65)
    r65=runpy.run_path(str(ROOT/R65[0]))
    pin(R64)
    r64=runpy.run_path(str(ROOT/R64[0]))
    pin(R74)
    first_four=runpy.run_path(str(ROOT/R74[0]))
    pin(R75)
    second_group=runpy.run_path(str(ROOT/R75[0]))
    blocked = (set(first_four['ORDER']) | set(second_group['ORDER']) | set(r54['ORDER']) | set(r51['ORDER'])
               | set(r56['ORDER']) | set(r72['ORDER']) | set(r57['ORDER']) | set(r64['ORDER']) | set(r65['ORDER']))
    require(set(ORDER).isdisjoint(blocked) and set(ORDER)==set(r58['ORDER'])|set(r59['ORDER'])
            and set(r58['ORDER']).isdisjoint(r59['ORDER']),
            'recent exact successor scope overlaps')
    r46=prior['R46']
    raw=pin(r46)
    trusted=runpy.run_path(str(ROOT/r46[0]))
    text=raw.decode()
    tree=ast.parse(text)
    scope=dict(trusted,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R47=R47,R54=R54,R51=R51,R72=R72,R74=R74,R75=R75,R56=R56,R57=R57,R58=R58,R59=R59,R64=R64,R65=R65,R46=r46,__file__=__file__)

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
        "    _, _, originals58 = module(R58)['entry']()[0]()\n"
        "    _, _, originals59 = module(R59)['entry']()[0]()\n"
        "    routes = dict(originals58[5][5], **originals59[5][5])\n"
        "    old_workers = dict(originals58[-3], **originals59[-3])\n"
        "    require(set(ORDER)==set(routes)==set(old_workers), 'existing exact accepted owner routes differ')\n"
        "    context = (*context[:5],{n:routes[n] for n in ORDER},context[6])\n"
    )+load[hi:]
    load=adapt(load,(
        ("    read_core,read_base,inputs = reader['load']()","    read_core,read_base,_,groups,_ = reader['load']()\n    by_node = {r['node']:r for group in groups.values() for r in group}\n    inputs = [by_node[n] for n in ORDER]"),
        ("    for parent, reference in ORIGINALS.items():","    for (group,names), reference in ORIGINALS.items():\n        parent=groups[group][0]['parent']"),
        ("        expected = [row for row in inputs if row['parent']==parent]","        expected = [by_node[n] for n in names]\n        require(set(names)<={row['node'] for row in groups[group]}, 'exact per-parent read sub-scope differs')"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",
         "            require(routes[node]['parent']==parent and old_workers[node]['parent']==parent and old_workers[node]['new_request']==request and old_workers[node]['renewal_seconds']==seconds,"),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","7dfe2ca6-fac1-4b6f-ac94-13aa1b2132db"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR'",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR'"),
        ("==7,'fixed seven exclusive new requests differ'","==10,'fixed exclusive new requests differ'"),
        ("original_parent_receipt_SHA256={p:s[1] for p,s in ORIGINALS.items()}",
         "original_exact_node_receipt_SHA256={n:s[1] for (_,names),s in ORIGINALS.items() for n in names}"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R47,R54,R51,R72,R74,R75,R56,R57,R58,R59,R64,R65,R46)"),
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
            for spec in (R47,R54,R51,R72,R74,R75,R56,R57,R58,R59,R64,R65,r46):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'passed prior guard or exact adopted owner source differs')
        def pinned_render(ctx,row,account,pins,revision):
            require(account==row['pinned_GPU'] and pins==(row['pinned_terminal'],row['cgroup_SHA256']),
                    'fresh predecessor differs from originally observed terminal or cgroup bytes')
            return render(ctx,row,account,pins,revision)
        return context,read_source,read,pinned_render,clone(dispatch,publication=verify,render=pinned_render),clone(observe,render=pinned_render)

    scope['runtime']=runtime
    main=adapt(source('main'),(
        ('seven authorized originals','four authorized originals'),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('                rendered = render(ctx,row,row[\'pinned_GPU\'],(row[\'pinned_terminal\'],digest(cgroup)),None)',
         "                render(ctx,row,row['pinned_GPU'],(row['pinned_terminal'],row['cgroup_SHA256']),None)\n"
         "                fixture_row=dict(row,cgroup_SHA256=digest(cgroup))\n"
         "                rendered=render(ctx,fixture_row,row['pinned_GPU'],(row['pinned_terminal'],digest(cgroup)),None)"),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14','HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows)'),
    ))
    exec(compile(main,'<four guarded singletons CLI and local test entry>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()
